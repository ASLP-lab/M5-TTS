"""M5 model and 16-codebook codec bridge for the Cosy-derived GRPO trainer."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import torch
from safetensors.torch import load_file
from transformers.cache_utils import DynamicCache
from .m5_model.model import CSM, ModelArgs
from .m5_conditioning import encode_conditioned_text, load_conditioning_tokenizer, validate_conditioning_bundle


@dataclass
class Segment:
    text: str | None
    speaker: int | str | None
    audio: torch.Tensor | None
    accent: str | None = None
    lang: str | None = None
    caption: str | None = None
    audio_codes: torch.Tensor | None = None


class QwenCodec:
    """Raw codec IDs only at this boundary; model IDs remain intact for replay."""
    def __init__(self, path, device):
        from qwen_tts import Qwen3TTSTokenizer
        config=json.loads((Path(path)/'config.json').read_text())
        assert config['encoder_valid_num_quantizers']==16
        assert config['encoder_config']['codebook_size']==2048
        self.sample_rate=config['output_sample_rate']
        self.frame_samples=config['decode_upsample_rate']
        self.frame_rate=self.sample_rate/self.frame_samples
        self.tokenizer=Qwen3TTSTokenizer.from_pretrained(path,device_map=str(device))
        self.tokenizer.model.eval().requires_grad_(False)

    def eval(self):
        self.tokenizer.model.eval(); return self

    def to(self, device):
        self.tokenizer.model.to(device); return self

    @torch.no_grad()
    def encode(self, audio):
        codes=self.tokenizer.encode(audio.detach().float().cpu().numpy(),sr=self.sample_rate).audio_codes[0]
        return torch.as_tensor(codes,dtype=torch.long)

    @torch.no_grad()
    def decode(self, raw_codes):
        assert raw_codes.ndim==2 and raw_codes.shape[-1]==16
        assert raw_codes.min()>=0 and raw_codes.max()<=2047
        wavs,sr=self.tokenizer.decode({'audio_codes':[raw_codes.long()]})
        assert sr==self.sample_rate
        return torch.as_tensor(np.asarray(wavs[0]),dtype=torch.float32).reshape(-1)


def decode_ids(model_codes):
    # Same codec boundary as M5 inference: shift once, clamp mixed EOS/reserved
    # categories for waveform decode only. Never clamp or shift PPO targets.
    return (model_codes.long()-1).clamp(0,2047)


def is_eos(frame):
    return bool(torch.all(frame==0))


class Generator:
    def __init__(self, model, tokenizer, codec):
        self._model=model
        self._text_tokenizer=tokenizer
        self._audio_tokenizer=codec
        self.device=next(model.parameters()).device
        self.sample_rate=codec.sample_rate

    def text_rows(self, text, segment):
        ids=encode_conditioned_text(self._text_tokenizer,text,speaker=segment.speaker,
                                    accent=segment.accent,lang=segment.lang,caption=segment.caption)
        rows=torch.zeros(len(ids),17,dtype=torch.long,device=self.device)
        masks=torch.zeros_like(rows,dtype=torch.bool)
        rows[:,-1]=torch.tensor(ids,device=self.device);masks[:,-1]=True
        return rows,masks

    def _prepare_prompt(self, text, speaker, context, max_frames, max_seq_len):
        """Build conditioning once per prompt, shared by its candidate batches."""
        tokens, masks = [], []
        if text:
            seg = context[0]
            seg = Segment(text, speaker, None, seg.accent, seg.lang, seg.caption)
            t, m = self.text_rows(text, seg)
            tokens.append(t); masks.append(m)
        else:
            for seg in context:
                t, m = self.text_rows(seg.text or '', seg)
                tokens.append(t); masks.append(m)
                if seg.audio is not None or seg.audio_codes is not None:
                    raw = seg.audio_codes if seg.audio_codes is not None else self._audio_tokenizer.encode(seg.audio)
                    a = raw.to(self.device) + 1
                    a = torch.cat([a, a.new_zeros(1, 16)])
                    a = torch.cat([a, a.new_zeros(len(a), 1)], -1)
                    am = torch.ones_like(a, dtype=torch.bool); am[:, -1] = False
                    tokens.append(a); masks.append(am)
        tokens, mask = torch.cat(tokens), torch.cat(masks)
        if len(tokens) >= max_seq_len - max_frames:
            raise ValueError('Inputs exceed max_seq_len - max_audio_frames')
        audio_rows = tokens[mask[:, 0], :16]
        codec_prefix = decode_ids(audio_rows[:-1]) if len(audio_rows) else None
        if not text:
            if not (bool(mask[-1, :16].all()) and not bool(mask[-1, -1])
                    and bool((tokens[-1, :16] == 0).all())):
                raise ValueError('ICL must end in reference-audio EOS; combine reference and target text first')
            tokens, mask = tokens[:-1], mask[:-1]
        return tokens[None], mask[None], codec_prefix

    @torch.inference_mode()
    def _generate_batch(self, prefix, mask, batch_size, max_frames, temperature, topk, rng, stop, return_metadata=False):
        """Batch dimension is candidates; time/codebook dimensions remain autoregressive."""
        tokens = prefix.expand(batch_size, -1, -1).contiguous()
        mask = mask.expand_as(tokens).contiguous()
        pos = torch.arange(tokens.shape[1], device=self.device)[None].expand(batch_size, -1)
        cache = DynamicCache(config=self._model.backbone.config)
        finished = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
        lengths = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        frames = []
        for frame_index in range(max_frames):
            if stop is not None and stop.is_set():
                raise RuntimeError('Rollout worker is stopping')
            sample = self._model.generate_frame(tokens, mask, cache, pos, temperature, topk,
                need_finish=frame_index > 6, do_sample=True, generator=rng)
            eos = (sample == 0).all(dim=-1)
            valid = ~finished & ~eos
            lengths += valid
            finished |= eos
            if bool(finished.all()):
                break
            # Keep cache rows stable. Finished candidates emit only discarded padding.
            sample = sample.masked_fill(~valid[:, None], 0)
            frames.append(sample)
            tokens = torch.cat([sample, sample.new_zeros(batch_size, 1)], -1)[:, None]
            mask = torch.ones_like(tokens, dtype=torch.bool); mask[:, :, -1] = False
            pos = pos[:, -1:] + 1
        sizes = lengths.tolist()  # One length transfer per candidate batch, not per row/frame.
        if not frames or any(size == 0 for size in sizes):
            raise ValueError('No generated acoustic frames')
        codes = torch.stack(frames, dim=1)
        outputs = [codes[i, :size] for i, size in enumerate(sizes)]
        if return_metadata:
            done = finished.tolist()
            return [(item, dict(has_eos_decision=bool(done[i]),
                                stop_reason='eos' if done[i] else 'max_frames'))
                    for i, item in enumerate(outputs)]
        return outputs

    @torch.inference_mode()
    def _decode_audio(self, codec_prefix, codes):
        raw = decode_ids(codes)
        if codec_prefix is not None:
            audio = self._audio_tokenizer.decode(torch.cat([codec_prefix, raw]))
            audio = audio[len(codec_prefix) * self._audio_tokenizer.frame_samples:]
        else:
            audio = self._audio_tokenizer.decode(raw)
        return audio[:-int(0.1 * self.sample_rate)]  # Preserve original SS tail trim.

    @torch.inference_mode()
    def iter_candidates(self, text, speaker, context, n, batch_size=1,
                        max_audio_length_ms=90_000, temperature=0.9, topk=50,
                        max_seq_len=2048, rng=None, stop=None, return_metadata=False):
        """Yield ordered (audio, codes, prefix) without retaining a whole rollout on GPU.

        Batch only siblings of the same prompt: no cross-prompt padding/group mixing.
        Sampling draws independent noise per row. Reproducibility requires the same
        seed AND batch size; stochastic outputs need not match serial RNG ordering.
        """
        if type(n) is not int or n < 1 or type(batch_size) is not int or not 1 <= batch_size <= 256:
            raise ValueError('Expected positive candidate count and batch size in 1..256')
        max_frames = int(max_audio_length_ms * self._audio_tokenizer.frame_rate / 1000)
        if max_frames < 1:
            raise ValueError('Generation length must contain at least one frame')
        prefix, mask, codec_prefix = self._prepare_prompt(text, speaker, context, max_frames, max_seq_len)
        for start in range(0, n, batch_size):
            size = min(batch_size, n - start)
            if return_metadata:
                codes_batch = self._generate_batch(prefix, mask, size, max_frames, temperature, topk, rng, stop,
                                                   return_metadata=True)
                for codes, metadata in codes_batch:
                    yield self._decode_audio(codec_prefix, codes), codes, prefix, metadata
            else:
                codes_batch = self._generate_batch(prefix, mask, size, max_frames, temperature, topk, rng, stop)
                for codes in codes_batch:
                    yield self._decode_audio(codec_prefix, codes), codes, prefix

    @torch.inference_mode()
    def generate_transformer_icl_ss(self, text, speaker, context, max_audio_length_ms=90_000,
                                    temperature=0.9, topk=50, max_seq_len=2048, rng=None, stop=None):
        """Backward-compatible single-utterance API used by Seed evaluation."""
        stream = self.iter_candidates(text, speaker, context, 1, 1, max_audio_length_ms,
                                      temperature, topk, max_seq_len, rng, stop)
        try:
            return next(stream)
        finally:
            stream.close()


def load_m5_model(cfg,device):
    """Load policy/tokenizer only; remote learners do not allocate a codec."""
    checkpoint=Path(cfg['checkpoint'])
    if checkpoint.is_dir(): checkpoint=checkpoint/'model.safetensors'
    bundle=checkpoint.parent
    if not (bundle/'conditioning.json').is_file(): bundle=bundle.parent
    tokenizer=load_conditioning_tokenizer(str(bundle/'tokenizer'))
    metadata=validate_conditioning_bundle(bundle,tokenizer)
    assert metadata['format']=='m5-caption-accent-speaker-lang-v3'
    args=ModelArgs(backbone_flavor=cfg['backbone_flavor'],decoder_flavor=cfg['decoder_flavor'],
                   text_vocab_size=len(tokenizer),audio_vocab_size=2051,audio_num_codebooks=16,
                   use_codebooks=16,attention_type='sdpa')
    model=CSM(args)
    model.load_state_dict(load_file(str(checkpoint),device='cpu'),strict=True)
    model.eval().to(device=device,dtype=torch.bfloat16)  # Original GRPO precision.
    return model,tokenizer


def load_m5_for_rl(cfg,device):
    model,tokenizer=load_m5_model(cfg,device)
    codec=QwenCodec(cfg['codec_path'],device)
    return Generator(model,tokenizer,codec),tokenizer
