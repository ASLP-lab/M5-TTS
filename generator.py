"""M5 inference with local model code and the Apache-licensed Qwen 12Hz codec."""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch
from torch import nn
import torchaudio

from m5.model.model import CSM, ModelArgs
from conditioning import (
    encode_conditioned_text,
    format_conditioned_text,
    load_conditioning_tokenizer,
    load_conditioning_weights,
    resolve_tokenizer_path,
    validate_conditioning_bundle,
)

from pathlib import Path
from m5.codec_qwen12hz import Qwen3TTSTokenizer
from transformers.cache_utils import DynamicCache
import time
from safetensors.torch import load_file


@dataclass
class Segment:
    speaker: int
    text: str
    audio: torch.Tensor
    accent: str
    lang: str = "None"
    caption: str = "None"


def load_llama3_tokenizer():
    """Compatibility wrapper around the shared training/inference tokenizer."""
    return load_conditioning_tokenizer()


def load_inference_tokenizer(ckpt_path):
    """Load the tokenizer bundle saved with a CPT checkpoint or its experiment."""
    checkpoint = Path(ckpt_path)
    checkpoint_dir = checkpoint if checkpoint.is_dir() else checkpoint.parent
    bundle_dir = checkpoint_dir
    for candidate in (checkpoint_dir, checkpoint_dir.parent):
        has_metadata = (candidate / "conditioning.json").is_file()
        has_tokenizer = (candidate / "tokenizer").is_dir()
        if has_metadata != has_tokenizer:
            raise ValueError(f"Incomplete conditioning bundle: tokenizer/ and conditioning.json must be paired in {candidate}")
        if has_metadata:
            bundle_dir = candidate
            break
    else:
        raise FileNotFoundError(f"Missing conditioning bundle for checkpoint: {ckpt_path}; use a CPT checkpoint with tokenizer/ and conditioning.json")
    resolved = resolve_tokenizer_path(ckpt_path)
    tokenizer = load_conditioning_tokenizer(resolved)
    validate_conditioning_bundle(bundle_dir, tokenizer)
    return tokenizer


class Generator:
    def __init__(
        self,
        model: CSM,
        use_codeboks: int,
        text_tokenizer=None,
        speech_tokenizer_path="ASLP-lab/M5-TTS-Audio-Tokenizer",
        local_files_only=False,
    ):
        self._model = model

        self._text_tokenizer = text_tokenizer if text_tokenizer is not None else load_llama3_tokenizer()

        device = next(model.parameters()).device

        self._audio_tokenizer = Qwen3TTSTokenizer.from_pretrained(
            str(speech_tokenizer_path),
            device_map=device,
            local_files_only=local_files_only,
        )
        self.audio_num_codebooks = use_codeboks

        self.sample_rate = 24000
        self.device = device


    def _tokenize_text_segment(self, text: str, speaker: int, accent: str, lang: str = "None", caption: str = "None") -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []
        print(format_conditioned_text(text, speaker=speaker, accent=accent, lang=lang, caption=caption))
        text_tokens = encode_conditioned_text(
            self._text_tokenizer, text, speaker=speaker, accent=accent, lang=lang, caption=caption
        )
        text_frame = torch.zeros(len(text_tokens), self.audio_num_codebooks + 1).long()
        text_frame_mask = torch.zeros(len(text_tokens), self.audio_num_codebooks + 1).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        audio = audio.to(self.device)
        # Qwen encode: audio as numpy array, returns enc.audio_codes[0] shape (T, K)
        audio_np = audio.cpu().numpy()
        enc = self._audio_tokenizer.encode(audio_np, sr=self.sample_rate)
        audio_tokens = enc.audio_codes[0].T  # (K, T), match Mimi format, K=16 for Qwen 12Hz
        audio_tokens = audio_tokens.to(self.device)
        # Truncate codebooks to match self.audio_num_codebooks (Qwen always outputs 16)
        audio_tokens = audio_tokens[:self.audio_num_codebooks, :]
        num_channel = audio_tokens.shape[0]
        audio_tokens += 1  # because +1 when training
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), num_channel + 1).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), num_channel + 1).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (seq_len, 17), (seq_len, 17)
        """
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker, segment.accent, segment.lang, segment.caption)

        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)


    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
        max_seq_len: int = 2048,
    ) -> torch.Tensor:
        max_audio_frames = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []

        
        if text == '': # icl处理
            for segment in context:
                segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
                tokens.append(segment_tokens)
                tokens_mask.append(segment_tokens_mask)
        else:
            gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker, context[0].accent, context[0].lang, context[0].caption)
            tokens.append(gen_segment_tokens)
            tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device) # [t, c]
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device) # [t, c]
        text_len = torch.argmax(prompt_tokens_mask[:,0].float())
        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0) # [1, t, c]
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device) # [1, t, c]

        max_seq_len = max_seq_len - max_audio_frames
        if curr_tokens.size(1) >= max_seq_len:
            raise ValueError(f"Inputs too long, must be below max_seq_len - max_audio_frames: {max_seq_len}")
        if text == '':
            curr_tokens = curr_tokens[:,:-1,:]
            curr_tokens_mask = curr_tokens_mask[:,:-1,:]
            curr_pos = curr_pos[:,:-1]
        else:
            pass

        min_tokens = 6
        first_flag = True 
        cache = DynamicCache(config=self._model.backbone.config) # create cache


        start_time = time.time()
        for _ in range(max_audio_frames):
            has_tokens_num = len(samples)
            if has_tokens_num <= min_tokens:
                need_finish = False
            else:
                need_finish = True


            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, cache, curr_pos, temperature, topk, need_finish=need_finish, do_sample=True)

            if torch.all(sample == 0):
                break  # eos
            samples.append(sample - 1)
            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1


        end_time = time.time()
        cost_time = end_time - start_time
        if not samples:
            raise RuntimeError("The model generated no audio frames")
        if not bool(torch.all(sample == 0)):
            import warnings
            warnings.warn("Generation reached max_audio_length_ms without EOS; output may be truncated", RuntimeWarning)
        if text_len != 0:
            
            prompt_audio = (prompt_tokens[text_len:-1,:-1].unsqueeze(0) - 1).permute(0, 2, 1) 
            audio_tokens = torch.stack(samples).permute(1, 2, 0)
            audio_tokens = torch.clamp(audio_tokens, min=0, max=2047)
            prompt_audio_dur = prompt_audio.shape[2] / 12.5
            all_audio_tokens = torch.cat([prompt_audio, audio_tokens], dim=2) # (1, K, T_all)
            # Qwen decode: needs (T, K) per sample
            all_audio_for_qwen = all_audio_tokens.squeeze(0).T  # (T_all, K)
            all_audio_for_qwen = torch.clamp(all_audio_for_qwen, min=0, max=2047)
            wavs, _sr = self._audio_tokenizer.decode({"audio_codes": [all_audio_for_qwen]})
            audio = torch.from_numpy(wavs[0]).to(self.device)

            audio = audio[int(prompt_audio_dur * 24000) :-int(0.1 * 24000)]

        else:
            audio_tokens = torch.stack(samples).permute(1, 2, 0)  # (1, K, T_gen)
            audio_for_qwen = audio_tokens.squeeze(0).T  # (T_gen, K)
            audio_for_qwen = torch.clamp(audio_for_qwen, min=0, max=2047)
            wavs, _sr = self._audio_tokenizer.decode({"audio_codes": [audio_for_qwen]})
            audio = torch.from_numpy(wavs[0]).to(self.device)
            audio = audio[: -int(0.1 * 24000)]



        return audio
    
def load_csm_300M(ckpt_path: str = "ckpt.pt", device: str = "cuda") -> Generator:

    tokenizer = load_inference_tokenizer(ckpt_path)
    model_args = ModelArgs(
        backbone_flavor="llama-250M",
        decoder_flavor="llama-60M",
        text_vocab_size=len(tokenizer),
        audio_vocab_size=2051,
        audio_num_codebooks=16,
        attention_type='flash_attention_2'
    )

    model = CSM(model_args).to(device=device, dtype=torch.bfloat16)
    ckpt = load_file(ckpt_path)
    load_conditioning_weights(model, ckpt, tokenizer, allow_vocab_expansion=False)

    generator = Generator(model, 16, text_tokenizer=tokenizer)
    return generator


def load_csm_130M(ckpt_path: str = "ckpt.pt", device: str = "cuda", use_codebook: int = 16) -> Generator:
    tokenizer = load_inference_tokenizer(ckpt_path)
    model_args = ModelArgs(
        backbone_flavor="llama-100M",
        decoder_flavor="llama-30M",
        text_vocab_size=len(tokenizer),
        audio_vocab_size=2051,
        audio_num_codebooks=16,
        use_codebooks=use_codebook,
        attention_type="flash_attention_2"
    )


    model = CSM(model_args)
    ckpt = load_file(ckpt_path, device="cpu")
    load_conditioning_weights(model, ckpt, tokenizer, allow_vocab_expansion=False)
    model.eval()
    model = model.to(device=device, dtype=torch.bfloat16)
    generator = Generator(model, use_codebook, text_tokenizer=tokenizer)
    return generator



def load_csm_50M(ckpt_path: str = "ckpt.pt", device: str = "cuda", use_codebook: int = 16) -> Generator:
    tokenizer = load_inference_tokenizer(ckpt_path)
    model_args = ModelArgs(
        backbone_flavor="llama-30M",
        decoder_flavor="llama-20M",
        text_vocab_size=len(tokenizer),
        audio_vocab_size=2051,
        audio_num_codebooks=16,
        use_codebooks=use_codebook,
        attention_type="flash_attention_2"
    )

    model = CSM(model_args)
    ckpt = load_file(ckpt_path, device="cpu")
    load_conditioning_weights(model, ckpt, tokenizer, allow_vocab_expansion=False)
    model.eval()
    model = model.to(device=device, dtype=torch.bfloat16)

    generator = Generator(model, use_codebook, text_tokenizer=tokenizer)
    return generator



