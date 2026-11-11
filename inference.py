"""Self-contained inference API. Checkpoints may be local folders or Hugging Face IDs."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from conditioning import load_conditioning_weights
from generator import Generator, Segment, load_inference_tokenizer
from m5.model.model import CSM, ModelArgs

DEFAULT_MODEL = "ASLP-lab/M5-TTS-50M-VoiceClone"
DEFAULT_CUSTOM_VOICE_MODEL = "ASLP-lab/M5-TTS-50M-CustomVoice"
DEFAULT_CODEC = "ASLP-lab/M5-TTS-Audio-Tokenizer"


def resolve_pretrained(source, *, revision="main", cache_dir=None, local_files_only=False, patterns=None):
    """Download weights/data only; never load Python code from a model repository."""
    path = Path(source).expanduser()
    if path.is_dir():
        return path.resolve()
    if path.exists() or path.is_absolute() or str(source).startswith((".", "~")):
        raise FileNotFoundError(f"Expected a checkpoint directory: {source}")
    return Path(snapshot_download(repo_id=str(source), revision=revision, cache_dir=cache_dir,
                                  local_files_only=local_files_only, allow_patterns=patterns))


def prepare_prompt(path, sample_rate, device="cpu"):
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(samples.mean(axis=1)).to(device)
    if not audio.numel() or not torch.isfinite(audio).all() or audio.abs().max() == 0:
        raise ValueError("Reference audio must be nonempty, finite and non-silent")
    if rate != sample_rate:
        audio = torchaudio.functional.resample(audio, rate, sample_rate)
    audio = audio / (audio.abs().max() + 1e-8)
    threshold = 0.02 if audio.numel() / sample_rate > 10 else 0.015
    values = audio.cpu().numpy()
    speech = np.flatnonzero(np.abs(values) > threshold)
    if not speech.size:
        raise ValueError("Reference audio contains no speech above the silence threshold")
    cuts = np.flatnonzero(np.diff(speech) - 1 >= int(0.15 * sample_rate))
    starts = np.r_[speech[0], speech[cuts + 1]]
    final_end = speech[-1] + 1 if len(values) - speech[-1] - 1 >= int(0.15 * sample_rate) else len(values)
    ends = np.r_[speech[cuts] + 1, final_end]
    padding = int(0.05 * sample_rate)
    return torch.from_numpy(np.concatenate([values[max(0, start - padding):min(len(values), end + padding)]
                                           for start, end in zip(starts, ends)]))


class M5TTS:
    def __init__(self, generator):
        self.generator = generator
        self.sample_rate = generator.sample_rate

    @classmethod
    def from_pretrained(cls, model=DEFAULT_MODEL, *, codec=DEFAULT_CODEC, device="cuda", revision="main",
                        codec_revision="main", cache_dir=None, local_files_only=False, attention="flash_attention_2"):
        """M5 bundle: model.safetensors, config.json, conditioning.json, tokenizer/."""
        if attention not in ("sdpa", "flash_attention_2"):
            raise ValueError("attention must be sdpa or flash_attention_2")
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("The reference inference requires CUDA; no automatic CPU/SDPA fallback is applied")
        bf16_supported = False
        if device.type == "cuda":
            with torch.cuda.device(device):
                bf16_supported = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16_supported else torch.float32
        if attention == "flash_attention_2":
            if device.type != "cuda" or not bf16_supported:
                raise ValueError("The reference flash_attention_2 backend requires a BF16-capable NVIDIA GPU")
            from transformers.utils import is_flash_attn_2_available
            if not is_flash_attn_2_available():
                raise ImportError("Install flash-attn==2.7.3 after PyTorch: python -m pip install --no-build-isolation flash-attn==2.7.3. No automatic SDPA fallback is applied.")
        else:
            import warnings
            warnings.warn("SDPA is an explicit alternative, not the reference FlashAttention 2 backend; generated results may differ", RuntimeWarning)
        bundle = resolve_pretrained(model, revision=revision, cache_dir=cache_dir, local_files_only=local_files_only,
                                    patterns=["model.safetensors", "config.json", "conditioning.json", "tokenizer/*"])
        for name in ("model.safetensors", "config.json", "conditioning.json", "tokenizer/tokenizer_config.json"):
            if not (bundle / name).is_file():
                raise FileNotFoundError(f"Incomplete M5 bundle: {bundle / name}. Architecture config example: m5/configs/m5_config.jsonl")
        config = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
        tokenizer = load_inference_tokenizer(bundle)
        if config.get("text_vocab_size", len(tokenizer)) != len(tokenizer):
            raise ValueError("config.json text_vocab_size does not match the saved tokenizer")
        if config.get("audio_num_codebooks", 16) != 16 or config.get("use_codebooks", 16) != 16:
            raise ValueError("The Qwen 12Hz codec requires all 16 codebooks")
        if config.get("audio_vocab_size", 2051) != 2051:
            raise ValueError("Expected M5 audio_vocab_size=2051")
        args = ModelArgs(backbone_flavor=config["backbone_flavor"], decoder_flavor=config["decoder_flavor"],
                         text_vocab_size=len(tokenizer), audio_vocab_size=2051, audio_num_codebooks=16,
                         use_codebooks=16, attention_type=attention)
        network = CSM(args)
        weights = load_file(str(bundle / "model.safetensors"), device="cpu")
        load_conditioning_weights(network, weights, tokenizer, allow_vocab_expansion=False)
        del weights
        codec_path = resolve_pretrained(codec, revision=codec_revision, cache_dir=cache_dir,
                                        local_files_only=local_files_only,
                                        patterns=["*.json", "*.safetensors"])
        network = network.eval().to(device=device, dtype=dtype)
        return cls(Generator(network, 16, text_tokenizer=tokenizer,
                             speech_tokenizer_path=codec_path, local_files_only=True))

    def infer(self, text, *, prompt_wav=None, prompt_text=None, speaker_id=1, lang="Chinese", accent=None,
              caption=None, temperature=0.9, topk=10, max_audio_length_ms=40000):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be nonempty")
        if not math.isfinite(temperature) or temperature <= 0 or not 1 <= topk <= 2051:
            raise ValueError("Invalid temperature or topk")
        if not math.isfinite(max_audio_length_ms) or max_audio_length_ms < 800:
            raise ValueError("max_audio_length_ms must be at least 800")
        cloning = prompt_wav is not None
        if cloning:
            if not isinstance(prompt_text, str) or not prompt_text.strip():
                raise ValueError("Cloning requires both prompt_wav and its accurate prompt_text")
            prompt = prepare_prompt(prompt_wav, self.sample_rate, device=self.generator.device).to(self.generator.device)
            context = [Segment(speaker=0, text=prompt_text.strip() + text.strip(), audio=prompt,
                               accent=accent, lang=lang, caption=caption)]
        else:
            if prompt_text is not None:
                raise ValueError("prompt_text requires prompt_wav")
            if isinstance(speaker_id, bool) or not isinstance(speaker_id, int) or speaker_id < 1:
                raise ValueError("speaker_id must be a positive integer present in the trained checkpoint")
            context = [Segment(speaker=speaker_id, text="", audio=None, accent=accent, lang=lang, caption=caption)]
        audio = self.generator.generate(
            text="" if cloning else text.strip(), speaker=0 if cloning else speaker_id, context=context,
            max_audio_length_ms=max_audio_length_ms, temperature=temperature, topk=topk)
        if audio.ndim != 1 or not audio.numel() or not torch.isfinite(audio).all():
            raise RuntimeError("The model returned invalid or empty audio")
        return audio.detach().float().cpu()

    def save(self, path, audio):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, audio.numpy(), self.sample_rate, subtype="PCM_16")
        return str(path)


def inference_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="M5 Hugging Face repo ID or local checkpoint bundle directory (default: %(default)s)")
    parser.add_argument("--codec", default=DEFAULT_CODEC, help="Audio tokenizer repo ID or local directory")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--codec-revision", default="main")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face download cache")
    parser.add_argument("--local-files-only", action="store_true", help="Use local files/cache without network access")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention", choices=("flash_attention_2", "sdpa"), default="flash_attention_2",
                        help="Reference backend: flash_attention_2 (requires flash-attn==2.7.3); sdpa is an explicit non-equivalent alternative")
    parser.add_argument("--text", required=True)
    parser.add_argument("--lang", default="Chinese")
    parser.add_argument("--accent", default=None)
    parser.add_argument("--caption", default=None)
    parser.add_argument("--output", default="output.wav")
    return parser


def load_from_args(args):
    return M5TTS.from_pretrained(args.model, codec=args.codec, device=args.device, revision=args.revision,
                                codec_revision=args.codec_revision, cache_dir=args.cache_dir,
                                local_files_only=args.local_files_only, attention=args.attention)
