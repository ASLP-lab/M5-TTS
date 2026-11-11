"""Standalone FP32 ONNX voice cloning on CPU or NVIDIA CUDA; no PyTorch.

Install: pip install -r requirements.txt
Run: python infer_onnx.py --device cpu --text "你好" \
    --prompt-wav reference.wav --prompt-text "参考转写" --output output.wav
Use --device cuda --device-id 0 for NVIDIA GPU inference with the same ONNX weights.
The default Hugging Face ID is the intended upload target, not proof of publication.
"""
import argparse
import hashlib
import json
from math import gcd
from pathlib import Path
import time
import warnings

import numpy as np
import onnxruntime as ort
import soundfile as sf
from huggingface_hub import snapshot_download
from scipy.signal import resample_poly
from tokenizers import Tokenizer

DEFAULT_MODEL = "ASLP-lab/M5-TTS-50M-VoiceClone-ONNX"
FILES = ("manifest.json", "tokenizer.json", "text_embeddings.npy", "audio_embeddings.npy",
         "backbone.onnx", "depth_frame.onnx", "codec_encoder.onnx", "codec_decoder.onnx",
         "codec_decoder_cached.onnx", "codec_cached.json", "codec_identity.json", "checksums.json")
CONTROL_TOKENS = ("<|caption_begin|>", "<|caption_end|>", "<|lang_begin|>", "<|lang_end|>",
                  "<|accent_begin|>", "<|accent_end|>", "<|speaker_start|>", "<|speaker_end|>")


def _condition(value):
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError("Language must be a string or None")
    value = value.strip()
    if any(token in value for token in CONTROL_TOKENS):
        raise ValueError("Reserved conditioning token in language")
    return "" if value.lower() in ("none", "null") else value


def conditioned_text(text, lang):
    if any(token in text for token in CONTROL_TOKENS):
        raise ValueError("Reserved conditioning token in text")
    return ("<|caption_begin|><|caption_end|><|accent_begin|><|accent_end|>"
            "<|speaker_start|>None<|speaker_end|>"
            f"<|lang_begin|>{_condition(lang)}<|lang_end|>{text}")


def read_reference(path, sample_rate=24000):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError(f"Empty or invalid reference audio: {path}")
    if sr != sample_rate:
        divisor = gcd(sr, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, sr // divisor).astype(np.float32)
    audio /= np.max(np.abs(audio)) + 1e-8
    threshold = 0.02 if len(audio) > 10 * sample_rate else 0.015
    speech = np.flatnonzero(np.abs(audio) > threshold)
    if len(speech):
        cuts = np.flatnonzero(np.diff(speech) > int(0.15 * sample_rate))
        starts = np.r_[speech[0], speech[cuts + 1]]
        final_end = speech[-1] + 1 if len(audio) - speech[-1] - 1 >= int(0.15 * sample_rate) else len(audio)
        ends = np.r_[speech[cuts] + 1, final_end]
        audio = np.concatenate([audio[max(0, a - 1200):min(len(audio), b + 1200)] for a, b in zip(starts, ends)])
    return np.ascontiguousarray(audio, dtype=np.float32)


def resolve_model(model, revision="main", cache_dir=None, local_files_only=False):
    path = Path(model).expanduser()
    if path.is_dir():
        return path.resolve()
    if path.exists() or path.is_absolute() or str(model).startswith((".", "~")):
        raise FileNotFoundError(f"Expected a local ONNX model directory: {model}")
    return Path(snapshot_download(str(model), revision=revision, cache_dir=cache_dir,
                                  local_files_only=local_files_only, allow_patterns=list(FILES)))


def verify_files(root):
    checks = json.loads((root / "checksums.json").read_text())
    for name in FILES[:-1]:
        spec = checks[name]
        path = root / name
        if path.stat().st_size != spec["bytes"]:
            raise ValueError(f"Model file size mismatch: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != spec["sha256"]:
            raise ValueError(f"Model file SHA256 mismatch: {name}")


def execution_providers(device, device_id):
    if device not in ("cpu", "cuda") or isinstance(device_id, bool) or not isinstance(device_id, int) or device_id < 0:
        raise ValueError("device must be cpu or cuda and device_id must be a non-negative integer")
    if device == "cpu":
        if device_id != 0:
            raise ValueError("device_id is only used with device=cuda")
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDA was requested but this ONNX Runtime has no CUDA provider. Install requirements.txt; remove legacy onnxruntime first and keep only onnxruntime-gpu.")
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls(cuda=True, cudnn=True, msvc=False)
    return [("CUDAExecutionProvider", {"device_id": device_id, "use_tf32": 0}), "CPUExecutionProvider"]


def create_session(path, options, providers, requested_provider):
    try:
        session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    except Exception as error:
        raise RuntimeError(f"Could not initialize {requested_provider} for {path.name}. Check the device index and CUDA/cuDNN libraries if using CUDA.") from error
    active = session.get_providers()
    if not active or active[0] != requested_provider:
        raise RuntimeError(f"{path.name}: requested {requested_provider}, got {active}; refusing silent backend fallback. Check the GPU index and CUDA/cuDNN runtime libraries.")
    session.disable_fallback()
    return session


class M5Onnx:
    """Single-request CPU/CUDA runtime; consume generate() fully before another request."""

    def __init__(self, model=DEFAULT_MODEL, *, threads=4, codec_mode="cached", revision="main",
                 cache_dir=None, local_files_only=False, verify=False, device="cpu", device_id=0):
        if threads < 1 or codec_mode not in ("cached", "window"):
            raise ValueError("threads must be positive and codec_mode must be cached or window")
        providers = execution_providers(device, device_id)
        self.device, self.device_id = device, device_id
        self.provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
        self.root = resolve_model(model, revision, cache_dir, local_files_only)
        for name in FILES[:-1]:
            path = self.root / name
            if not path.is_file() or (path.suffix in (".onnx", ".npy") and path.stat().st_size < 1024):
                raise FileNotFoundError(f"Missing or incomplete ONNX asset: {path}")
        if verify:
            verify_files(self.root)
        self.meta = json.loads((self.root / "manifest.json").read_text())
        expected = {"format_version": 1, "codebooks": 16, "audio_vocab": 2051, "sample_rate": 24000,
                    "hop_length": 1920, "topk": 10, "temperature": 0.9, "min_frames": 6}
        if any(self.meta.get(k) != v for k, v in expected.items()):
            raise ValueError("Unsupported ONNX bundle format or baked-in sampling configuration")
        self.sample_rate = self.meta["sample_rate"]
        self.threads, self.codec_mode = threads, codec_mode
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.sessions = {}
        for name in ("backbone", "depth_frame", "codec_encoder", "codec_decoder"):
            suffix = "_cached" if name == "codec_decoder" and codec_mode == "cached" else ""
            self.sessions[name] = create_session(self.root / f"{name}{suffix}.onnx", options, providers, self.provider)
        self.session_providers = {name: session.get_providers() for name, session in self.sessions.items()}
        self.provider_options = {name: session.get_provider_options() for name, session in self.sessions.items()}
        self.codec_meta = json.loads((self.root / "codec_cached.json").read_text())
        self.text_embeddings = np.load(self.root / "text_embeddings.npy", mmap_mode="r", allow_pickle=False)
        self.audio_embeddings = np.load(self.root / "audio_embeddings.npy", mmap_mode="r", allow_pickle=False)
        if (self.text_embeddings.dtype != np.float32 or self.audio_embeddings.dtype != np.float32
                or self.text_embeddings.shape[1] != self.meta["hidden_size"]
                or self.audio_embeddings.shape != (16 * 2051, self.meta["hidden_size"])):
            raise ValueError("Invalid embedding shape or precision")
        self.tokenizer = Tokenizer.from_file(str(self.root / "tokenizer.json"))
        if self.tokenizer.get_vocab_size() != self.text_embeddings.shape[0]:
            raise ValueError("Tokenizer and text embedding vocabulary sizes differ")
        self.offsets = np.arange(16) * 2051
        self.stats = {}

    def encode_reference(self, audio):
        codes = self.sessions["codec_encoder"].run(None, {"audio": audio[None]})[0]
        codes = codes[0, :, :int(np.ceil(len(audio) / self.meta["hop_length"]))].T.copy()
        if codes.ndim != 2 or codes.shape[1] != 16 or not len(codes) or np.any(codes < 0) or np.any(codes > 2047):
            raise ValueError("Invalid reference codec output")
        return codes

    def _reset_codec(self):
        meta = self.codec_meta
        self.codec_offset = 0
        self.codec_past = np.zeros((meta["layers"], 2, 1, meta["kv_heads"], 0, meta["head_dim"]), np.float32)
        self.codec_states = [np.zeros(spec["shape"], np.float32) for spec in meta["states"]]

    def _decode(self, codes):
        codes = np.ascontiguousarray(codes.T[None], dtype=np.int64)
        if self.codec_mode == "window":
            return self.sessions["codec_decoder"].run(None, {"codes": codes})[0].reshape(-1)
        feed = dict(zip(self.codec_meta["inputs"], [codes, np.array([self.codec_offset], np.int64),
                                                  self.codec_past, *self.codec_states]))
        pcm, self.codec_past, *self.codec_states = self.sessions["codec_decoder"].run(None, feed)
        self.codec_offset += codes.shape[-1]
        return pcm.reshape(-1)

    def generate(self, text, prompt_wav, prompt_text, *, lang="Chinese", seed=1234, max_frames=500,
                 stream=False, first_frames=4, chunk_frames=4, left_context=8):
        """Yield 24kHz float32 PCM. This is generation streaming, not audio playback."""
        if not isinstance(text, str) or not text.strip() or not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ValueError("Target text and accurate reference transcript are required")
        if max_frames < 1 or first_frames < 1 or chunk_frames < 1 or left_context < 0 or seed < 0:
            raise ValueError("Invalid seed or generation/chunk limits")
        sentence = conditioned_text(prompt_text.strip() + text.strip(), lang)
        audio = read_reference(prompt_wav, self.sample_rate)
        self.stats = {"status": "running", "provider": self.provider, "precision": "fp32",
                      "device": self.device, "device_id": self.device_id if self.device == "cuda" else None,
                      "session_providers": self.session_providers, "provider_options": self.provider_options,
                      "cuda_use_tf32": False if self.device == "cuda" else None}
        started = time.perf_counter()
        prompt = self.encode_reference(audio)
        ids = np.array(self.tokenizer.encode(sentence).ids)
        prefix = self.audio_embeddings[prompt + 1 + self.offsets].sum(axis=1)
        hidden = np.concatenate((self.text_embeddings[ids], prefix), axis=0)[None]
        if hidden.shape[1] + max_frames > self.meta["max_sequence"]:
            raise ValueError("Reference plus text and generation budget exceed the model context length")
        past = np.empty((self.meta["layers"], 2, 1, self.meta["kv_heads"], 0, self.meta["head_dim"]), np.float32)
        positions = np.arange(hidden.shape[1], dtype=np.int64)[None]
        if self.codec_mode == "cached":
            self._reset_codec()
        codec_start = max(0, len(prompt) - left_context) if stream else 0
        rng = np.random.default_rng(seed)
        generated = []
        generated_buffer = np.empty((max_frames, 16), np.int64)
        decoded, emitted = 0, 0
        pending = np.empty(0, np.float32)
        first_pcm_s, eos = None, False

        def decode_new():
            nonlocal decoded
            new_start = len(prompt) + decoded
            window_start = max(0, new_start - left_context) if stream else 0
            if self.codec_mode == "cached":
                cache_start = codec_start + self.codec_offset
                total_end = len(prompt) + len(generated)
                new_prompt = prompt[cache_start:min(total_end, len(prompt))]
                new_generated = generated_buffer[max(cache_start, len(prompt)) - len(prompt):len(generated)]
                new_codes = np.concatenate((new_prompt, new_generated), axis=0)
                pcm = np.concatenate([self._decode(new_codes[i:i + 32]) for i in range(0, len(new_codes), 32)])
                window_start = cache_start
            else:
                all_codes = np.concatenate((prompt, np.stack(generated)), axis=0)
                if stream:
                    pcm = self._decode(all_codes[window_start:])
                else:
                    pcm = np.concatenate([self._decode(all_codes[max(0, i - 25):i + 300])[
                        (i - max(0, i - 25)) * self.meta["hop_length"]:] for i in range(0, len(all_codes), 300)])
            decoded = len(generated)
            return pcm[(new_start - window_start) * self.meta["hop_length"]:]

        for index in range(max_frames):
            last, past = self.sessions["backbone"].run(None, dict(hidden=hidden, positions=positions, past=past))
            noise = rng.exponential(size=(16, 1, self.meta["audio_vocab"])).astype(np.float32)
            noise = np.maximum(noise, np.finfo(np.float32).tiny)
            codes, hidden = self.sessions["depth_frame"].run(None, dict(hidden=last, noise=noise,
                                                                       allow_eos=np.array([index > self.meta["min_frames"]])))
            if np.all(codes == 0):
                eos = True
                break
            generated.append(np.clip(codes[0] - 1, 0, 2047))
            generated_buffer[index] = generated[-1]
            positions = positions[:, -1:] + 1
            if stream and len(generated) - decoded >= (first_frames if decoded == 0 else chunk_frames):
                pending = np.concatenate((pending, decode_new()))
                if len(pending) > 2400:
                    pcm, pending = pending[:-2400], pending[-2400:]
                    if not np.isfinite(pcm).all():
                        raise RuntimeError("Codec returned non-finite PCM")
                    if first_pcm_s is None:
                        first_pcm_s = time.perf_counter() - started
                    emitted += len(pcm)
                    yield pcm
        if len(generated) > decoded:
            pending = np.concatenate((pending, decode_new()))
        if len(pending) > 2400:
            pcm = pending[:-2400]
            if not np.isfinite(pcm).all():
                raise RuntimeError("Codec returned non-finite PCM")
            if first_pcm_s is None:
                first_pcm_s = time.perf_counter() - started
            emitted += len(pcm)
            yield pcm
        self.generated_codes = np.stack(generated) if generated else np.empty((0, 16), np.int64)
        elapsed = time.perf_counter() - started
        self.stats.update(status="complete" if eos else "truncated", frames=len(generated), eos=eos,
                          truncated=not eos, audio_s=emitted / self.sample_rate, elapsed_s=elapsed,
                          rtf=elapsed / (emitted / self.sample_rate) if emitted else None,
                          first_pcm_s=first_pcm_s, threads=self.threads, codec_mode=self.codec_mode,
                          streaming=stream, seed=seed, checkpoint_sha256=self.meta.get("checkpoint_sha256"))
        if not emitted:
            raise RuntimeError("No audio generated; check the reference and generation budget")
        if not eos:
            warnings.warn("Generation reached max_frames without EOS; the audio may be incomplete", RuntimeWarning)

    def infer(self, text, prompt_wav, prompt_text, **kwargs):
        return np.concatenate(list(self.generate(text, prompt_wav, prompt_text, **kwargs)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF ONNX repository ID or local ONNX directory")
    parser.add_argument("--text", required=True)
    parser.add_argument("--prompt-wav", required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--output", default="outputs/onnx.wav")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="ONNX execution device (default: cpu)")
    parser.add_argument("--device-id", type=int, default=0, help="CUDA GPU index among devices visible to this process")
    parser.add_argument("--lang", default="Chinese")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--codec-mode", choices=("cached", "window"), default="cached")
    parser.add_argument("--stream", action="store_true", help="Write generated chunks incrementally; no live playback")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Verify the packaged SHA256 checksums before loading")
    args = parser.parse_args()
    output = Path(args.output).expanduser()
    stats_path = output.with_suffix(".json")
    if output.suffix.lower() != ".wav" or output.exists() or stats_path.exists():
        parser.error("Choose a new .wav output path; existing output/statistics are not overwritten")
    model = M5Onnx(args.model, threads=args.threads, codec_mode=args.codec_mode, revision=args.revision,
                   cache_dir=args.cache_dir, local_files_only=args.local_files_only, verify=args.verify,
                   device=args.device, device_id=args.device_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        parser.error(f"Temporary output already exists: {temporary}")
    try:
        with sf.SoundFile(temporary, mode="w", samplerate=model.sample_rate, channels=1, subtype="PCM_16", format="WAV") as stream:
            for pcm in model.generate(args.text, args.prompt_wav, args.prompt_text, lang=args.lang, seed=args.seed,
                                      max_frames=args.max_frames, stream=args.stream):
                stream.write(pcm)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    stats_path.write_text(json.dumps(model.stats, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output), **model.stats), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
