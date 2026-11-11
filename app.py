"""Gradio voice-cloning demo matching ASLP-lab/M5-TTS-Space."""
import argparse
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from inference import DEFAULT_CODEC, DEFAULT_MODEL, M5TTS

LOGGER = logging.getLogger(__name__)
REFERENCE_AUDIO = Path(__file__).resolve().parent / "assets" / "aslp_default_voice.wav"
REFERENCE_TEXT = "猪携带该疾病，然后通过蚊子传染给人类。"
TARGET_TEXT = "欢迎来到ASLP实验室。我们专注于语音、语言与智能交互技术研究，期待用声音连接更广阔的世界。"
LANGUAGES = [("Chinese", "zh"), ("English", "en"), ("German", "de"),
             ("French", "fr"), ("Japanese", "ja"), ("Korean", "ko")]
LANGUAGE_LABELS = {code: label for label, code in LANGUAGES}
ACCENTS = ["None", "yue", "chuan"]


@dataclass(frozen=True)
class DemoSettings:
    model: str = DEFAULT_MODEL
    codec: str = DEFAULT_CODEC
    device: str = "cuda"
    attention: str = "flash_attention_2"
    revision: str = "main"
    codec_revision: str = "main"
    cache_dir: str | None = None
    local_files_only: bool = False


def validate_inputs(reference_audio, reference_text, target_text, language, accent,
                    temperature, topk, seed):
    if not reference_audio or not Path(reference_audio).is_file():
        raise gr.Error("Upload or record a reference audio clip first.")
    if not isinstance(reference_text, str) or not reference_text.strip():
        raise gr.Error("Enter the exact transcript of the reference audio.")
    if not isinstance(target_text, str) or not target_text.strip():
        raise gr.Error("Enter the text to synthesize.")
    if len(reference_text.strip()) > 500:
        raise gr.Error("Reference transcript must be at most 500 characters.")
    if len(target_text.strip()) > 250:
        raise gr.Error("Target text must be at most 250 characters.")
    if language not in LANGUAGE_LABELS or accent not in ACCENTS:
        raise gr.Error("Choose one of the available language and accent options.")
    if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or not 0.1 <= temperature <= 1.5:
        raise gr.Error("Temperature must be between 0.1 and 1.5.")
    for label, value, lower, upper in (("Top-k", topk, 1, 50), ("Seed", seed, 0, 2**32 - 1)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or int(value) != value or not lower <= value <= upper):
            raise gr.Error(f"{label} must be an integer between {lower} and {upper}.")
    if Path(reference_audio).stat().st_size > 25 * 1024 * 1024:
        raise gr.Error("Reference audio must be smaller than 25 MB.")
    try:
        info = sf.info(reference_audio)
    except (OSError, RuntimeError, ValueError) as exc:
        raise gr.Error("Cannot read this recording. Please upload a WAV or FLAC file.") from exc
    if not 0.5 <= info.duration <= 30:
        raise gr.Error("Reference audio must be between 0.5 and 30 seconds long.")
    if info.channels not in (1, 2) or not 8000 <= info.samplerate <= 192000:
        raise gr.Error("Use mono or stereo audio with a sample rate between 8 and 192 kHz.")


class DemoBackend:
    def __init__(self, settings):
        self.settings = settings
        self._model = None
        self._lock = threading.Lock()

    def _load_model(self):
        if self._model is None:
            self._model = M5TTS.from_pretrained(
                self.settings.model, codec=self.settings.codec,
                device=self.settings.device, attention=self.settings.attention,
                revision=self.settings.revision, codec_revision=self.settings.codec_revision,
                cache_dir=self.settings.cache_dir, local_files_only=self.settings.local_files_only,
            )
        return self._model

    def synthesize(self, reference_audio, reference_text, target_text, language, accent,
                   temperature, topk, seed):
        validate_inputs(reference_audio, reference_text, target_text, language, accent,
                        temperature, topk, seed)
        with self._lock:
            try:
                model = self._load_model()
            except Exception as exc:
                LOGGER.exception("Model loading failed")
                raise gr.Error("Model loading failed. Check model/codec paths, dependencies and GPU settings in the server logs.") from exc
            device = torch.device(self.settings.device)
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
            started = time.perf_counter()
            try:
                with torch.random.fork_rng(devices=devices), torch.inference_mode():
                    torch.random.default_generator.manual_seed(int(seed))
                    if devices:
                        with torch.cuda.device(devices[0]):
                            torch.cuda.manual_seed(int(seed))
                    audio = model.infer(
                        target_text.strip(), prompt_wav=reference_audio,
                        prompt_text=reference_text.strip(), lang=LANGUAGE_LABELS[language],
                        accent=None if accent == "None" else accent,
                        temperature=float(temperature), topk=int(topk), max_audio_length_ms=40000,
                    )
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
            except Exception as exc:
                LOGGER.exception("Speech generation failed")
                raise gr.Error("Speech generation failed. See the server logs for details.") from exc
            elapsed = time.perf_counter() - started
            waveform = (np.clip(audio.numpy(), -1, 1) * 32767).astype(np.int16)
            metadata = {
                "seed": int(seed), "language": language, "accent": accent,
                "temperature": float(temperature), "top_k": int(topk),
                "sample_rate": model.sample_rate,
                "audio_duration_seconds": round(len(waveform) / model.sample_rate, 3),
                "generation_seconds": round(elapsed, 3),
                "attention": self.settings.attention,
            }
            return (model.sample_rate, waveform), metadata


def create_demo(settings=None):
    backend = DemoBackend(settings or DemoSettings())
    with gr.Blocks(title="M5-TTS", analytics_enabled=False, delete_cache=(3600, 86400)) as demo:
        gr.Markdown(
            "# M5-TTS\nUpload a reference recording and its exact transcript, "
            "then synthesize new speech with the same voice."
        )
        with gr.Row():
            with gr.Column():
                reference_audio = gr.Audio(
                    value=str(REFERENCE_AUDIO), type="filepath", label="Reference audio",
                    sources=["upload", "microphone"],
                )
                reference_text = gr.Textbox(
                    value=REFERENCE_TEXT, lines=3, label="Reference transcript",
                    info="Update this transcript when uploading another voice.",
                )
                target_text = gr.Textbox(
                    value=TARGET_TEXT, lines=5, label="Target text", max_length=250,
                    placeholder="Text to synthesize (maximum 250 characters)",
                )
            with gr.Column():
                language = gr.Dropdown(choices=LANGUAGES, value="zh", label="Language")
                accent = gr.Dropdown(choices=ACCENTS, value="None", label="Accent")
                temperature = gr.Slider(0.1, 1.5, value=0.9, step=0.05, label="Temperature")
                topk = gr.Slider(1, 50, value=10, step=1, label="Top-k")
                seed = gr.Number(value=42, precision=0, label="Seed")
        generate = gr.Button("Generate", variant="primary")
        output_audio = gr.Audio(label="Generated audio")
        metadata = gr.JSON(label="Generation metadata")
        generate.click(
            backend.synthesize,
            inputs=[reference_audio, reference_text, target_text, language, accent, temperature, topk, seed],
            outputs=[output_audio, metadata], api_name="synthesize", concurrency_limit=1,
        )
    return demo.queue(default_concurrency_limit=1, max_size=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("M5_MODEL", DEFAULT_MODEL))
    parser.add_argument("--codec", default=os.getenv("M5_CODEC", DEFAULT_CODEC))
    parser.add_argument("--device", default=os.getenv("M5_DEVICE", "cuda"))
    parser.add_argument("--attention", choices=("flash_attention_2", "sdpa"),
                        default=os.getenv("M5_ATTENTION", "flash_attention_2"))
    parser.add_argument("--revision", default=os.getenv("M5_REVISION", "main"))
    parser.add_argument("--codec-revision", default=os.getenv("M5_CODEC_REVISION", "main"))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--host", default="0.0.0.0" if os.getenv("SPACE_ID") else "127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "7860")))
    args = parser.parse_args()
    settings = DemoSettings(**{name: getattr(args, name) for name in DemoSettings.__dataclass_fields__})
    logging.basicConfig(level=logging.INFO)
    create_demo(settings).launch(
        server_name=args.host, server_port=args.port, share=False,
        show_error=False, max_file_size="25mb",
    )


if __name__ == "__main__":
    main()
