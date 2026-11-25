---
license: apache-2.0
pipeline_tag: text-to-speech
language:
- zh
- en
- de
- fr
- ja
- ko
tags:
- m5-tts
- voice-cloning
- multilingual
- onnx
- onnxruntime
---

# M5-TTS-50M-ONNX

English | [简体中文](README_zh.md)

[GitHub](https://github.com/ASLP-lab/M5-TTS) · [Demo](https://aslp-lab.github.io/M5-TTS-Demo-Page/) · [Space](https://huggingface.co/spaces/ASLP-lab/M5-TTS-Space) · [Models](https://huggingface.co/collections/ASLP-lab/m5-tts)

**M5-TTS** is a compact multilingual and multi-dialect text-to-speech system from ASLP@NPU and LycheeAI Lab. It combines a 30M temporal backbone with a 20M local multi-token prediction decoder, using 16 audio codebooks at 12.5 frames/s for **24 kHz** speech generation. The speech codec is separate from the language model.

This repository provides a **complete FP32 ONNX inference bundle**, including TTS graphs, text/audio embeddings, the text tokenizer and reference-audio encoder/decoder graphs. The same files support CPU and NVIDIA GPU voice cloning; no separate codec download is needed. The source checkpoint is identified in `manifest.json` and should not be assumed identical to another M5 checkpoint.

## Quickstart

Use Python 3.10 on Linux. Install from the code repository:

```bash
git clone https://github.com/ASLP-lab/M5-TTS.git
cd M5-TTS
conda create -n m5tts python=3.10 -y
conda activate m5tts
pip install -r requirements.txt
```

**CPU inference**, from the repository root:

```bash
python infer_onnx.py \
  --model ASLP-lab/M5-TTS-50M-ONNX \
  --device cpu --threads 4 \
  --text "你好，欢迎使用我们的语音合成模型。" \
  --prompt-wav assets/aslp_default_voice.wav \
  --prompt-text "猪携带该疾病，然后通过蚊子传染给人类。" \
  --output outputs/onnx_cpu.wav
```

**NVIDIA GPU inference:**

```bash
python infer_onnx.py \
  --model ASLP-lab/M5-TTS-50M-ONNX \
  --device cuda --device-id 0 \
  --text "你好，欢迎使用我们的语音合成模型。" \
  --prompt-wav assets/aslp_default_voice.wav \
  --prompt-text "猪携带该疾病，然后通过蚊子传染给人类。" \
  --output outputs/onnx_gpu.wav
```

CUDA execution requires a compatible NVIDIA driver. The shared requirements install `onnxruntime-gpu`, which also supports CPU execution; do not additionally install `onnxruntime`. FlashAttention is not used by this entry point.

Add `--stream` to write generated audio chunks incrementally. Output is a WAV plus a same-stem JSON statistics file. The current ONNX entry point requires reference audio and its transcript; it does not expose preset speaker IDs or dialect/accent conditioning. CPU and GPU results may differ numerically.

See the [ONNX guide](https://github.com/ASLP-lab/M5-TTS#onnx-inference) and [Android example](https://github.com/ASLP-lab/M5-TTS/tree/main/android). Only clone voices you own or have permission to use.

## License

Apache-2.0. The Qwen-derived audio codec retains its upstream attribution; see the [third-party notice](https://github.com/ASLP-lab/M5-TTS/blob/main/m5/codec_qwen12hz/NOTICE).
