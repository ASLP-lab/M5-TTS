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
- chuan
- sichuanese
- 四川话
- m5-tts
- voice-cloning
- multilingual
- pytorch
---

# M5-TTS

English | [简体中文](README_zh.md)

[GitHub](https://github.com/ASLP-lab/M5-TTS) · [Demo](https://aslp-lab.github.io/M5-TTS-Demo-Page/) · [Space](https://huggingface.co/spaces/ASLP-lab/M5-TTS-Space) · [Models](https://huggingface.co/collections/ASLP-lab/m5-tts)

**M5-TTS** is a compact multilingual and multi-dialect text-to-speech system from ASLP@NPU and LycheeAI Lab. It combines a 30M temporal backbone with a 20M local multi-token prediction decoder, using 16 audio codebooks at 12.5 frames/s for **24 kHz** speech generation. The speech codec is separate.

**Supported languages and dialects:** Mandarin Chinese · English · German · French · Japanese · Korean · Cantonese · Sichuanese.

This repository provides the reference-based voice-cloning checkpoint. It supports Chinese, English, German, French, Japanese and Korean, with Cantonese and Sichuan dialect conditioning. For preset-speaker synthesis without reference audio, use [M5-TTS-50M-CustomVoice](https://huggingface.co/ASLP-lab/M5-TTS-50M-CustomVoice).

## Quickstart

Use Python 3.10 on Linux with a BF16-capable NVIDIA GPU and a compatible CUDA driver. Install from the code repository:

```bash
git clone https://github.com/ASLP-lab/M5-TTS.git
cd M5-TTS
conda create -n m5tts python=3.10 -y
conda activate m5tts
pip install -r requirements.txt
pip install --no-build-isolation flash-attn==2.7.3
```

Run from the repository root:

```bash
python infer_clone.py \
  --model ASLP-lab/M5-TTS \
  --text "你好，欢迎使用我们的语音合成模型。" \
  --prompt-wav assets/aslp_default_voice.wav \
  --prompt-text "猪携带该疾病，然后通过蚊子传染给人类。" \
  --output outputs/clone.wav
```

Replace the reference recording and its transcript together to clone another voice. The M5 checkpoint, text tokenizer and [audio tokenizer](https://huggingface.co/ASLP-lab/M5-TTS-Audio-Tokenizer) are downloaded automatically. Use `--lang` for the target language; Cantonese uses `--lang Chinese --accent yue`, and Sichuan uses `--lang Chinese --accent chuan`.

For full usage and evaluation results, see the [project README](https://github.com/ASLP-lab/M5-TTS#quickstart). Only clone voices you own or have permission to use.

## License

Apache-2.0. The Qwen-derived audio codec retains its upstream attribution; see the [third-party notice](https://github.com/ASLP-lab/M5-TTS/blob/main/m5/codec_qwen12hz/NOTICE).
