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
- custom-voice
- multilingual
- pytorch
---

# M5-TTS-50M-CustomVoice

English | [简体中文](README_zh.md)

[GitHub](https://github.com/ASLP-lab/M5-TTS) · [Demo](https://aslp-lab.github.io/M5-TTS-Demo-Page/) · [Space](https://huggingface.co/spaces/ASLP-lab/M5-TTS-Space) · [Models](https://huggingface.co/collections/ASLP-lab/m5-tts)

**M5-TTS** is a compact multilingual and multi-dialect text-to-speech system from ASLP@NPU and LycheeAI Lab. It combines a 30M temporal backbone with a 20M local multi-token prediction decoder, using 16 audio codebooks at 12.5 frames/s for **24 kHz** speech generation. The speech codec is separate.

**CustomVoice** is the dedicated checkpoint for preset-speaker synthesis: provide text and a trained speaker ID, with **no reference audio or reference transcript required**. The example below uses speaker ID `1`. Available IDs and voice capabilities depend on the checkpoint.

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
python infer_speaker.py \
  --model ASLP-lab/M5-TTS-50M-CustomVoice \
  --text "你好，今天的天气很好，适合出去散散步。" \
  --speaker-id 1 \
  --output outputs/speaker.wav
```

The M5 checkpoint, text tokenizer and [audio tokenizer](https://huggingface.co/ASLP-lab/M5-TTS-Audio-Tokenizer) are downloaded automatically. Use `--lang Chinese`, `English`, `German`, `French`, `Japanese` or `Korean` to set the target language.

This checkpoint is different from [VoiceClone](https://huggingface.co/ASLP-lab/M5-TTS-50M-VoiceClone). Setting a speaker ID does not switch a loaded model's weights. See the [project README](https://github.com/ASLP-lab/M5-TTS#quickstart) for Python API usage and evaluation results.

## License

Apache-2.0. The Qwen-derived audio codec retains its upstream attribution; see the [third-party notice](https://github.com/ASLP-lab/M5-TTS/blob/main/m5/codec_qwen12hz/NOTICE).
