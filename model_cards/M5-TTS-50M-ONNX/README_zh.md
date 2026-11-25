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

[English](README.md) | 简体中文

[GitHub](https://github.com/ASLP-lab/M5-TTS) · [音频展示](https://aslp-lab.github.io/M5-TTS-Demo-Page/) · [在线体验](https://huggingface.co/spaces/ASLP-lab/M5-TTS-Space) · [模型合集](https://huggingface.co/collections/ASLP-lab/m5-tts)

**M5-TTS** 是由 ASLP@NPU 和 LycheeAI Lab 开发的轻量多语种、多方言语音合成系统。模型结合 30M 时序骨干与 20M 局部多 token 预测解码器，使用 16 个音频码本、12.5 帧/秒的 codec 表示生成 **24 kHz** 语音。语音 codec 与语言模型分开。

本仓库提供**完整的 FP32 ONNX 推理模型包**，包含 TTS 图、文本/音频 embedding、文本 tokenizer 和参考音频编解码图。同一份文件支持 CPU 和 NVIDIA GPU 音色克隆，无需另外下载 codec。源 checkpoint 的身份记录在 `manifest.json` 中，不应假定它与其他 M5 checkpoint 完全相同。

## 快速开始

在 Linux 上使用 Python 3.10，从代码仓库安装：

```bash
git clone https://github.com/ASLP-lab/M5-TTS.git
cd M5-TTS
conda create -n m5tts python=3.10 -y
conda activate m5tts
pip install -r requirements.txt
```

在代码仓库根目录执行 **CPU 推理**：

```bash
python infer_onnx.py \
  --model ASLP-lab/M5-TTS-50M-ONNX \
  --device cpu --threads 4 \
  --text "你好，欢迎使用我们的语音合成模型。" \
  --prompt-wav assets/aslp_default_voice.wav \
  --prompt-text "猪携带该疾病，然后通过蚊子传染给人类。" \
  --output outputs/onnx_cpu.wav
```

**NVIDIA GPU 推理：**

```bash
python infer_onnx.py \
  --model ASLP-lab/M5-TTS-50M-ONNX \
  --device cuda --device-id 0 \
  --text "你好，欢迎使用我们的语音合成模型。" \
  --prompt-wav assets/aslp_default_voice.wav \
  --prompt-text "猪携带该疾病，然后通过蚊子传染给人类。" \
  --output outputs/onnx_gpu.wav
```

CUDA 推理需要兼容的 NVIDIA 驱动。统一依赖安装的 `onnxruntime-gpu` 同时支持 CPU，不要再安装 `onnxruntime`。此入口不使用 FlashAttention。

添加 `--stream` 可将生成音频分块写入文件。输出包含 WAV 和同名 JSON 统计文件。当前 ONNX 入口需要参考音频及转写，不提供精品说话人 ID 或方言/口音条件控制。CPU 与 GPU 结果可能存在数值差异。

详细说明见 [ONNX 指南](https://github.com/ASLP-lab/M5-TTS#onnx-inference)和 [Android 示例](https://github.com/ASLP-lab/M5-TTS/tree/main/android)。仅使用本人或已获授权的声音进行克隆。

## 许可证

Apache-2.0。基于 Qwen 的音频 codec 保留上游归属，见[第三方声明](https://github.com/ASLP-lab/M5-TTS/blob/main/m5/codec_qwen12hz/NOTICE)。
