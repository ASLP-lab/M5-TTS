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
- pytorch
---

# M5-TTS-50M-VoiceClone

[English](README.md) | 简体中文

[GitHub](https://github.com/ASLP-lab/M5-TTS) · [音频展示](https://aslp-lab.github.io/M5-TTS-Demo-Page/) · [在线体验](https://huggingface.co/spaces/ASLP-lab/M5-TTS-Space) · [模型合集](https://huggingface.co/collections/ASLP-lab/m5-tts)

**M5-TTS** 是由 ASLP@NPU 和 LycheeAI Lab 开发的轻量多语种、多方言语音合成系统。模型结合 30M 时序骨干与 20M 局部多 token 预测解码器，使用 16 个音频码本、12.5 帧/秒的 codec 表示生成 **24 kHz** 语音。语音 codec 为独立组件。

**VoiceClone** 根据目标文本、参考音频及其准确转写生成语音，无需针对参考说话人进行额外训练。支持中文、英语、德语、法语、日语和韩语，以及粤语、四川话条件。

## 快速开始

推荐使用 Linux、Python 3.10、支持 BF16 的 NVIDIA GPU 和兼容的 CUDA 驱动。从代码仓库安装：

```bash
git clone https://github.com/ASLP-lab/M5-TTS.git
cd M5-TTS
conda create -n m5tts python=3.10 -y
conda activate m5tts
pip install -r requirements.txt
pip install --no-build-isolation flash-attn==2.7.3
```

在代码仓库根目录执行：

```bash
python infer_clone.py \
  --model ASLP-lab/M5-TTS-50M-VoiceClone \
  --text "你好，欢迎使用我们的语音合成模型。" \
  --prompt-wav assets/aslp_default_voice.wav \
  --prompt-text "猪携带该疾病，然后通过蚊子传染给人类。" \
  --output outputs/clone.wav
```

更换音色时，请同时替换参考录音和准确转写。M5 权重、文本 tokenizer 和[音频 tokenizer](https://huggingface.co/ASLP-lab/M5-TTS-Audio-Tokenizer) 会自动下载。通过 `--lang` 指定目标语言；粤语使用 `--lang Chinese --accent yue`，四川话使用 `--lang Chinese --accent chuan`。

精品说话人播报请使用独立的 [CustomVoice 权重](https://huggingface.co/ASLP-lab/M5-TTS-50M-CustomVoice)。完整用法和评测结果见[项目 README](https://github.com/ASLP-lab/M5-TTS#quickstart)。仅使用本人或已获授权的声音进行克隆。

## 许可证

Apache-2.0。基于 Qwen 的音频 codec 保留上游归属，见[第三方声明](https://github.com/ASLP-lab/M5-TTS/blob/main/m5/codec_qwen12hz/NOTICE)。
