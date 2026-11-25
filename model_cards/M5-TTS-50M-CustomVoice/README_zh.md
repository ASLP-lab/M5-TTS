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

[English](README.md) | 简体中文

[GitHub](https://github.com/ASLP-lab/M5-TTS) · [音频展示](https://aslp-lab.github.io/M5-TTS-Demo-Page/) · [在线体验](https://huggingface.co/spaces/ASLP-lab/M5-TTS-Space) · [模型合集](https://huggingface.co/collections/ASLP-lab/m5-tts)

**M5-TTS** 是由 ASLP@NPU 和 LycheeAI Lab 开发的轻量多语种、多方言语音合成系统。模型结合 30M 时序骨干与 20M 局部多 token 预测解码器，使用 16 个音频码本、12.5 帧/秒的 codec 表示生成 **24 kHz** 语音。语音 codec 为独立组件。

**CustomVoice** 是精品说话人播报的专用权重，只需提供文本和训练过的说话人 ID，**无需参考音频或参考转写**。下面使用说话人 ID `1`；可用 ID 和音色能力取决于权重。

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
python infer_speaker.py \
  --model ASLP-lab/M5-TTS-50M-CustomVoice \
  --text "你好，今天的天气很好，适合出去散散步。" \
  --speaker-id 1 \
  --output outputs/speaker.wav
```

M5 权重、文本 tokenizer 和[音频 tokenizer](https://huggingface.co/ASLP-lab/M5-TTS-Audio-Tokenizer) 会自动下载。使用 `--lang Chinese`、`English`、`German`、`French`、`Japanese` 或 `Korean` 指定目标语言。

本权重与 [VoiceClone](https://huggingface.co/ASLP-lab/M5-TTS-50M-VoiceClone) 不同，设置说话人 ID 不会切换已加载的模型权重。Python API 用法及评测结果见[项目 README](https://github.com/ASLP-lab/M5-TTS#quickstart)。

## 许可证

Apache-2.0。基于 Qwen 的音频 codec 保留上游归属，见[第三方声明](https://github.com/ASLP-lab/M5-TTS/blob/main/m5/codec_qwen12hz/NOTICE)。
