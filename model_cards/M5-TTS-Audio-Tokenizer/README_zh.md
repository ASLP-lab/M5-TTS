---
license: apache-2.0
pipeline_tag: feature-extraction
tags:
- m5-tts
- audio-tokenizer
- audio-codec
- pytorch
---

# M5-TTS-Audio-Tokenizer

[English](README.md) | 简体中文

[GitHub](https://github.com/ASLP-lab/M5-TTS) · [音频展示](https://aslp-lab.github.io/M5-TTS-Demo-Page/) · [在线体验](https://huggingface.co/spaces/ASLP-lab/M5-TTS-Space) · [模型合集](https://huggingface.co/collections/ASLP-lab/m5-tts)

本仓库提供 **M5-TTS 使用的 Qwen 12Hz 音频 codec**，将参考录音编码为离散音频 token，并将生成的 token 解码为语音。它**不是独立的文本转语音模型**，也不同于 M5 checkpoint 中的文本 tokenizer。

| 属性 | 数值 |
| --- | --- |
| 音频采样率 | 24 kHz |
| Codec 帧率 | 12.5 帧/秒 |
| M5 使用的码本数 | 16 |
| M5 适配层的 codes 布局 | 补齐后的 batch 为 `(batch, frames, 16)` |

[VoiceClone](https://huggingface.co/ASLP-lab/M5-TTS-50M-VoiceClone) 和 [CustomVoice](https://huggingface.co/ASLP-lab/M5-TTS-50M-CustomVoice) 推理脚本会自动下载本 codec，通常无需单独加载。

## 快速开始

单独进行音频编码与重建时，使用 Python 3.10 和 M5 代码仓库：

```bash
git clone https://github.com/ASLP-lab/M5-TTS.git
cd M5-TTS
conda create -n m5tts python=3.10 -y
conda activate m5tts
pip install -r requirements.txt
```

在代码仓库根目录运行，下面的 CPU 示例不需要 FlashAttention：

```python
import soundfile as sf
from m5.codec_qwen12hz import Qwen3TTSTokenizer

codec = Qwen3TTSTokenizer.from_pretrained(
    "ASLP-lab/M5-TTS-Audio-Tokenizer",
    device_map="cpu",
)

encoded = codec.encode("assets/aslp_default_voice.wav")
waveforms, sample_rate = codec.decode(encoded)
sf.write("reconstructed.wav", waveforms[0], sample_rate)
```

`encode()` 也支持 NumPy 音频数组，但需明确输入采样率，例如 `codec.encode(audio, sr=24000)`。适配层会按需重采样；codec 重建是有损的。TTS 推理脚本会另行执行参考音频预处理。

## 许可证与归属

Apache-2.0。本仓库使用 [Qwen3-TTS 12Hz tokenizer](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz)。M5 仓库包含其最小 codec 实现，并保留[上游许可证和修改说明](https://github.com/ASLP-lab/M5-TTS/tree/main/m5/codec_qwen12hz)。无需额外安装 `qwen-tts` 包。
