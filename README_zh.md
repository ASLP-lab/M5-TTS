<p align="center">
  <img src="assets/logos/m5-tts-logo.png" alt="M5-TTS 蜂鸟标志" width="440">
</p>

<h1 align="center">M5-TTS</h1>

<p align="center">
  <b>A Mini-Scale Multilingual and Multi-Dialect Text-to-Speech Language Model with Multi-Token Prediction</b>
</p>

<p align="center">
  <b>50M LM · 六种语言 · 中文方言 · 零样本音色克隆 · CPU 流式合成</b>
</p>

<p align="center">
  <a href="README.md">English</a> | 简体中文<br>
  <a href="https://arxiv.org/abs/2611.10162">技术报告</a> ·
  <a href="https://aslp-lab.github.io/M5-TTS-Demo-Page/">音频演示</a> ·
  <a href="https://huggingface.co/spaces/ASLP-lab/M5-TTS-Space">在线体验</a> ·
  <a href="https://huggingface.co/collections/ASLP-lab/m5-tts">模型合集</a> ·
  <a href="https://huggingface.co/ASLP-lab/M5-TTS">模型仓库</a>
</p>

## 概述
https://github.com/user-attachments/assets/9fa7b835-a084-4cf6-9074-c45cb28f9b9e

**M5-TTS** 是基于语言模型的轻量语音生成系统，支持多语种、中文方言、零样本音色克隆，以及精品说话人的单人多语合成。其 **50M LM 参数量不含语音 codec**，由时间维度 backbone 和局部多 Token 预测（MTP）模块组成。

**LM 参数量仅为 MOSS-TTS-Nano 的一半，Seed-ZH / Seed-EN 相对错误率分别降低 62.2% / 35.3%。** M5-TTS（GRPO）达到 **1.37% CER / 2.91% WER**，Nano 对应为 **3.62% / 4.50%**。在报告所测的四核 Xeon 配置下，CPU 流式合成达到 **RTF 0.653** 和 **225 ms 预热后首包延迟**。

**VoiceJudge 自然度评测：** 内部评测模型在两种 A/B 顺序下，对 M5-TTS 相比 MOSS-TTS-Nano 的偏好率分别为 **84.89% 和 86.50%**。

### 核心特性

|  | 特性 | M5-TTS 能力 |
| :--- | :--- | :--- |
| **M₁** | **Mini** | 50M LM：30M 时间 backbone 与 20M 局部解码器；codec 参数单独计算。 |
| **M₂** | **Multilingual** | 中文、英语、德语、法语、日语、韩语。 |
| **M₃** | **Multi-Dialect** | 粤语（Yue）和四川话（Chuan），提供对应方言评测。 |
| **M₄** | **Multi-Token Prediction** | 16 个码本，codec 帧率为 12.5 Hz；分层进行时间建模和帧内生成。 |
| **M₅** | **Language Modeling** | 参考语音驱动的零样本克隆、精品说话人单人多语合成，以及 GRPO 后训练。 |

- **24 kHz 语音输出**，使用因果语音 codec 解码器。
- **CPU 流式推理**：结合计算图优化、缓存复用、有状态 codec 解码和自适应分块调度。
- **全局 + 局部后训练反馈**：相似度和停顿奖励分别组内归一化后加权，再结合由字符编辑错误产生的 CER 引导的局部过程反馈。

### 在线体验

在[在线 Space](https://huggingface.co/spaces/ASLP-lab/M5-TTS-Space)体验参考语音驱动的音色克隆，在[音频演示页](https://aslp-lab.github.io/M5-TTS-Demo-Page/)试听多语种和方言对比样例。

| 资源 | 链接 |
| :--- | :--- |
| 技术报告 | [M5-TTS 技术报告 — v3](https://aslp-lab.github.io/M5-TTS-Demo-Page/m5-tts-v3.pdf) |
| 音频演示 | [多语种与方言样例](https://aslp-lab.github.io/M5-TTS-Demo-Page/) |
| 在线体验 | [Hugging Face Space](https://huggingface.co/spaces/ASLP-lab/M5-TTS-Space) |
| 模型合集 | [ASLP-lab / M5-TTS collection](https://huggingface.co/collections/ASLP-lab/m5-tts) |
| 模型仓库 | [ASLP-lab / M5-TTS](https://huggingface.co/ASLP-lab/M5-TTS) |

**开放范围：** 推理代码、语音克隆模型权重及精品说话人模型权重；**不开放预训练、CPT 和 GRPO 训练代码**。VoiceJudge 的模型权重与推理代码也计划开放，具体文件以本仓库和模型仓库的发布内容为准。

## 目录

- [概述](#概述)
  - [核心特性](#核心特性)
  - [在线体验](#在线体验)
- [支持的语言](#支持的语言)
- [快速开始](#快速开始)
  - [安装](#安装)
  - [音色克隆](#音色克隆)
  - [精品说话人播报](#精品说话人播报)
  - [Python API](#python-api)
  - [网页 Demo](#网页-demo)
  - [ONNX 推理](#onnx-推理)
    - [CPU](#cpu)
    - [NVIDIA GPU](#nvidia-gpu)
  - [导出 ONNX](#导出-onnx)
  - [Android ONNX Runtime 示例](#android-onnx-runtime-示例)
- [评测](#评测)
- [方法概述](#方法概述)
- [当前限制与后续方向](#当前限制与后续方向)
- [贡献者](#贡献者)
  - [数据致谢](#数据致谢)
- [许可证](#许可证)
- [引用](#引用)
- [联系我们](#联系我们)

## 支持的语言

| 语言 | `--lang` / Python `lang` |
| --- | --- |
| 中文 | `Chinese` |
| 英语 | `English` |
| 德语 | `German` |
| 法语 | `French` |
| 日语 | `Japanese` |
| 韩语 | `Korean` |

粤语使用 `--lang Chinese --accent yue`；四川话使用 `--lang Chinese --accent chuan`。输出为 24 kHz 单声道音频。

## 快速开始

模型权重自动从 Hugging Face 下载。使用 `--model <仓库ID或本地目录>` 可选择其他模型。PyTorch 推理和 ONNX 导出默认使用 [ASLP-lab/M5-TTS-Audio-Tokenizer](https://huggingface.co/ASLP-lab/M5-TTS-Audio-Tokenizer) 作为音频 codec，可通过 `--codec` 指定其他仓库或本地目录。

| 推理入口 | 默认模型 |
| --- | --- |
| `infer_clone.py` / 网页 Demo | `ASLP-lab/M5-TTS-50M-VoiceClone` |
| `infer_speaker.py` | `ASLP-lab/M5-TTS-50M-CustomVoice` |
| `infer_onnx.py` | `ASLP-lab/M5-TTS-50M-ONNX` |

音色克隆和精品说话人使用不同 checkpoint。在已加载的模型上设置 `speaker_id=1`，不会自动切换权重。

### 安装

推荐使用 Python 3.10 和 Linux x86_64。在仓库根目录执行：

```bash
conda create -n m5tts python=3.10 -y
conda activate m5tts
pip install -r requirements.txt
```

使用 PyTorch 推理时，在安装 PyTorch 后补充安装 FlashAttention：

```bash
pip install --no-build-isolation flash-attn==2.7.3
```

PyTorch 推理采用 NVIDIA GPU 上的 FlashAttention 2 和 BF16。从源码编译 FlashAttention 需要兼容的 CUDA Toolkit 和 C++ 编译器。ONNX CPU 推理不需要 FlashAttention 或 GPU。

### 音色克隆

提供目标文本、参考音频及其准确转写：

```bash
python infer_clone.py \
  --text "你好，欢迎使用我们的语音合成模型。" \
  --prompt-wav assets/aslp_default_voice.wav \
  --prompt-text "猪携带该疾病，然后通过蚊子传染给人类。" \
  --output outputs/clone.wav
```

使用自己的音色时，同时替换参考音频和转写。仅使用本人或已获授权的录音。

### 精品说话人播报

不提供参考音频，直接指定说话人 ID：

```bash
python infer_speaker.py \
  --text "你好，今天的天气很好，适合出去散散步。" \
  --speaker-id 1 \
  --output outputs/speaker.wav
```

说话人 ID 必须是所选权重实际支持的 ID。

### Python API

```python
from inference import M5TTS

model = M5TTS.from_pretrained(device="cuda")
audio = model.infer(
    "你好，欢迎使用我们的语音合成模型。",
    prompt_wav="assets/aslp_default_voice.wav",
    prompt_text="猪携带该疾病，然后通过蚊子传染给人类。",
    lang="Chinese",
)
model.save("outputs/clone_python.wav", audio)
```

精品说话人播报应单独加载 CustomVoice 权重：

```python
from inference import M5TTS

model = M5TTS.from_pretrained("ASLP-lab/M5-TTS-50M-CustomVoice", device="cuda")
audio = model.infer("你好，今天的天气很好。", speaker_id=1)
model.save("outputs/speaker_python.wav", audio)
```

### 网页 Demo

体验在线 [M5-TTS-Space](https://huggingface.co/spaces/ASLP-lab/M5-TTS-Space)，或本地启动：

```bash
python app.py
```

浏览器打开 **http://localhost:7860**，提供参考音频和转写，输入目标文本后点击 **Generate**。服务端及模型参数见 `python app.py --help`。

### ONNX 推理

`infer_onnx.py` 支持同一份 ONNX 模型在 CPU 和 NVIDIA GPU 上运行，默认下载 `ASLP-lab/M5-TTS-50M-ONNX`。

#### CPU

```bash
python infer_onnx.py \
  --device cpu --threads 4 \
  --text "你好，欢迎使用我们的语音合成模型。" \
  --prompt-wav assets/aslp_default_voice.wav \
  --prompt-text "猪携带该疾病，然后通过蚊子传染给人类。" \
  --output outputs/onnx_cpu.wav
```

#### NVIDIA GPU

```bash
python infer_onnx.py \
  --device cuda --device-id 0 \
  --text "你好，欢迎使用我们的语音合成模型。" \
  --prompt-wav assets/aslp_default_voice.wav \
  --prompt-text "猪携带该疾病，然后通过蚊子传染给人类。" \
  --output outputs/onnx_gpu.wav
```

依赖文件安装的 `onnxruntime-gpu` 同时支持 CPU，并由 PyTorch 提供兼容的 CUDA 12/cuDNN 9 运行库。GPU 推理仍需兼容的 NVIDIA 驱动。同一环境不要再安装 `onnxruntime`。

添加 `--stream` 可将生成音频分块写入文件。脚本保存 WAV 及同名 JSON 统计文件，不覆盖已有输出。ONNX 当前提供参考音频克隆，精品说话人 ID 请使用 PyTorch API。CPU/GPU 的数值结果和速度可能不同。

### 导出 ONNX

从 Hugging Face 加载 M5 checkpoint 并导出：

```bash
python onnx/export.py \
  --model ASLP-lab/M5-TTS-50M-VoiceClone \
  --output outputs/onnx_export
```

导出器自动下载配套 codec，生成完整的 FP32 ONNX 模型包。`--model` 也可指定包含权重、配置、条件元数据及配套文本 tokenizer 的本地目录。输出目录必须不存在。

更新 M5 权重时，也可以复用已有的兼容 codec：

```bash
python onnx/export.py \
  --model ASLP-lab/M5-TTS-50M-VoiceClone \
  --reuse-codec ASLP-lab/M5-TTS-50M-ONNX \
  --output outputs/onnx_reexport
```

将导出目录作为 `infer_onnx.py` 的 `--model` 即可推理。缓存、版本和离线选项见 `python onnx/export.py --help`。

### Android ONNX Runtime 示例

原生 Android 示例位于 [`android/`](android/README.md)，支持 Android 9+ arm64 设备上的离线 ONNX 推理。构建 App 后，将兼容的 ONNX 模型打包为 `.m5model`，再导入 App。

JDK、Android SDK、APK 构建和模型打包方法见 [Android 指南](android/README.md)。

## 评测

以下表格保留 **Seed、MiniMax 和方言评测的全部模型配置与结果**，并补充最新的 VoiceJudge 评测和 CPU 性能实测。CER、WER、MER 单位为百分比，越低越好；SIM 为说话人相似度，越高越好。**加粗指标**表示该列所列结果中的最优值，方言结果按方言分别比较。Seed 和 MiniMax 表中，<ins>下划线指标</ins>表示该列次优值；并列结果使用相同标记。`—` 表示未报告的测量，不表示模型不支持该能力。

**Voice Cloning（音色克隆）**与 **Custom Voice（固定音色）**使用不同的说话人条件，分别报告，固定音色结果并非相同条件下的零样本克隆对比。具有多个预置音色的系统每种语言选择一个音色；未加脚注的 Custom Voice 配置使用单个固定音色。

### Seed-TTS Eval

测试集：[Seed-TTS Eval](https://github.com/BytedanceSpeech/seed-tts-eval)。下表包含报告中的全部 **12 行模型配置**。

沿用官方评测脚本：中文转写使用 Paraformer-zh，英文转写使用 OpenAI Whisper-large-v3；SIM 为经过说话人验证微调的 WavLM-large 提取的 embedding 余弦相似度。

| 模型 | 主模型参数量（M） | 设置 | Seed-ZH CER (%) ↓ | Seed-ZH SIM ↑ | Seed-EN WER (%) ↓ | Seed-EN SIM ↑ | Seed-Hard CER (%) ↓ | Seed-Hard SIM ↑ |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **M5-TTS (CPT)** | 50 | Voice Cloning | 1.70 | 0.64 | 3.60 | <ins>0.53</ins> | 20.92 | 0.57 |
| **M5-TTS (GRPO)** | 50 | Voice Cloning | 1.37 | <ins>0.65</ins> | 2.91 | <ins>0.53</ins> | 17.48 | <ins>0.59</ins> |
| **M5-TTS** | 50 | Custom Voice | <ins>1.26</ins> | — | <ins>1.88</ins> | — | 18.19 | — |
| Audio8-TTS Preview | 170 | Voice Cloning | **1.21** | **0.68** | 1.96 | **0.57** | 19.63 | **0.63** |
| MOSS-TTS-Nano | 100 | Voice Cloning | 3.62 | 0.64 | 4.50 | 0.50 | 15.77 | **0.63** |
| Pocket TTS | 100 | Voice Cloning | — | — | **1.86** | 0.47 | — | — |
| Hojo-TTS-Light | 80 | Voice Cloning | 2.15 | 0.40 | 3.63 | 0.17 | 21.10 | 0.36 |
| Pocket TTS | 100 | Custom Voice<sup>a</sup> | — | — | 1.90 | — | — | — |
| Kokoro-TTS | 82 | Custom Voice<sup>b</sup> | 1.40 | — | 2.04 | — | **6.33** | — |
| Hojo-TTS-Light | 40 | Custom Voice<sup>c</sup> | 1.78 | — | 2.97 | — | 22.70 | — |
| Sano TTS | 2.27 | Custom Voice<sup>d</sup> | 2.42 | — | 2.40 | — | <ins>7.08</ins> | — |
| Tiny TTS | 1.6 | Custom Voice | — | — | 3.30 | — | — | — |

参数量单位统一为**百万（M）**，不包含语音 codec。Audio8-TTS Preview 使用 [Preview-0.1B checkpoint](https://huggingface.co/Edge0/Audio8-TTS-Preview-0.1b)，报告中的主模型参数量为 170M。

预置音色选择：

- <sup>a</sup> Pocket TTS：EN `alba`。
- <sup>b</sup> Kokoro-TTS：ZH `zf_xiaobei`；EN `af_heart`。
- <sup>c</sup> Hojo-TTS-Light（40M）：ZH `hojo_zh_f_01`；EN `hojo_en_m_02`。
- <sup>d</sup> Sano TTS：ZH `Xiaoya`；EN `Amy large`。

**M5-TTS（GRPO）**的 LM 参数量为 MOSS-TTS-Nano 的一半，Seed-ZH CER 和 Seed-EN WER 分别相对降低 **62.2%** 和 **35.3%**。Nano 在 **Seed-Hard** 的 CER 和 SIM 上仍更好。相比 Audio8-TTS Preview，M5-TTS（GRPO）的 Seed-Hard CER 更低，但 Seed-ZH/EN 错误率更高，三个子集的 SIM 也更低。

### MiniMax 多语种评测

测试集：[MiniMax TTS-Multilingual-Test-Set](https://huggingface.co/datasets/MiniMaxAI/TTS-Multilingual-Test-Set)。以下两张表均包含报告中的全部 **8 行模型配置**。ZH / EN / DE / FR / JA / KO 分别为中文 / 英语 / 德语 / 法语 / 日语 / 韩语；ZH、JA、KO 使用 CER，EN、DE、FR 使用 WER。

MiniMax 的中文转写使用本地 [SeACo-Paraformer](https://arxiv.org/abs/2308.03266)，不额外加载 VAD 或标点恢复模型；英、德、法、日、韩使用 OpenAI Whisper-large-v3，并显式指定目标语言。SIM 与 Seed-TTS Eval 使用相同的 WavLM-large 模型和余弦相似度计算方式。

#### 内容准确度 — CER/WER (%) ↓

| 模型 | 设置 | ZH CER | EN WER | DE WER | FR WER | JA CER | KO CER |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **M5-TTS (CPT)** | Voice Cloning | 2.54 | 3.67 | 4.60 | 8.15 | 12.71 | 3.10 |
| **M5-TTS (GRPO)** | Voice Cloning | 2.05 | 3.06 | 3.81 | 8.14 | <ins>9.70</ins> | <ins>2.88</ins> |
| **M5-TTS** | Custom Voice | <ins>1.16</ins> | 2.24 | **0.70** | <ins>5.33</ins> | 10.27 | **2.09** |
| MOSS-TTS-Nano | Voice Cloning | 1.80 | 5.34 | 11.18 | 18.64 | 40.22 | 8.21 |
| Pocket TTS | Voice Cloning | — | 2.41 | 3.50 | 6.71 | — | — |
| Pocket TTS | Custom Voice<sup>a</sup> | — | <ins>1.99</ins> | 4.14 | 7.88 | — | — |
| Sano TTS | Custom Voice<sup>b</sup> | 1.41 | **1.85** | <ins>3.18</ins> | 8.91 | — | — |
| Kokoro-TTS | Custom Voice<sup>c</sup> | **1.03** | 2.58 | — | **4.79** | **3.72** | — |

#### 说话人相似度 — SIM ↑

| 模型 | 设置 | ZH | EN | DE | FR | JA | KO |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **M5-TTS (CPT)** | Voice Cloning | 0.70 | 0.62 | **0.63** | <ins>0.51</ins> | **0.72** | <ins>0.66</ins> |
| **M5-TTS (GRPO)** | Voice Cloning | <ins>0.71</ins> | <ins>0.63</ins> | **0.63** | **0.53** | **0.72** | <ins>0.66</ins> |
| **M5-TTS** | Custom Voice | — | — | — | — | — | — |
| MOSS-TTS-Nano | Voice Cloning | **0.73** | **0.65** | <ins>0.54</ins> | <ins>0.51</ins> | <ins>0.70</ins> | **0.67** |
| Pocket TTS | Voice Cloning | — | 0.61 | <ins>0.54</ins> | **0.53** | — | — |
| Pocket TTS | Custom Voice<sup>a</sup> | — | — | — | — | — | — |
| Sano TTS | Custom Voice<sup>b</sup> | — | — | — | — | — | — |
| Kokoro-TTS | Custom Voice<sup>c</sup> | — | — | — | — | — | — |

预置音色选择：

- <sup>a</sup> Pocket TTS：EN `alba`；DE `juergen`；FR `estelle`。
- <sup>b</sup> Sano TTS：ZH `Xiaoya`；EN `Amy large`；DE `Thorsten`；FR `Siwis`。
- <sup>c</sup> Kokoro-TTS：ZH `zf_xiaobei`；EN `af_heart`；FR `ff_siwis`；JA `jf_alpha`。

**M5-TTS（GRPO）**在六种语言中的**五种语言**上取得比 Nano 更低的内容错误率，EN / DE / FR / JA / KO 相对降幅分别为 **42.7% / 65.9% / 56.3% / 75.9% / 64.9%**。Nano 的中文 CER 仍更低；M5-TTS 的德语、法语和日语 SIM 更高。Pocket TTS 在已报告的 EN/DE/FR 克隆评测中错误率更低，M5-TTS 则在 EN/DE 的 SIM 上更高，FR 的 SIM 在当前显示精度下相同。

相比 CPT，GRPO 降低了三个 Seed 子集和六种 MiniMax 语言的错误率，SIM 在当前显示精度下保持或提高。这些结果反映完整后训练方案的效果，未单独验证各项奖励的贡献。

### 中文方言评测

#### 粤语（Yue）— MER (%) ↓ / SIM ↑

测试集：[WenetSpeech-Yue](https://aslp-lab.github.io/WenetSpeech-Yue/)。**混合错误率（MER）**以中文字符和英文单词为评测单位，三个系统均使用 U2pp-Conformer-Yue 进行 ASR 评测。

| 模型 | Easy MER (%) ↓ | Easy SIM ↑ | Hard MER (%) ↓ | Hard SIM ↑ |
| :--- | ---: | ---: | ---: | ---: |
| **M5-TTS (GRPO)** | **8.55** | 0.47 | **8.70** | 0.51 |
| CosyVoice2-Yue | 10.33 | **0.62** | 9.49 | **0.67** |
| Llasa-1B-Yue | 10.89 | 0.51 | 12.78 | 0.55 |

#### 四川话（Chuan）— CER (%) ↓ / SIM ↑

测试集：[WenetSpeech-Chuan](https://github.com/ASLP-lab/WenetSpeech-Chuan)。两个系统均使用 FireRedASR-AED 进行 ASR 评测。

| 模型 | Easy CER (%) ↓ | Easy SIM ↑ | Hard CER (%) ↓ | Hard SIM ↑ |
| :--- | ---: | ---: | ---: | ---: |
| **M5-TTS (GRPO)** | 5.08 | 0.58 | **7.78** | 0.59 |
| CosyVoice2-WSC | **4.28** | **0.64** | 8.78 | **0.65** |

M5-TTS 在粤语两个子集上的 MER 均为所列系统中最低，四川话 Hard CER 也低于 CosyVoice2-WSC；四川话 Easy CER 和方言说话人相似度仍有提升空间。

### 基于 VoiceJudge 的自然度评测

**VoiceJudge** 是内部的自然度、表现力等多维 A/B 评测模型，采用自监督语音表征与 Transformer 评分模块，使用约 **10 万对人工标注偏好数据**训练。本实验仅使用其**自然度**维度，在与训练数据严格去重的 **311 对音频**上比较 M5-TTS 与 MOSS-TTS-Nano，每对音频使用相同目标文本。

每对音频交换 A/B 顺序评测两次，统计前将 A/B 选择映射回实际模型名称。

| 指标 | 原始顺序：A=MOSS，B=M5 | 交换顺序：A=M5，B=MOSS |
| :--- | ---: | ---: |
| 有效评测对数 | 311 | 311 |
| **M5-TTS 胜出** | **264（84.89%）** | **269（86.50%）** |
| MOSS-TTS-Nano 胜出 | 47（15.11%） | 42（13.50%） |
| 平局 | 0 | 0 |
| M5-TTS 平均 reward | 0.2586 | 0.3231 |
| MOSS-TTS-Nano 平均 reward | -0.6542 | -0.6234 |

| 优选模型：原始顺序 → 交换顺序 | 对数 | 占比 |
| :--- | ---: | ---: |
| M5-TTS → M5-TTS | 255 | 81.99% |
| MOSS-TTS-Nano → MOSS-TTS-Nano | 33 | 10.61% |
| MOSS-TTS-Nano → M5-TTS | 14 | 4.50% |
| M5-TTS → MOSS-TTS-Nano | 9 | 2.89% |
| **优选模型一致（小计）** | **288** | **92.60%** |

两种顺序下 VoiceJudge 均更偏好 M5-TTS，交换顺序后胜率增加 **1.61 个百分点**。两次使用的是**同一批 311 对音频**，其中 **92.60%** 的音频对在交换顺序后保持相同的优选模型。表中 reward 为 VoiceJudge 输出分数，本实验将 VoiceJudge 用于评测。

### CPU 流式性能

| 模型 | 硬件 | CPU 核心数 | RTF ↓ | 预热后首包延迟 ↓ |
| :--- | :--- | ---: | ---: | ---: |
| **M5-TTS** | Intel Xeon Gold 6530 | 4 | 0.653 | 225 ms |

RTF 为合成耗时与生成音频时长之比；**RTF < 1** 表示生成速度快于实时播放。计时不含模型加载与预热。报告仅给出 M5-TTS 的此项 CPU 实测，未提供基线运行时对比；该结果不能直接代表手机或 ARM 平台表现。

## 方法概述

### 分层语音生成

<p align="left">
  <img src="assets/architecture.png" alt="M5-TTS backbone 与 MTP 分层架构" width="900">
</p>

时间 backbone 在每一帧预测 **codebook 0**，局部 MTP Transformer 随后自回归生成 **codebooks 1–15**。历史帧的码本 embedding 求和后进入时间建模，完整帧由 [Qwen3-TTS 语音 tokenizer](https://github.com/QwenLM/Qwen3-TTS)增量解码为波形。

### 训练与后训练

1. **基础预训练：** 约 **400,000 小时**多语种与区域语音，包含 Emilia、Emilia-YODAS、内部录音和 WenetSpeech 系列数据。
2. **持续预训练（CPT）：** 从预训练语料筛选约 **65,000 小时子集**，优先选取高质量录音、表现力语音和复杂发音样本。该部分为原语料子集，并非额外独立数据。
3. **GRPO：** 句级 **SIM** 和**停顿**奖励先在同一 prompt 的候选组内**分别归一化，再加权求和**，得到全局优势。ASR 编辑对齐先为目标文本的各字符分配编辑错误计数，再在每个字符位置对候选组进行归一化，得到相对优势。负反馈按字符距离指数衰减，向同一分句内更早的字符传播，并通过时间对齐投影到 codec 帧。这一 **CER 引导的局部过程反馈** 加到全局优势上用于策略更新，同一帧内 16 个 codec token 共享组合后的帧级优势。
4. **精品说话人微调：** 使用指定说话人的精选录音，训练具有一致音色的单人多语模型。

全局优势为 `1.0 × norm(SIM) + 0.5 × norm(Pause)`。两种奖励各自在同一 prompt 的候选组内归一化，**加权求和后不再次归一化**；CER 引导的局部过程优势权重为 **0.5**，KL 系数为 **0.01**，句级 CER 广播关闭。

预训练条件顺序为 `accent → language → text`，精品说话人微调时加入 speaker：`accent → speaker → language → text`。

<p align="left">
  <img src="assets/grpo.png" alt="GRPO：SIM 和停顿奖励分别组内归一化后加权，再结合从字符投影到 codec 帧的 CER 引导的局部过程反馈" width="900">
</p>

### 高效推理

报告中的优化包括局部解码器帧计算图、投影融合、KV 与参考语音缓存、有状态 codec 解码，以及 CPU 上自适应 **4 / 8 / 16 帧**调度。GPU 路径使用 BF16 TTS LM、FP32 codec 和 `torch.compile`；上方性能指标为四核 CPU 的实测。

## 当前限制与后续方向

当前限制包括 16 码本局部自回归解码的带宽开销、仅支持 24 kHz 输出、强表现力语音中的声学瑕疵，以及困难中文输入和方言音色保持方面的不足。

后续计划探索 **8 码本推理**、**48 kHz 输出**，改进困难样本筛选与数据配比，并支持对**情感、说话风格、笑声和叹息等副语言表达**的显式控制。

## 贡献者

<table>
  <tr>
    <td align="left" valign="middle" width="240" height="140">
      <a href="https://www.nwpu.edu.cn/"><img src="assets/logos/npu.png" alt="西北工业大学" width="220"></a>
    </td>
    <td align="left" valign="middle" width="240" height="140">
      <img src="assets/logos/pimei.jpg" alt="媲美科技 · LycheeAI Lab" width="120">
    </td>
    <td align="left" valign="middle" width="240" height="140">
      <img src="assets/logos/moonstep.jpg" alt="Moonstep AI" width="120">
    </td>
  </tr>
  <tr>
    <td align="left" valign="middle" width="240" height="140">
      <img src="assets/logos/wenet.png" alt="WeNet 开源社区" width="120">
    </td>
    <td align="left" valign="middle" width="240" height="140">
      <img src="assets/logos/nju.png" alt="南京大学" width="180">
    </td>
    <td align="left" valign="middle" width="240" height="140">
      <img src="assets/logos/originbeat.png" alt="Originbeat Inc." width="220">
    </td>
  </tr>
</table>

### 数据致谢

精品说话人模型使用了 [2021 多说话人多风格语音克隆挑战赛（M2VoC）](https://arxiv.org/abs/2104.01818)中的数据。感谢 **Fen Hong（Originbeat Inc.）** 对相关数据工作的支持。

## 许可证

代码与模型权重的许可范围以各发布文件、模型仓库和 Space 随附的许可证为准。

仓库内的 Qwen 12Hz codec 实现采用 Apache-2.0 许可证，见[第三方声明](m5/codec_qwen12hz/NOTICE)。

## 引用

如使用本工作，请引用技术报告：

```bibtex
@misc{m5tts2026,
  title = {{M5-TTS: A Mini-Scale Multilingual and Multi-Dialect Text-to-Speech Language Model with Multi-Token Prediction}},
  author = {Wenhao Li and Jingbin Hu and Shengming Ling and Xiaming Ren and Haoyu Zhang and Jinrui Liang and Hanke Xie and Qirui Zhan and Guobin Ma and Huakang Chen and Chunbo Hao and Jianguo Zheng and Wenkang Hou and Dake Guo and Yuepeng Jiang and Linhan Ma and Yunxiang Chen and Houdun Liu and Su Feng and Chuang Ding and Pengcheng Zhu and Liumeng Xue and Lei Xie},
  year = {2026},
  howpublished = {Technical report},
  url = {https://aslp-lab.github.io/M5-TTS-Demo-Page/m5-tts-v3.pdf}
}
```

## 联系我们

如有问题或反馈，欢迎通过邮箱 [wenhao.li@mail.nwpu.edu.cn](mailto:wenhao.li@mail.nwpu.edu.cn) 联系 **Wenhao Li**。也可通过本仓库的 [Issues](https://github.com/ASLP-lab/M5-TTS/issues) 交流，或访问 [ASLP@NPU 实验室官网](http://www.npu-aslp.org/english)。

<p align="left">
  <a href="http://www.npu-aslp.org/english"><img src="assets/logos/aslp.jpeg" alt="ASLP@NPU 音频、语音与语言处理实验室" width="500"></a>
</p>

欢迎加入 M5-TTS 微信交流群，交流使用问题与项目动态。扫描下方二维码，或点击图片查看原图。如二维码已过期，请通过邮箱联系我们获取新的邀请。

<p align="left">
  <a href="assets/wechat-group.jpg"><img src="assets/wechat-group.jpg" alt="M5-TTS 微信交流群二维码" width="320"></a>
</p>
