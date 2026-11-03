<p align="center">
  <img src="m5-tts-logo.png" alt="M5-TTS 蜂鸟标志" width="440">
</p>

<h1 align="center">M5-TTS</h1>

<p align="center">
  <b>Mini, Multilingual, Multi-Accent, Multi-Token Prediction, and LLM-based Modeling</b>
</p>

<p align="center">
  <b>50M LM · 多语种与多方言 · 音色克隆 · 流式合成</b>
</p>

<p align="center">
  <a href="README.md">English</a> | 简体中文<br>
  <a href="https://huggingface.co/collections/ASLP-lab/m5-tts">🤗 模型合集</a> ·
  <a href="https://huggingface.co/ASLP-lab/M5-TTS">🤗 模型仓库</a>
</p>

**M5-TTS** 是一个基于语言模型的语音合成项目，采用 **50M 参数的语言模型**，支持多语种与中文方言合成、音色克隆和流式推理。我们的目标是以小模型生成清晰、自然的语音，并实现实用的 CPU / GPU 部署。

**LM 参数量仅为 MOSS-TTS-Nano 的一半：** M5-TTS 的 LM 为 **50M**，MOSS-TTS-Nano 为 **100M**。在 Seed-ZH / Seed-EN 上，M5-TTS 达到 **1.46% CER / 3.13% WER**，Nano 对应为 **3.62% / 4.50%**。

## ✨ 核心特性

| | 特性 | M5-TTS |
| :--- | :--- | :--- |
| **M₁** | **Mini：小型化** | **50M LM 参数**，仅为 MOSS-TTS-Nano LM 参数量的一半。 |
| **M₂** | **Multilingual：多语种** | 已评测**中文、英语、德语、法语、日语、韩语**六种语言的语音合成。 |
| **M₃** | **Multi-Accent：多口音 / 多方言** | 覆盖**川渝方言、吴语、粤语**的中文方言合成。 |
| **M₄** | **Multi-Token Prediction：多 Token 预测** | 采用 MTP 进行语音 Token 建模，服务于紧凑模型的语音生成。 |
| **M₅** | **LLM-based Modeling：语言模型驱动** | 基于语言模型生成语音，支持**音色克隆、精品发言人**与 GRPO 后训练。 |

- **CPU / GPU 流式合成：** 下方性能记录中，四核 CPU 可实现实时流式生成。
- **24 kHz 语音输出：** 以小模型服务多语种合成与音色克隆应用。

## 📊 与 MOSS-TTS-Nano 对比

| 模型 | LM 参数量，不含 codec ↓ |
| :--- | ---: |
| [MOSS-TTS-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M) | 100M |
| **M5-TTS** | **50M** |

每项成对比较中，**加粗**表示较优结果。空白表示暂无结果或评测信息；各 benchmark 沿用对应记录中的评测配置。

### Seed：发音准确度与音色相似度

<!-- 来源：提供的开发记录第 6 页，标记为 2.0-100step 的成对比较表。 -->

| 测试集 | 指标 | MOSS-TTS-Nano | M5-TTS |
| :--- | :--- | ---: | ---: |
| Seed-ZH | CER ↓ | 3.62% | **1.46%** |
| Seed-ZH | SIM ↑ | 0.6421 | **0.6453** |
| Seed-EN | WER ↓ | 4.50% | **3.13%** |
| Seed-EN | SIM ↑ | 0.5007 | **0.5272** |
| Seed-Hard | CER ↓ | **15.77%** | 18.41% |
| Seed-Hard | SIM ↑ | **0.6250** | 0.5854 |

CER 为字错误率，WER 为词错误率；SIM 为音色相似度，保留原记录的 0–1 数值。

### CV3：多语种合成

<!-- 来源：提供的开发记录第 6 页，双方均使用 Qwen3-ASR 的 CV3 对照表。不混入原 ASR 列或第 10 页无模型标识的记录。 -->

双方统一使用 **Qwen3-ASR-1.7B**、相同参考文本与评分口径，以下为语料级错误率。

| 语言 | 指标 ↓ | MOSS-TTS-Nano | M5-TTS |
| :--- | :--- | ---: | ---: |
| 中文 | CER | **3.0910%** | 6.4214% |
| 英语 | WER | **8.0996%** | 10.2723% |
| 德语 | WER | 19.6214% | **12.1045%** |
| 法语 | WER | 33.3211% | **21.9895%** |
| 日语 | CER | 40.8071% | **19.4897%** |
| 韩语 | CER | 18.4907% | **11.1277%** |

### Minimax Multilingual：多语种合成

<!-- 来源：提供的开发记录第 10 页，明确标注 M5 / MOSS-TTS-Nano 的对比表。 -->

| 语言 / 变体 | 指标 ↓ | MOSS-TTS-Nano | M5-TTS |
| :--- | :--- | ---: | ---: |
| 中文 | CER | **1.1470%** | 2.3983% |
| 英语 | WER | 4.4841% | **3.7817%** |
| 德语 | WER | 9.0470% | **3.5912%** |
| 法语 | WER | **15.0593%** | 16.7612% |
| 日语 | CER | 35.8611% | **11.1923%** |
| 韩语 | CER | 10.6874% | **4.6992%** |
| 粤语 | CER | | 17.9779% |

### 中文方言评测

<!-- 来源：提供的开发记录第 12–13 页，标记为 grpo 的记录行。缺少指标定义与 Nano 结果，不推断指标、百分比单位或 Nano 数值。 -->

| 测试集 | 子集 | 指标 | MOSS-TTS-Nano | M5-TTS |
| :--- | :--- | :--- | ---: | ---: |
| WSC-Eval-TTS | Easy | | | 4.44 |
| WSC-Eval-TTS | Hard | | | 6.63 |
| WSW | Easy | | | 17.15 |
| WSW | Hard | | | 67.00 |
| WSYue-TTS-eval | Base, with period | | | 7.06 |
| WSYue-TTS-eval | Coverage | | | 8.18 |

### 流式合成与推理速度

<!-- 来源：提供的开发记录第 24 页，M5 / Nano 成对性能表。 -->

| 硬件 | 模式 | Nano RTF ↓ | M5 RTF ↓ | Nano 首包 ↓ | M5 首包 ↓ |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Xeon Gold 6530，单核 CPU | 流式 | **1.054** | 1.587 | **330 ms** | 541 ms |
| Xeon Gold 6530，四核 CPU | 流式 | **0.372** | 0.653 | **127 ms** | 225 ms |
| Xeon Gold 6530，四核 CPU | 非流式 | **0.335** | 0.627 | | |
| RTX 4090 | 流式 | 0.206 | **0.098** | 45.8 ms | **20.1 ms** |

**测试条件：** 每组对比使用相同的 10 句文本，不计 warmup。首包为预热后指标；M5 CPU 测试未计入五个音色约 6.2 秒的准备时间。M5 CPU 使用 FP32 TTS + FP32 codec；GPU 上 Nano 使用 ONNX CUDA FP32，M5 使用 PyTorch BF16 + 编译优化。首包音频长度不同：CPU 上 Nano / M5 为 80 / 220 ms，GPU 上为 80 / 60 ms。以上比较对应所列部署配置。RTF 为推理耗时与生成音频时长之比，小于 1 表示生成速度快于实时播放。

## 🤗 模型与资源

| 资源 | 链接 |
| :--- | :--- |
| Hugging Face 合集 | [ASLP-lab / M5-TTS](https://huggingface.co/collections/ASLP-lab/m5-tts) |
| 模型仓库 | [ASLP-lab/M5-TTS](https://huggingface.co/ASLP-lab/M5-TTS) |
| 技术报告 | |
| 音频 Demo | |
| 推理代码 | |

| Checkpoint 文件 | LM 参数量 | 下载 |
| :--- | ---: | :--- |
| | 50M | |

### 安装

<!-- 在此补充已验证的环境要求与安装命令。 -->

### 推理

<!-- 在此补充基于实际发布 checkpoint 与推理接口的可运行示例。 -->

## 🎧 音频示例

| 能力 | 参考音频 | 合成音频 |
| :--- | :--- | :--- |
| 音色克隆 | | |
| 多语种合成 | | |
| 方言合成 | | |
| 精品发言人 | | |

## 许可证

[Hugging Face 模型卡](https://huggingface.co/ASLP-lab/M5-TTS/blob/main/README.md) 标记为 `Apache-2.0`。

| 组件 | 许可证 |
| :--- | :--- |
| 仓库代码 | |

## 引用

<!-- 在此补充公开技术报告与核实后的 BibTeX。 -->
