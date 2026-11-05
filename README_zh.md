<p align="center">
  <img src="m5-tts-logo.png" alt="M5-TTS 蜂鸟标志" width="440">
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
  <a href="https://aslp-lab.github.io/M5-TTS-Demo-Page/M5_TTS__A_Mini_Scale_Multilingual_and_Multi_Dialect_Text_to_Speech_Language_Model_with_Multi_Token_Prediction_v2.pdf">技术报告</a> ·
  <a href="https://aslp-lab.github.io/M5-TTS-Demo-Page/">音频演示</a> ·
  <a href="https://huggingface.co/spaces/ASLP-lab/M5-TTS-Space">在线体验</a> ·
  <a href="https://huggingface.co/collections/ASLP-lab/m5-tts">模型合集</a>
</p>

**M5-TTS** 是基于语言模型的轻量语音生成系统，支持多语种、中文方言、零样本音色克隆，以及精品说话人的单人多语合成。其 **50M LM 参数量不含语音 codec**，由时间维度 backbone 和局部多 Token 预测（MTP）模块组成。

**LM 参数量仅为 MOSS-TTS-Nano 的一半，Seed-ZH / Seed-EN 相对错误率分别降低 62.2% / 35.3%。** M5-TTS（GRPO）达到 **1.37% CER / 2.91% WER**，Nano 对应为 **3.62% / 4.50%**。在报告所测的四核 Xeon 配置下，CPU 流式合成达到 **RTF 0.653** 和 **225 ms 预热后首包延迟**。

## ✨ 核心特性

|  | 特性 | M5-TTS 能力 |
| :--- | :--- | :--- |
| **M₁** | **Mini** | 50M LM：30M 时间 backbone 与 20M 局部解码器；codec 参数单独计算。 |
| **M₂** | **Multilingual** | 中文、英语、德语、法语、日语、韩语。 |
| **M₃** | **Multi-Dialect** | 粤语（Yue）和四川话（Chuan），提供对应方言评测。 |
| **M₄** | **Multi-Token Prediction** | 16 个码本，codec 帧率为 12.5 Hz；分层进行时间建模和帧内生成。 |
| **M₅** | **Language Modeling** | 参考语音驱动的零样本克隆、精品说话人单人多语合成，以及 GRPO 后训练。 |

- **24 kHz 语音输出**，使用因果语音 codec 解码器。
- **CPU 流式推理**：结合计算图优化、缓存复用、有状态 codec 解码和自适应分块调度。
- **全局 + 局部后训练反馈**：说话人相似度与停顿奖励结合 token-level CER 过程优势。

## 🎧 在线体验

打开[在线 Space](https://huggingface.co/spaces/ASLP-lab/M5-TTS-Space)，上传参考语音及其准确转写，输入目标文本，再选择语言和可选口音（`yue` / `chuan`）。当前演示最多接受 250 个字符的目标文本。[音频演示页](https://aslp-lab.github.io/M5-TTS-Demo-Page/)提供多语种与方言对比样例。

在线 Space 使用 GPU，返回完整音频片段；下方 CPU 流式指标对应技术报告所评测的推理系统。

### 发布状态（2026-11-06）

| 资源 | 状态 |
| :--- | :--- |
| [技术报告](https://aslp-lab.github.io/M5-TTS-Demo-Page/M5_TTS__A_Mini_Scale_Multilingual_and_Multi_Dialect_Text_to_Speech_Language_Model_with_Multi_Token_Prediction_v2.pdf) | 已公开。 |
| [Audio demos](https://aslp-lab.github.io/M5-TTS-Demo-Page/) | 已公开。 |
| [Online Space](https://huggingface.co/spaces/ASLP-lab/M5-TTS-Space) | 已上线，Space 仓库提供[推理源码与一份 checkpoint](https://huggingface.co/spaces/ASLP-lab/M5-TTS-Space/tree/main)。 |
| [Model collection](https://huggingface.co/collections/ASLP-lab/m5-tts) / [model repository](https://huggingface.co/ASLP-lab/M5-TTS) | 模型仓库已建立，独立模型发布仍在准备中；当前模型仓库包含模型卡，尚无权重文件。 |
| 预训练、CPT 和 GRPO 代码 | 计划开放。 |
| 完整本地 / CPU 流式推理工程 | 计划在本仓库开放。 |
| 独立的克隆模型与精品说话人模型 | 计划开放；Space 中现有 checkpoint 未标明训练阶段。 |

## 📊 实验结果

以下表格完整收录技术报告**表 2–5 中所有模型配置及其测试结果**，并列出报告中的 CPU 性能实测。CER/WER 单位为百分比，越低越好；SIM 为说话人相似度，越高越好。**加粗指标**表示该列所列结果中的最优值，方言结果按方言分别比较；`—` 表示未报告的测量，不表示模型不支持该能力。

**Voice Cloning（音色克隆）**与 **Custom Voice（固定音色）**使用不同的说话人条件，分别报告，固定音色结果并非相同条件下的零样本克隆对比。具有多个预置音色的系统每种语言选择一个音色；未加脚注的 Custom Voice 配置使用单个固定音色。

### Seed-TTS Eval

测试集：[Seed-TTS Eval](https://github.com/BytedanceSpeech/seed-tts-eval)。下表包含报告中的全部 **12 行模型配置**。

| 模型 | 主模型参数量（M） | 设置 | Seed-ZH CER (%) ↓ | Seed-ZH SIM ↑ | Seed-EN WER (%) ↓ | Seed-EN SIM ↑ | Seed-Hard CER (%) ↓ | Seed-Hard SIM ↑ |
| :--- | :---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **M5-TTS (CPT)** | 50 | Voice Cloning | 1.70 | 0.64 | 3.60 | 0.53 | 20.92 | 0.57 |
| **M5-TTS (GRPO)** | 50 | Voice Cloning | 1.37 | 0.65 | 2.91 | 0.53 | 17.48 | 0.59 |
| **M5-TTS** | 50 | Custom Voice | 1.26 | — | 1.88 | — | 18.19 | — |
| Audio8-TTS Preview | 170 | Voice Cloning | **1.21** | **0.68** | 1.96 | **0.57** | 19.63 | **0.63** |
| MOSS-TTS-Nano | 100 | Voice Cloning | 3.62 | 0.64 | 4.50 | 0.50 | 15.77 | **0.63** |
| Pocket TTS | 100 | Voice Cloning | — | — | **1.86** | 0.47 | — | — |
| Hojo-TTS-Light | 80 | Voice Cloning | 2.15 | 0.40 | 3.63 | 0.17 | 21.10 | 0.36 |
| Pocket TTS | 100 | Custom Voice<sup>a</sup> | — | — | 1.90 | — | — | — |
| Kokoro-TTS | 82 | Custom Voice<sup>b</sup> | 1.40 | — | 2.04 | — | **6.33** | — |
| Hojo-TTS-Light | 40 | Custom Voice<sup>c</sup> | 1.78 | — | 2.97 | — | 22.70 | — |
| Sano TTS | 2.27 | Custom Voice<sup>d</sup> | 2.42 | — | 2.40 | — | 7.08 | — |
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

#### 内容准确度 — CER/WER (%) ↓

| 模型 | 设置 | ZH CER | EN WER | DE WER | FR WER | JA CER | KO CER |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **M5-TTS (CPT)** | Voice Cloning | 2.54 | 3.67 | 4.60 | 8.15 | 12.71 | 3.10 |
| **M5-TTS (GRPO)** | Voice Cloning | 2.05 | 3.06 | 3.81 | 8.14 | 9.70 | 2.88 |
| **M5-TTS** | Custom Voice | 1.16 | 2.24 | **0.70** | 5.33 | 10.27 | **2.09** |
| MOSS-TTS-Nano | Voice Cloning | 1.80 | 5.34 | 11.18 | 18.64 | 40.22 | 8.21 |
| Pocket TTS | Voice Cloning | — | 2.41 | 3.50 | 6.71 | — | — |
| Pocket TTS | Custom Voice<sup>a</sup> | — | 1.99 | 4.14 | 7.88 | — | — |
| Sano TTS | Custom Voice<sup>b</sup> | 1.41 | **1.85** | 3.18 | 8.91 | — | — |
| Kokoro-TTS | Custom Voice<sup>c</sup> | **1.03** | 2.58 | — | **4.79** | **3.72** | — |

#### 说话人相似度 — SIM ↑

| 模型 | 设置 | ZH | EN | DE | FR | JA | KO |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **M5-TTS (CPT)** | Voice Cloning | 0.70 | 0.62 | **0.63** | 0.51 | **0.72** | 0.66 |
| **M5-TTS (GRPO)** | Voice Cloning | 0.71 | 0.63 | **0.63** | **0.53** | **0.72** | 0.66 |
| **M5-TTS** | Custom Voice | — | — | — | — | — | — |
| MOSS-TTS-Nano | Voice Cloning | **0.73** | **0.65** | 0.54 | 0.51 | 0.70 | **0.67** |
| Pocket TTS | Voice Cloning | — | 0.61 | 0.54 | **0.53** | — | — |
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

| 方言 | 模型 | ASR 评测模型 | Easy CER (%) ↓ | Easy SIM ↑ | Hard CER (%) ↓ | Hard SIM ↑ |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| Yue | **M5-TTS (GRPO)** | U2pp-Conformer-Yue | **8.55** | 0.47 | **8.70** | 0.51 |
| Yue | CosyVoice2-Yue | U2pp-Conformer-Yue | 10.33 | **0.62** | 9.49 | **0.67** |
| Yue | Llasa-1B-Yue | U2pp-Conformer-Yue | 10.89 | 0.51 | 12.78 | 0.55 |
| Chuan | **M5-TTS (GRPO)** | FireRedASR-AED | 5.08 | 0.58 | **7.78** | 0.59 |
| Chuan | CosyVoice2-WSC | FireRedASR-AED | **4.28** | **0.64** | 8.78 | **0.65** |

Yue 为粤语，Chuan 为四川话，同一方言的所有系统使用相同 ASR。M5-TTS 在粤语两个子集上的 CER 均为所列系统中最低，四川话 Hard CER 也低于 CosyVoice2-WSC；四川话 Easy CER 和方言 SIM 仍落后于对应基线。

### 主观评测

**10 名听众**，使用 **1–5 分**评分。N-MOS：自然度；S-MOS：说话人相似度；I-MOS：可懂度；P-MOS：韵律与表现力。

| 模型 | N-MOS ↑ | S-MOS ↑ | I-MOS ↑ | P-MOS ↑ |
| :--- | ---: | ---: | ---: | ---: |
| Kokoro-TTS | 2.85 | — | **3.10** | 2.60 |
| MOSS-TTS-Nano | **3.00** | 2.85 | 2.65 | 2.80 |
| **M5-TTS** | 2.90 | **3.05** | 2.95 | **2.85** |

M5-TTS 的相似度、可懂度和韵律评分高于 Nano，自然度评分则较低；Kokoro-TTS 的 I-MOS 最高。以上为描述性数值差异，报告未给出统计显著性结论。

### CPU 流式性能

| 模型 | 硬件 | CPU 核心数 | RTF ↓ | 预热后首包延迟 ↓ |
| :--- | :--- | :---: | ---: | ---: |
| **M5-TTS** | Intel Xeon Gold 6530 | 4 | 0.653 | 225 ms |

RTF 为合成耗时与生成音频时长之比；**RTF < 1** 表示生成速度快于实时播放。计时不含模型加载与预热。报告仅给出 M5-TTS 的此项 CPU 实测，未提供基线运行时对比；该结果不能直接代表手机或 ARM 平台表现。

## 🧩 方法概述

### 分层语音生成

<p align="center">
  <img src="https://aslp-lab.github.io/M5-TTS-Demo-Page/assets/architecture.png" alt="M5-TTS backbone 与 MTP 分层架构" width="900">
</p>

时间 backbone 在每一帧预测 **codebook 0**，局部 MTP Transformer 随后自回归生成 **codebooks 1–15**。历史帧的码本 embedding 求和后进入时间建模，完整帧由 [Qwen3-TTS 语音 tokenizer](https://github.com/QwenLM/Qwen3-TTS)增量解码为波形。

### 训练与后训练

1. **基础预训练：** 约 **400,000 小时**多语种与区域语音，包含 Emilia、Emilia-YODAS、内部录音和 WenetSpeech 系列数据。
2. **持续预训练（CPT）：** 从预训练语料筛选约 **65,000 小时子集**，优先选取高质量录音、表现力语音和复杂发音样本。该部分为原语料子集，并非额外独立数据。
3. **GRPO：** 句级 **SIM + 停顿**反馈构成全局优势；ASR 编辑对齐产生逐字符的组内相对反馈，将负优势按字符距离指数衰减传播至同一分句内更早的字符，再映射到 codec 帧。局部 CER 过程优势**加到**全局优势上，同一帧的全部码本共享组合后的信号。
4. **精品说话人微调：** 使用指定说话人的精选录音，训练具有一致音色的单人多语模型。

报告中全局奖励的权重为 SIM **1.0**、停顿 **0.5**，局部 CER 优势权重为 **0.5**，KL 系数为 **0.01**；句级 CER 广播关闭。预训练条件顺序为 `accent → language → text`，精品说话人微调时加入 speaker：`accent → speaker → language → text`。

### 高效推理

报告中的优化包括局部解码器帧计算图、投影融合、KV 与参考语音缓存、有状态 codec 解码，以及 CPU 上自适应 **4 / 8 / 16 帧**调度。GPU 路径使用 BF16 TTS LM、FP32 codec 和 `torch.compile`；上方性能指标为四核 CPU 的实测。

## 当前限制与后续方向

当前限制包括 16 码本局部自回归解码的带宽开销、仅支持 24 kHz 输出、强表现力语音中的声学瑕疵，以及困难中文输入和方言音色保持方面的不足。

后续计划探索 **8 码本推理**、**48 kHz 输出**，改进困难样本筛选与数据配比，并支持对**情感、说话风格、笑声和叹息等副语言表达**的显式控制。

## 许可证

[Hugging Face 模型卡](https://huggingface.co/ASLP-lab/M5-TTS/blob/main/README.md)标记为 `Apache-2.0`。GitHub 代码与 Space 中的文件尚未提供独立许可证。

## 引用

如使用本工作，请引用技术报告：

```bibtex
@misc{m5tts2026,
  title = {{M5-TTS: A Mini-Scale Multilingual and Multi-Dialect Text-to-Speech Language Model with Multi-Token Prediction}},
  author = {Wenhao Li and Jingbin Hu and Shengming Ling and Xiaming Ren and Haoyu Zhang and Jinrui Liang and Hanke Xie and Qirui Zhan and Guobin Ma and Huakang Chen and Chunbo Hao and Jianguo Zheng and Wenkang Hou and Dake Guo and Yuepeng Jiang and Linhan Ma and Yunxiang Chen and Houdun Liu and Su Feng and Chuang Ding and Pengcheng Zhu and Liumeng Xue and Lei Xie},
  year = {2026},
  howpublished = {Technical report},
  url = {https://github.com/ASLP-lab/M5-TTS}
}
```
