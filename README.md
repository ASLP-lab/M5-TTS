<p align="center">
  <img src="assets/m5-tts-logo.png" alt="M5-TTS hummingbird logo" width="440">
</p>

<h1 align="center">M5-TTS</h1>

<p align="center">
  <b>Mini, Multilingual, Multi-Accent, Multi-Token Prediction, and LLM-based Modeling</b>
</p>

<p align="center">
  Compact speech models · Multilingual voices · CPU streaming
</p>

<p align="center">
  English | <a href="README_zh.md">简体中文</a><br>
  <a href="https://huggingface.co/collections/ASLP-lab/m5-tts">🤗 Model Collection</a> ·
  <a href="https://huggingface.co/ASLP-lab/M5-TTS">🤗 Model Repository</a>
</p>

**M5-TTS** explores how small an LLM-based text-to-speech system can be while retaining clear speech, voice cloning, and multilingual generation. We focus on compact models, practical CPU inference, and streaming synthesis, with SFT and GRPO experiments to improve generation quality.

## ✨ Features

| | Feature | What M5-TTS explores |
| :--- | :--- | :--- |
| **M₁** | **Mini** | Model scaling experiments including a **50M-scale configuration (30M + 20M)**, targeting efficient deployment and reliable everyday speech. |
| **M₂** | **Multilingual** | Evaluation across **Chinese, English, German, French, Japanese, and Korean**. |
| **M₃** | **Multi-Accent** | Chinese dialect experiments covering **Sichuanese, Wu, and Cantonese**, with dialect fine-tuning and GRPO evaluation. |
| **M₄** | **Multi-Token Prediction** | MTP for speech token modeling, targeting efficient generation in compact models. |
| **M₅** | **LLM-based Modeling** | Language-model-based speech generation, with **voice cloning**, dedicated speaker fine-tuning, and GRPO post-training experiments. |

- **Streaming on CPU and GPU.** Four-core CPU streaming reaches **RTF 0.653** with a **225 ms** warm first chunk in the recorded test below.
- **24 kHz speech output.** Higher-resolution output and richer instruction control are on the roadmap.

The model sizes above refer to experimental configurations. The parameter count and included components of each downloadable release will be listed with its checkpoint. A standalone MTP speedup or ablation result has not yet been reported.

## 📊 Current Results

These are development results. Tables use different experimental configurations; they do not describe one shared release checkpoint. Missing values and release mappings are left blank.

### Streaming and CPU Inference

<!-- Source: supplied development notes, p. 24, embedded performance table. -->

| Hardware | Mode | RTF ↓ | Warm first chunk ↓ |
| :--- | :--- | ---: | ---: |
| Intel Xeon Gold 6530, 1 CPU core | Streaming | 1.587 | 541 ms |
| Intel Xeon Gold 6530, 4 CPU cores | Streaming | **0.653** | **225 ms** |
| Intel Xeon Gold 6530, 4 CPU cores | Non-streaming | 0.627 | |
| NVIDIA RTX 4090 | Streaming | **0.098** | **20.1 ms** |

**Test conditions:** Each configuration uses the same 10 sentences, excluding warmup. CPU inference uses FP32 TTS and FP32 codec; GPU inference uses PyTorch BF16 with compilation. The CPU run's approximately 6.2 seconds of preparation for five voices is excluded. First-chunk audio durations are 220 ms on CPU and 60 ms on GPU. These are recorded measurements, not a new rerun. RTF is inference time divided by generated audio duration; values below 1 indicate faster-than-real-time generation.

| Checkpoint | Batch size | OS / runtime versions |
| :--- | :--- | :--- |
| | | |

### Speech Intelligibility and Speaker Similarity

<!-- Source: supplied development notes, p. 6, independent table labeled 2.0-100step. -->

Recorded experiment: **`2.0-100step`**.

| Benchmark | Metric ↓ | Error rate | Speaker similarity ↑ |
| :--- | :--- | ---: | ---: |
| Seed-ZH | CER | **1.46%** | 0.6453 |
| Seed-EN | WER | **3.13%** | 0.5272 |
| Seed-Hard | CER | 18.41% | 0.5854 |

CER: character error rate. WER: word error rate. Speaker similarity is reported on the original 0–1 scale.

| Full checkpoint | ASR evaluator | Speaker encoder | Sample count |
| :--- | :--- | :--- | :--- |
| | | | |

<details>
<summary>Earlier CPT → GRPO experiments</summary>

<!-- Source: supplied development notes, p. 6, white Seed table. -->

| Experiment | Seed-ZH CER ↓ | ZH SIM ↑ | Seed-EN WER ↓ | EN SIM ↑ |
| :--- | ---: | ---: | ---: | ---: |
| `cpt_model_4_109620` | 3.81% | 0.6548 | 4.72% | 0.5572 |
| `grpo_cn_50step` | 3.55% | 0.6576 | 4.00% | 0.5584 |
| `grpo_cn_100step` | 2.74% | 0.6641 | 4.86% | 0.5611 |

These runs show improvements in Chinese intelligibility. English WER does not improve consistently. `grpo_cn_100step` and `2.0-100step` are separate experiment labels.

</details>

### Multilingual Synthesis

<!-- Source: supplied development notes, p. 10, Minimax multilingual table labeled M5. -->

Benchmark: **Minimax multilingual**; model label in the experiment log: **M5**.

| Language / variety | Metric ↓ | Error rate |
| :--- | :--- | ---: |
| Chinese | CER | 2.3983% |
| English | WER | 3.7817% |
| German | WER | 3.5912% |
| French | WER | 16.7612% |
| Japanese | CER | 11.1923% |
| Korean | CER | 4.6992% |
| Cantonese | CER | 17.9779% |

| Checkpoint | ASR evaluator | Sample count |
| :--- | :--- | :--- |
| | | |

<details>
<summary>Dialect experiment records</summary>

<!-- Source: supplied development notes, pp. 12–13, row labeled grpo. -->

| Benchmark | Subset | Metric | Recorded value |
| :--- | :--- | :--- | ---: |
| WSC-Eval-TTS | Easy | | 4.44 |
| WSC-Eval-TTS | Hard | | 6.63 |
| WSW | Easy | | 17.15 |
| WSW | Hard | | 67.00 |
| WSYue-TTS-eval | Base | | 7.06 |
| WSYue-TTS-eval | Coverage | | 8.18 |

The source row is labeled `grpo`; its metric definition and checkpoint are not specified. Values are retained as recorded, without assigning a metric or percentage unit. Wu synthesis, especially the hard subset, remains an area for improvement.

</details>

## 🤗 Models and Resources

| Resource | Link |
| :--- | :--- |
| Hugging Face collection | [ASLP-lab / M5-TTS](https://huggingface.co/collections/ASLP-lab/m5-tts) |
| Model repository | [ASLP-lab/M5-TTS](https://huggingface.co/ASLP-lab/M5-TTS) |
| Technical report | |
| Audio demo | |
| Inference code | |

The Hugging Face repository currently contains the model card. Checkpoint files, configurations, and verified setup instructions will be added as they become available.

| Checkpoint file | Parameters | Supported languages / dialects | Download |
| :--- | :--- | :--- | :--- |
| | | | |

### Installation

<!-- Add verified environment requirements and installation commands here. -->

### Inference

<!-- Add a runnable example using the released checkpoint and actual inference API here. -->

## 🎧 Audio Examples

| Capability | Reference audio | Generated audio |
| :--- | :--- | :--- |
| Voice cloning | | |
| Multilingual synthesis | | |
| Dialect synthesis | | |
| Dedicated speaker voices | | |

## 🎯 Goals and Roadmap

Our first priority is **a compact TTS system with clear pronunciation, useful voice cloning, multilingual coverage, and practical streaming on CPU**.

- [x] Model-size exploration, including 50M-scale configurations
- [x] Multilingual and Chinese dialect experiments
- [x] Voice-cloning and dedicated-speaker training experiments
- [x] GRPO experiments and preliminary streaming / CPU measurements
- [x] Hugging Face model entry
- [ ] Upload model weights and configurations, with checkpoint-to-result mappings
- [ ] Release inference code, reproducible evaluation, and audio demos
- [ ] Improve hard Chinese text, Wu synthesis, and prosody / pauses
- [ ] Evaluate single-speaker generation across multiple dialects
- [ ] Add emotion / style tags, instruction control, and pronunciation / pinyin correction
- [ ] Evaluate non-verbal vocalization control
- [ ] Optimize ONNX, Transformers, vLLM / SGLang integration and integer quantization
- [ ] Validate mobile and edge-device deployment
- [ ] Explore distillation and on-policy distillation
- [ ] Explore 48 kHz stereo output through a higher-resolution vocoder

## License

The [Hugging Face model card](https://huggingface.co/ASLP-lab/M5-TTS/blob/main/README.md) is tagged `Apache-2.0`.

| Component | License |
| :--- | :--- |
| Repository code | |

## Citation

<!-- Add the public technical report and its verified BibTeX entry here. -->
