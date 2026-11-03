<p align="center">
  <img src="m5-tts-logo.png" alt="M5-TTS hummingbird logo" width="440">
</p>

<h1 align="center">M5-TTS</h1>

<p align="center">
  <b>Mini, Multilingual, Multi-Accent, Multi-Token Prediction, and LLM-based Modeling</b>
</p>

<p align="center">
  <b>50M LM · Multilingual & Multi-Dialect · Voice Cloning · Streaming</b>
</p>

<p align="center">
  English | <a href="README_zh.md">简体中文</a><br>
  <a href="https://huggingface.co/collections/ASLP-lab/m5-tts">🤗 Model Collection</a> ·
  <a href="https://huggingface.co/ASLP-lab/M5-TTS">🤗 Model Repository</a>
</p>

**M5-TTS** is an LLM-based text-to-speech project featuring a **50M-parameter language model**, multilingual and Chinese dialect synthesis, voice cloning, and streaming inference. Our goal is clear, natural speech from a compact model that is practical to deploy on CPU and GPU.

**Half the LM parameters of MOSS-TTS-Nano:** M5-TTS uses a **50M LM**, compared with **100M** for MOSS-TTS-Nano. On Seed-ZH / Seed-EN, M5-TTS achieves **1.46% CER / 3.13% WER**, compared with **3.62% / 4.50%** for Nano.

## ✨ Features

| | Feature | M5-TTS |
| :--- | :--- | :--- |
| **M₁** | **Mini** | **50M LM parameters**, half the LM parameter count of MOSS-TTS-Nano. |
| **M₂** | **Multilingual** | Speech synthesis evaluated in **Chinese, English, German, French, Japanese, and Korean**. |
| **M₃** | **Multi-Accent** | Chinese dialect synthesis covering **Sichuanese, Wu, and Cantonese**. |
| **M₄** | **Multi-Token Prediction** | MTP-based speech token modeling for compact speech generation. |
| **M₅** | **LLM-based Modeling** | Language-model-based speech generation with **voice cloning**, dedicated speaker voices, and GRPO post-training. |

- **CPU and GPU streaming:** Real-time streaming on a four-core CPU in the recorded benchmark below.
- **24 kHz speech output:** Multilingual synthesis and voice-cloning applications from one compact TTS project.

## 📊 Comparison with MOSS-TTS-Nano

| Model | LM parameters, excluding codec ↓ |
| :--- | ---: |
| [MOSS-TTS-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M) | 100M |
| **M5-TTS** | **50M** |

**Bold** marks the better result in each paired comparison. Empty cells indicate unavailable results or evaluation information. Each benchmark follows its own recorded evaluation configuration.

### Seed: Intelligibility and Speaker Similarity

<!-- Source: supplied development notes, p. 6, paired table labeled 2.0-100step. -->

| Benchmark | Metric | MOSS-TTS-Nano | M5-TTS |
| :--- | :--- | ---: | ---: |
| Seed-ZH | CER ↓ | 3.62% | **1.46%** |
| Seed-ZH | SIM ↑ | 0.6421 | **0.6453** |
| Seed-EN | WER ↓ | 4.50% | **3.13%** |
| Seed-EN | SIM ↑ | 0.5007 | **0.5272** |
| Seed-Hard | CER ↓ | **15.77%** | 18.41% |
| Seed-Hard | SIM ↑ | **0.6250** | 0.5854 |

CER: character error rate. WER: word error rate. SIM: speaker similarity, reported on the original 0–1 scale.

### CV3: Multilingual Synthesis

<!-- Source: supplied development notes, p. 6, CV3 comparison using Qwen3-ASR for both models. Do not mix in the original-ASR column or unlabeled rows on p. 10. -->

Both models are evaluated with **Qwen3-ASR-1.7B**, the same reference text, and the same scoring procedure. Results are corpus-level error rates.

| Language | Metric ↓ | MOSS-TTS-Nano | M5-TTS |
| :--- | :--- | ---: | ---: |
| Chinese | CER | **3.0910%** | 6.4214% |
| English | WER | **8.0996%** | 10.2723% |
| German | WER | 19.6214% | **12.1045%** |
| French | WER | 33.3211% | **21.9895%** |
| Japanese | CER | 40.8071% | **19.4897%** |
| Korean | CER | 18.4907% | **11.1277%** |

### Minimax Multilingual

<!-- Source: supplied development notes, p. 10, explicit M5 / MOSS-TTS-Nano comparison. -->

| Language / variety | Metric ↓ | MOSS-TTS-Nano | M5-TTS |
| :--- | :--- | ---: | ---: |
| Chinese | CER | **1.1470%** | 2.3983% |
| English | WER | 4.4841% | **3.7817%** |
| German | WER | 9.0470% | **3.5912%** |
| French | WER | **15.0593%** | 16.7612% |
| Japanese | CER | 35.8611% | **11.1923%** |
| Korean | CER | 10.6874% | **4.6992%** |
| Cantonese | CER | | 17.9779% |

### Chinese Dialect Benchmarks

<!-- Source: supplied development notes, pp. 12–13, row labeled grpo. Metric definitions and Nano results are absent; do not infer a metric, percentage unit, or Nano score. -->

| Benchmark | Subset | Metric | MOSS-TTS-Nano | M5-TTS |
| :--- | :--- | :--- | ---: | ---: |
| WSC-Eval-TTS | Easy | | | 4.44 |
| WSC-Eval-TTS | Hard | | | 6.63 |
| WSW | Easy | | | 17.15 |
| WSW | Hard | | | 67.00 |
| WSYue-TTS-eval | Base, with period | | | 7.06 |
| WSYue-TTS-eval | Coverage | | | 8.18 |

### Streaming and Inference Speed

<!-- Source: supplied development notes, p. 24, paired M5 / Nano performance table. -->

| Hardware | Mode | Nano RTF ↓ | M5 RTF ↓ | Nano first chunk ↓ | M5 first chunk ↓ |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Xeon Gold 6530, 1 CPU core | Streaming | **1.054** | 1.587 | **330 ms** | 541 ms |
| Xeon Gold 6530, 4 CPU cores | Streaming | **0.372** | 0.653 | **127 ms** | 225 ms |
| Xeon Gold 6530, 4 CPU cores | Non-streaming | **0.335** | 0.627 | | |
| RTX 4090 | Streaming | 0.206 | **0.098** | 45.8 ms | **20.1 ms** |

**Test conditions:** Each paired configuration uses the same 10 sentences, excluding warmup. First-chunk latency is measured after warmup; the M5 CPU run excludes approximately 6.2 seconds of preparation for five voices. M5 CPU inference uses FP32 TTS + FP32 codec. On GPU, Nano uses ONNX CUDA FP32, while M5 uses PyTorch BF16 + compilation. First-chunk audio durations differ: Nano / M5 are 80 / 220 ms on CPU and 80 / 60 ms on GPU. These measurements compare the listed deployment configurations. RTF is inference time divided by generated audio duration; values below 1 indicate faster-than-real-time generation.

## 🤗 Models and Resources

| Resource | Link |
| :--- | :--- |
| Hugging Face collection | [ASLP-lab / M5-TTS](https://huggingface.co/collections/ASLP-lab/m5-tts) |
| Model repository | [ASLP-lab/M5-TTS](https://huggingface.co/ASLP-lab/M5-TTS) |
| Technical report | |
| Audio demo | |
| Inference code | |

| Checkpoint file | LM parameters | Download |
| :--- | ---: | :--- |
| | 50M | |

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

## License

The [Hugging Face model card](https://huggingface.co/ASLP-lab/M5-TTS/blob/main/README.md) is tagged `Apache-2.0`.

| Component | License |
| :--- | :--- |
| Repository code | |

## Citation

<!-- Add the public technical report and its verified BibTeX entry here. -->
