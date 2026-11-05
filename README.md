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
