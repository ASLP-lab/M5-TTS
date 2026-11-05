<p align="center">
  <img src="m5-tts-logo.png" alt="M5-TTS hummingbird logo" width="440">
</p>

<h1 align="center">M5-TTS</h1>

<p align="center">
  <b>A Mini-Scale Multilingual and Multi-Dialect Text-to-Speech Language Model with Multi-Token Prediction</b>
</p>

<p align="center">
  <b>50M LM · 6 Languages · Chinese Dialects · Zero-Shot Voice Cloning · CPU Streaming</b>
</p>

<p align="center">
  English | <a href="README_zh.md">简体中文</a><br>
  <a href="https://aslp-lab.github.io/M5-TTS-Demo-Page/M5_TTS__A_Mini_Scale_Multilingual_and_Multi_Dialect_Text_to_Speech_Language_Model_with_Multi_Token_Prediction_v2.pdf">Technical Report</a> ·
  <a href="https://aslp-lab.github.io/M5-TTS-Demo-Page/">Audio Demos</a> ·
  <a href="https://huggingface.co/spaces/ASLP-lab/M5-TTS-Space">Try Online</a> ·
  <a href="https://huggingface.co/collections/ASLP-lab/m5-tts">Model Collection</a>
</p>

**M5-TTS** is a compact language-model-based speech generation system for multilingual and Chinese multidialect synthesis, zero-shot voice cloning, and dedicated multilingual voices. Its **50M-parameter LM excludes the speech codec** and combines a temporal backbone with a local multi-token prediction (MTP) module.

**Half the LM parameters of MOSS-TTS-Nano, with relative error reductions of 62.2% on Seed-ZH and 35.3% on Seed-EN.** M5-TTS (GRPO) achieves **1.37% CER / 2.91% WER**, compared with **3.62% / 4.50%** for Nano. Four-core CPU streaming achieves **RTF 0.653** and **225 ms warm first-chunk latency** in the reported Xeon configuration.

## ✨ Features

|  | Feature | What M5-TTS provides |
| :--- | :--- | :--- |
| **M₁** | **Mini** | 50M LM: a 30M temporal backbone and a 20M local decoder; codec parameters are separate. |
| **M₂** | **Multilingual** | Chinese, English, German, French, Japanese, and Korean. |
| **M₃** | **Multi-Dialect** | Cantonese (Yue) and Sichuanese (Chuan), with dialect-specific evaluations. |
| **M₄** | **Multi-Token Prediction** | 16 codebooks at 12.5 codec frames/s, with hierarchical temporal and within-frame generation. |
| **M₅** | **Language Modeling** | Reference-based zero-shot voice cloning, speaker-specific multilingual synthesis, and GRPO refinement. |

- **24 kHz speech output** with a causal speech-codec decoder.
- **CPU streaming** through inference graph optimization, reusable caches, stateful codec decoding, and adaptive chunk scheduling.
- **Global + local post-training feedback:** speaker similarity and pause rewards are combined with token-level CER process advantages.

## 🎧 Try M5-TTS

Open the [interactive Space](https://huggingface.co/spaces/ASLP-lab/M5-TTS-Space), upload reference speech and its accurate transcript, then enter the target text and select a language and optional accent (`yue` / `chuan`). The current demo accepts up to 250 characters of target text. Listen to multilingual and dialect comparisons on the [audio demo page](https://aslp-lab.github.io/M5-TTS-Demo-Page/).

The hosted Space runs on GPU and returns a completed audio clip. The CPU streaming measurements below describe the inference system evaluated in the report.

### Release status (2026-11-06)

| Resource | Status |
| :--- | :--- |
| [Technical report](https://aslp-lab.github.io/M5-TTS-Demo-Page/M5_TTS__A_Mini_Scale_Multilingual_and_Multi_Dialect_Text_to_Speech_Language_Model_with_Multi_Token_Prediction_v2.pdf) | Available. |
| [Audio demos](https://aslp-lab.github.io/M5-TTS-Demo-Page/) | Available. |
| [Online Space](https://huggingface.co/spaces/ASLP-lab/M5-TTS-Space) | Available, with [inference source and a checkpoint](https://huggingface.co/spaces/ASLP-lab/M5-TTS-Space/tree/main). |
| [Model collection](https://huggingface.co/collections/ASLP-lab/m5-tts) / [model repository](https://huggingface.co/ASLP-lab/M5-TTS) | Repository created; standalone model releases are planned. The model repository currently contains a model card, without weights. |
| Training, CPT, and GRPO code | Planned for release. |
| Packaged local / CPU streaming inference | Planned for release in this repository. |
| Standalone voice-cloning and speaker-specific checkpoints | Planned; the Space checkpoint is not labeled by training stage. |

## 📊 Evaluation

Results below follow the technical report. **The main Seed and MiniMax comparisons use the M5-TTS GRPO voice-cloning model.** CER/WER are percentages (lower is better); SIM is speaker similarity (higher is better). Bold highlights better results. Missing measurements are shown as `—` and do not imply that a capability is unsupported.

| Model | LM parameters (M) ↓ |
| :--- | :---: |
| MOSS-TTS-Nano | 100 |
| **M5-TTS** | **50** |

Parameter counts exclude speech codecs.

### Seed-TTS Eval

Benchmark: [Seed-TTS Eval](https://github.com/BytedanceSpeech/seed-tts-eval).

| Subset | Metric | MOSS-TTS-Nano | M5-TTS (GRPO) |
| :--- | :--- | ---: | ---: |
| Seed-ZH | CER (%) ↓ | 3.62 | **1.37** |
| Seed-ZH | SIM ↑ | 0.64 | **0.65** |
| Seed-EN | WER (%) ↓ | 4.50 | **2.91** |
| Seed-EN | SIM ↑ | 0.50 | **0.53** |
| Seed-Hard | CER (%) ↓ | **15.77** | 17.48 |
| Seed-Hard | SIM ↑ | **0.63** | 0.59 |

M5-TTS reduces Seed-ZH CER and Seed-EN WER by **62.2%** and **35.3%** relative to Nano. Nano remains stronger on **Seed-Hard** in both content accuracy and speaker similarity.

### MiniMax Multilingual

Benchmark: [MiniMax TTS-Multilingual-Test-Set](https://huggingface.co/datasets/MiniMaxAI/TTS-Multilingual-Test-Set).

| Language | Error metric (%) ↓ | Nano error | M5 error | Nano SIM ↑ | M5 SIM ↑ |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Chinese | CER | **1.80** | 2.05 | **0.73** | 0.71 |
| English | WER | 5.34 | **3.06** | **0.65** | 0.63 |
| German | WER | 11.18 | **3.81** | 0.54 | **0.63** |
| French | WER | 18.64 | **8.14** | 0.51 | **0.53** |
| Japanese | CER | 40.22 | **9.70** | 0.70 | **0.72** |
| Korean | CER | 8.21 | **2.88** | **0.67** | 0.66 |

M5-TTS has lower content error rates in **five of six languages**. Relative reductions for EN / DE / FR / JA / KO are **42.7% / 65.9% / 56.3% / 75.9% / 64.9%**. Nano retains lower Chinese CER; speaker-similarity results vary by language.

<details>
<summary>CPT, GRPO, and dedicated-speaker results</summary>

**Seed-TTS Eval**

| M5-TTS variant | Setting | ZH CER (%) | ZH SIM | EN WER (%) | EN SIM | Hard CER (%) | Hard SIM |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPT | Voice Cloning | 1.70 | 0.64 | 3.60 | 0.53 | 20.92 | 0.57 |
| GRPO | Voice Cloning | 1.37 | 0.65 | 2.91 | 0.53 | 17.48 | 0.59 |
| Speaker-specific | Custom Voice | 1.26 | — | 1.88 | — | 18.19 | — |

**MiniMax**

Each cell is `CER/WER (%) / SIM`; ZH / JA / KO use CER and EN / DE / FR use WER.

| M5-TTS variant | ZH | EN | DE | FR | JA | KO |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPT | 2.54 / 0.70 | 3.67 / 0.62 | 4.60 / 0.63 | 8.15 / 0.51 | 12.71 / 0.72 | 3.10 / 0.66 |
| GRPO | 2.05 / 0.71 | 3.06 / 0.63 | 3.81 / 0.63 | 8.14 / 0.53 | 9.70 / 0.72 | 2.88 / 0.66 |
| Speaker-specific | 1.16 / — | 2.24 / — | 0.70 / — | 5.33 / — | 10.27 / — | 2.09 / — |

The dedicated-speaker model is evaluated as **Custom Voice**, separately from zero-shot cloning. Its scores are not a matched voice-cloning comparison. Full baseline settings and selected preset voices are documented in the report.

</details>

### Chinese dialects

| Dialect | Model | Easy CER (%) ↓ | Easy SIM ↑ | Hard CER (%) ↓ | Hard SIM ↑ |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Yue | **M5-TTS (GRPO)** | **8.55** | 0.47 | **8.70** | 0.51 |
| Yue | CosyVoice2-Yue | 10.33 | **0.62** | 9.49 | **0.67** |
| Yue | Llasa-1B-Yue | 10.89 | 0.51 | 12.78 | 0.55 |
| Chuan | **M5-TTS (GRPO)** | 5.08 | 0.58 | **7.78** | 0.59 |
| Chuan | CosyVoice2-WSC | **4.28** | **0.64** | 8.78 | **0.65** |

Yue is Cantonese and Chuan is Sichuanese. Systems use **U2pp-Conformer-Yue** for Cantonese and **FireRedASR-AED** for Sichuanese. M5-TTS has the lowest reported Cantonese CER and improves Sichuanese Hard CER, while dialectal SIM remains below the compared baselines.

### Subjective evaluation

**10 listeners**, scores from **1 to 5**. N-MOS: naturalness; S-MOS: speaker similarity; I-MOS: intelligibility; P-MOS: prosody and expressiveness.

| Model | N-MOS ↑ | S-MOS ↑ | I-MOS ↑ | P-MOS ↑ |
| :--- | ---: | ---: | ---: | ---: |
| Kokoro-TTS | 2.85 | — | **3.10** | 2.60 |
| MOSS-TTS-Nano | **3.00** | 2.85 | 2.65 | 2.80 |
| **M5-TTS** | 2.90 | **3.05** | 2.95 | **2.85** |

M5-TTS receives higher similarity and intelligibility ratings than Nano, with a lower naturalness rating. These are descriptive score differences; the report does not establish statistical significance.

### CPU streaming

| Hardware | CPU cores | RTF ↓ | Warm first-chunk latency ↓ |
| :--- | :---: | ---: | ---: |
| Intel Xeon Gold 6530 | 4 | 0.653 | 225 ms |

RTF is synthesis time divided by generated audio duration; **RTF < 1** indicates faster-than-real-time generation. Timing excludes model loading and warmup. These measurements apply to the tested server CPU and do not establish mobile or ARM performance.

## 🧩 How it works

### Hierarchical speech generation

<p align="center">
  <img src="https://aslp-lab.github.io/M5-TTS-Demo-Page/assets/architecture.png" alt="M5-TTS hierarchical backbone and MTP architecture" width="900">
</p>

The temporal backbone predicts **codebook 0** at each frame. The local MTP Transformer then autoregressively generates **codebooks 1–15**. Historical frame embeddings are summed before temporal modeling, and completed frames are decoded incrementally by [Qwen3-TTS's speech tokenizer](https://github.com/QwenLM/Qwen3-TTS).

### Training and post-training

1. **Basic pre-training:** approximately **400,000 hours** of multilingual and regional speech, including Emilia, Emilia-YODAS, in-house recordings, and WenetSpeech datasets.
2. **Continued pre-training:** a curated **65,000-hour subset** of the pre-training corpus, emphasizing recording quality, expressiveness, and challenging pronunciation. This is a subset, not additional independent training data.
3. **GRPO:** utterance-level **SIM + pause** feedback provides a global advantage. ASR edit alignment produces per-character group-relative feedback; negative advantages propagate backward within a clause with exponential decay, then map to codec frames. The local CER process advantage is **added** to the global advantage, and all codebooks in a frame share the combined signal.
4. **Speaker-specific fine-tuning:** curated recordings of a designated speaker support a dedicated multilingual voice.

The global reward weights are SIM **1.0** and pause **0.5**; the local CER advantage weight is **0.5**, with KL coefficient **0.01**. Sentence-level CER broadcasting is disabled. Pre-training conditions on `accent → language → text`; speaker-specific fine-tuning adds the speaker field: `accent → speaker → language → text`.

### Efficient inference

The report describes a local-decoder frame graph, fused projections, KV and reference caches, stateful codec decoding, and adaptive **4 / 8 / 16-frame** CPU scheduling. GPU inference uses BF16 for the TTS LM, FP32 for the codec, and `torch.compile`; the quantitative runtime result above is the four-core CPU measurement.

## Limitations and roadmap

Current limitations include the bandwidth cost of 16-codebook local autoregressive decoding, 24 kHz output, artifacts in highly expressive speech, and weaker performance on difficult Chinese inputs and dialectal speaker preservation.

Planned directions include **8-codebook inference**, **48 kHz output**, better challenging-sample data selection and mixture ratios, and explicit control over **emotion, speaking style, and paralinguistic expressions** such as laughter and sighs.

## License

The [Hugging Face model card](https://huggingface.co/ASLP-lab/M5-TTS/blob/main/README.md) is tagged `Apache-2.0`. Separate licenses for the GitHub code and Space artifacts have not yet been published.

## Citation

If you use this work, please cite the technical report:

```bibtex
@misc{m5tts2026,
  title = {{M5-TTS: A Mini-Scale Multilingual and Multi-Dialect Text-to-Speech Language Model with Multi-Token Prediction}},
  author = {Wenhao Li and Jingbin Hu and Shengming Ling and Xiaming Ren and Haoyu Zhang and Jinrui Liang and Hanke Xie and Qirui Zhan and Guobin Ma and Huakang Chen and Chunbo Hao and Jianguo Zheng and Wenkang Hou and Dake Guo and Yuepeng Jiang and Linhan Ma and Yunxiang Chen and Houdun Liu and Su Feng and Chuang Ding and Pengcheng Zhu and Liumeng Xue and Lei Xie},
  year = {2026},
  howpublished = {Technical report},
  url = {https://github.com/ASLP-lab/M5-TTS}
}
```
