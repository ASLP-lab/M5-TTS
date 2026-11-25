# M5 TTS GRPO with verl

English | [简体中文](README_zh.md)

Single-node synchronous GRPO for M5 TTS using verl 0.9.1 and FSDP2. Each step runs 32 prompts with eight completions per prompt across eight GPUs. The rollout policy is refreshed from the learner before each step.

`verl_trainer/` contains the training entry point and worker orchestration. `m5_tts/` contains the M5 model, audio rollout, and reward computation.

| Signal | Weight | Applied to |
| --- | ---: | --- |
| Local CER (`char_exp`) | 0.5 | Audio frame decisions, including sampled EOS when credited |
| Utterance speaker SIM (WavLM/ECAPA) | 1.0 | Group-normalized outcome advantage |
| Pause | 0.5 | Group-normalized outcome advantage |

SIM and pause are standardized within each prompt group. Their weighted sum is standardized again and broadcast to the sampled decisions. The learner adds `char_exp` credit after standardizing character edit rewards across the same eight completions and projecting them through the ASR alignment.

## Setup

This subsystem maintains its own environment list, [`requirements.txt`](requirements.txt), and must use a separate Python 3.11 virtual environment. Its PyTorch 2.9.1/CUDA 12.8 stack is not compatible with the repository-root `requirements.txt` (PyTorch 2.6.0/CUDA 12.4); do not install either list into the other environment.

Use Linux, eight BF16-capable CUDA GPUs, and at least 2 GiB of shared memory for Ray. TorchCodec also requires FFmpeg shared libraries. Run the setup from the repository root; training commands run inside this `grpo/` directory.

```bash
cd grpo
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install --no-deps qwen-tts==0.0.3
python -m pip install --no-deps verl==0.9.1
git clone https://github.com/s3prl/s3prl.git assets/s3prl
git -C assets/s3prl checkout ec8064b5889f81ca460fbe2c094ce576a6f120b7
```

The verl wheel declares Transformers 5.x for its full runtime. This recipe uses its controller, engine and algorithm interfaces with the Transformers 4.57.6 build used by the M5 model. `qwen-tts` is installed with `--no-deps` because it pins Transformers 4.57.3 while `qwen-asr` requires 4.57.6; its runtime dependencies are listed in `requirements.txt`.

## Inputs

Fill in the paths in [`config.yaml`](config.yaml). Relative paths are resolved from the config file. Download the assets from Hugging Face into `assets/`; weights and data are not included in this repository.

- An M5 `model.safetensors` export, with its matching `tokenizer/` and `conditioning.json` beside it or one directory above it. The bundle uses the `m5-caption-accent-speaker-lang-v3` format and 16 audio codebooks.
- The Qwen3-TTS 12 Hz `speech_tokenizer/`, Qwen3-ASR-0.6B, Qwen3-ForcedAligner-0.6B, WavLM Large and its matching ECAPA verification checkpoint. The SIM checkpoint must expose model weights under `model`.
- A Mandarin prompt JSONL with `text` and `lang` on each line. Set `lang` to `Chinese`. Both alternating line shards need a prompt.
- A reference JSON array with a positive even number of entries and unique speakers. Each entry needs `reference_id`, `speaker_id`, `wav_path`, and `prompt_text`. Relative waveform paths are resolved from the reference JSON file.

```json
{"utt":"example-1","text":"今天的天气很好。","lang":"Chinese"}
```

```json
{"reference_id":"ref-1","speaker_id":"speaker-1","wav_path":"audio/ref-1.wav","prompt_text":"现在开始录音。","lang":"Chinese"}
```

## Train

```bash
python -m verl_trainer.colocated_trainer --config config.yaml
```

Use `--max-steps 1` for a one-update check. The run directory contains TensorBoard logs, `metrics.jsonl` and `state.json`. Checkpoints are written at step 1, each save interval and the final step. Resume with `--resume PATH_TO_STEP_DIRECTORY` and the original M5 checkpoint and data. A `STOP` file in the run directory requests a checkpoint and exit at the next step boundary.

Third-party code and model terms are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0). Only load checkpoints and reward weights from sources you trust.
