# M5-TTS GRPO 训练（基于 verl）

[English](README.md) | 简体中文

基于 verl 0.9.1 与 FSDP2 的单机同步 GRPO 训练。每步在八张 GPU 上对 32 条提示各采样 8 个结果，并在每步开始前从 learner 刷新 rollout 策略。

`verl_trainer/` 包含训练入口与 worker 编排，`m5_tts/` 包含 M5 模型、音频 rollout 与奖励计算。

| 信号 | 权重 | 作用位置 |
| --- | ---: | --- |
| 局部 CER（`char_exp`） | 0.5 | 音频帧决策，包括计入奖励时的采样 EOS |
| 句级说话人 SIM（WavLM/ECAPA） | 1.0 | 组内标准化后的结果优势 |
| 停顿 Pause | 0.5 | 组内标准化后的结果优势 |

SIM 与 Pause 先在同一提示的候选组内分别标准化，加权求和后再次标准化，再广播到采样决策。`char_exp` 则在同一组 8 个候选上标准化字符编辑奖励，经 ASR 对齐投影后由 learner 叠加。

## 环境安装

本子模块维护自己的环境列表 [`requirements.txt`](requirements.txt)，必须使用独立的 Python 3.11 虚拟环境。它依赖 PyTorch 2.9.1/CUDA 12.8，与仓库根目录 `requirements.txt`（PyTorch 2.6.0/CUDA 12.4）不兼容，请勿把任一清单安装到对方环境中。

需要 Linux、八张支持 BF16 的 CUDA GPU 以及至少 2 GiB 的共享内存供 Ray 使用；TorchCodec 还需要 FFmpeg 动态库。安装命令从仓库根目录执行，训练命令在本 `grpo/` 目录下执行。

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

verl 的 wheel 声明其完整运行时依赖 Transformers 5.x；本配方只使用其 controller、engine 和算法接口，并搭配 M5 模型所用的 Transformers 4.57.6。`qwen-tts` 用 `--no-deps` 单独安装，因为它锁定 Transformers 4.57.3，而 `qwen-asr` 要求 4.57.6；其运行时依赖已列在 `requirements.txt` 中。

## 输入

填写 [`config.yaml`](config.yaml) 中的路径，相对路径以配置文件所在目录为基准。请从 Hugging Face 下载所需资产到 `assets/`，权重和数据不包含在本仓库中。

- M5 的 `model.safetensors`，以及同目录或上一级目录下配套的 `tokenizer/` 与 `conditioning.json`，需使用 `m5-caption-accent-speaker-lang-v3` 格式和 16 个音频码本。
- Qwen3-TTS 12 Hz 的 `speech_tokenizer/`、Qwen3-ASR-0.6B、Qwen3-ForcedAligner-0.6B、WavLM Large 及配套的 ECAPA 说话人验证权重；SIM 权重的模型参数需位于 `model` 字段下。
- 中文提示 JSONL，每行包含 `text` 和 `lang`，`lang` 设为 `Chinese`；两条交替分片都需要提示。
- 参考音频 JSON 数组，条目数为正偶数且说话人互不重复；每条需包含 `reference_id`、`speaker_id`、`wav_path` 和 `prompt_text`。相对波形路径以参考 JSON 文件为基准解析。

```json
{"utt":"example-1","text":"今天的天气很好。","lang":"Chinese"}
```

```json
{"reference_id":"ref-1","speaker_id":"speaker-1","wav_path":"audio/ref-1.wav","prompt_text":"现在开始录音。","lang":"Chinese"}
```

## 训练

```bash
python -m verl_trainer.colocated_trainer --config config.yaml
```

`--max-steps 1` 可只做一次参数更新检查。运行目录包含 TensorBoard 日志、`metrics.jsonl` 和 `state.json`；检查点在第 1 步、每个保存间隔和最终步写出。用 `--resume PATH_TO_STEP_DIRECTORY` 并配合原始的 M5 检查点和数据续训；在运行目录创建 `STOP` 文件可在下一步边界保存并退出。

第三方代码与模型条款见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0)。请只加载可信来源的检查点和奖励模型权重。
