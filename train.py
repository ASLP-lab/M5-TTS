"""M5-TTS training entry point.

Single GPU::

    accelerate launch --num_machines 1 --num_processes 1 train.py \
        --config m5/configs/train_example.json --exp_dir outputs/train_demo

Multi GPU::

    accelerate launch --num_machines 1 --num_processes 8 train.py \
        --config m5/configs/train_example.json --exp_dir outputs/train_demo

The tokenizer (and, optionally, the initial weights) are downloaded from
Hugging Face by default. Training data are JSONL shards that you prepare
yourself; see the Training section of the README for the record format.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

# ``accelerate launch`` executes this file directly; keep repository-root imports stable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from datasets import Features, IterableDataset as HFIterableDataset, Sequence, Value
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch.optim import AdamW
from torch.utils.data import DataLoader

from conditioning import (
    load_conditioning_tokenizer,
    load_conditioning_weights,
    validate_conditioning_bundle,
)
from m5.model.model import CSM, ModelArgs
from m5.training.data import BucketDynamicBatch, iter_jsonl, make_collate_fn
from m5.training.text import load_pinyin_vocab
from m5.training.trainer import Trainer

DEFAULT_TOKENIZER_MODEL = "ASLP-lab/M5-TTS-50M-VoiceClone"
AUDIO_VOCAB_SIZE = 2051  # 2048 codec codes + 1 offset for EOS


def log(message, **_print_kwargs):
    """Print only on the global main process to keep multi-GPU logs readable."""
    if int(os.environ.get("RANK", "0")) == 0:
        print(message, flush=True)


@dataclass
class TrainingConfig:
    # Directory containing one or more *.jsonl shards (searched recursively).
    data_path: str
    backbone_flavor: str = "llama-30M"
    decoder_flavor: str = "llama-20M"
    # Hugging Face repo ID or local bundle directory. Defaults to init_model,
    # or to DEFAULT_TOKENIZER_MODEL when neither is set.
    tokenizer: str | None = None
    # Optional Hugging Face repo ID or local bundle used for weights-only initialization.
    init_model: str | None = None
    allow_vocab_expansion: bool = False
    audio_num_codebooks: int = 16
    attention_type: str = "flash_attention_2"
    precision: str = "bf16"  # bf16, fp16 or no
    learning_rate: float = 1e-4
    min_lr: float = 1e-5
    total_steps: int = 100_000
    num_warmup_steps: int = 1_000
    scheduler_steps: int = 90_000
    save_every: int = 5_000
    log_every: int = 10
    max_tokens: int = 13_000
    batch_max_tokens: int = 6_000
    num_buckets: int = 100
    max_batch_size: int | None = None
    num_workers: int = 6
    shuffle_seed: int = 42
    decoder_fraction: float = 0.125
    lambda_depth: float = 1.0
    spk_drop_prob: float = 0.2
    pad_multiple: int = 16
    # Optional data augmentation: a user-provided pinyin allowlist, one syllable per line.
    pinyin_vocab_path: str | None = None
    pinyin_aug_ratio: float = 0.3
    # Full-state resume directory (model_*/ with optimizer state). Checkpoints
    # inside exp_dir are detected automatically.
    resume_from_checkpoint: str | None = None
    report_to: str = "none"  # none, wandb or tensorboard
    project_name: str = "m5-tts"
    seed: int | None = None


def resolve_source(source: str, patterns: list[str]) -> Path:
    """Resolve a local bundle directory or download a Hugging Face repository."""
    path = Path(source).expanduser()
    if path.is_dir():
        return path.resolve()
    if path.exists() or path.is_absolute() or str(source).startswith((".", "~")):
        raise FileNotFoundError(f"Expected a local directory or a Hugging Face repo ID: {source}")
    return Path(snapshot_download(repo_id=str(source), allow_patterns=patterns))


def load_tokenizer(config: TrainingConfig):
    source = config.tokenizer or config.init_model or DEFAULT_TOKENIZER_MODEL
    bundle = resolve_source(source, ["tokenizer/*", "conditioning.json"])
    tokenizer_dir = bundle / "tokenizer"
    if not (tokenizer_dir / "tokenizer_config.json").is_file():
        raise FileNotFoundError(f"Missing tokenizer/tokenizer_config.json in {bundle}")
    tokenizer = load_conditioning_tokenizer(tokenizer_dir)
    if (bundle / "conditioning.json").is_file():
        validate_conditioning_bundle(bundle, tokenizer)
    log(f"Loaded conditioning tokenizer ({len(tokenizer)} tokens) from {bundle}", flush=True)
    return tokenizer


def create_model(config: TrainingConfig, vocab_size: int) -> CSM:
    args = ModelArgs(
        backbone_flavor=config.backbone_flavor,
        decoder_flavor=config.decoder_flavor,
        text_vocab_size=vocab_size,
        audio_vocab_size=AUDIO_VOCAB_SIZE,
        audio_num_codebooks=config.audio_num_codebooks,
        use_codebooks=config.audio_num_codebooks,
        pad_multiple=config.pad_multiple,
        attention_type=config.attention_type,
        spk_drop_prob=config.spk_drop_prob,
        lambda_depth=config.lambda_depth,
    )
    return CSM(args)


def load_initial_weights(model: CSM, config: TrainingConfig, tokenizer):
    bundle = resolve_source(config.init_model, ["model.safetensors", "config.json", "conditioning.json", "tokenizer/*"])
    weights_path = bundle / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing model.safetensors in {bundle}")
    config_path = bundle / "config.json"
    if config_path.is_file():
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        for key in ("backbone_flavor", "decoder_flavor"):
            if key in saved and saved[key] != getattr(config, key):
                raise ValueError(
                    f"init_model uses {key}={saved[key]!r}, but the training config uses "
                    f"{getattr(config, key)!r}. Match the architecture or set init_model=null."
                )
    weights = load_file(str(weights_path), device="cpu")
    result = load_conditioning_weights(
        model, weights, tokenizer,
        allow_vocab_expansion=config.allow_vocab_expansion,
        checkpoint_path=bundle,
    )
    log(f"Initialized weights from {weights_path} (vocab expansion: {result['expanded']})", flush=True)


def build_dataloader(config: TrainingConfig, text_tokenizer, pinyin_vocab) -> DataLoader:
    data_path = Path(config.data_path).expanduser()
    if not data_path.is_dir():
        raise FileNotFoundError(f"data_path is not a directory: {data_path}")
    shards = sorted(data_path.rglob("*.jsonl"))
    random.Random(config.shuffle_seed).shuffle(shards)
    if not shards:
        raise FileNotFoundError(f"No .jsonl shards found under {data_path}")
    log(f"Using {len(shards)} JSONL shards from {data_path}", flush=True)

    # Extra columns are declared because Hugging Face datasets rejects records
    # containing undeclared fields; they are ignored by the collate function.
    features = Features({
        "utt": Value("string"),
        "text": Value("string"),
        "caption": Value("string"),
        "accent": Value("string"),
        "lang": Value("string"),
        "speaker": Value("string"),
        "token": Sequence(Sequence(Value("int64"))),
        "wav": Value("string"),
        "duration": Value("float64"),
        "start_time": Value("float64"),
        "end_time": Value("float64"),
    })
    dataset = HFIterableDataset.from_generator(
        iter_jsonl, gen_kwargs={"shards": [path.as_posix() for path in shards]}, features=features,
    ).shuffle(seed=config.shuffle_seed)

    batched = BucketDynamicBatch(
        dataset,
        max_tokens_per_bucket=config.max_tokens,
        max_tokens=config.batch_max_tokens,
        num_buckets=config.num_buckets,
        drop_oversized_samples=True,
        text_tokenizer=text_tokenizer,
        max_batch_size=config.max_batch_size,
    )
    collate_fn = make_collate_fn(
        text_tokenizer, config.audio_num_codebooks, pinyin_vocab, config.pinyin_aug_ratio
    )
    return DataLoader(
        batched, collate_fn=collate_fn, num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(), prefetch_factor=None, persistent_workers=False,
    )


def _checkpoint_step(name: str) -> int:
    import re
    numbers = re.findall(r"\d+", name)
    return int(numbers[-1]) if numbers else -1


def find_resume_dir(exp_dir: Path) -> Path | None:
    """Find the newest complete checkpoint inside the experiment directory."""
    candidates = sorted(
        (path for path in exp_dir.glob("model_*")
         if (path / "model.safetensors").is_file() and (path / "optimizer.bin").is_file()),
        key=lambda path: _checkpoint_step(path.name), reverse=True,
    )
    return candidates[0] if candidates else None


def validate_config(config: TrainingConfig):
    if config.lambda_depth < 0:
        raise ValueError("lambda_depth must be non-negative")
    if config.audio_num_codebooks != 16:
        raise ValueError("The Qwen 12 Hz codec requires audio_num_codebooks=16")
    if config.attention_type not in ("flash_attention_2", "sdpa"):
        raise ValueError("attention_type must be flash_attention_2 or sdpa")
    if config.precision not in ("bf16", "fp16", "no"):
        raise ValueError("precision must be bf16, fp16 or no")
    if config.report_to not in ("none", "wandb", "tensorboard"):
        raise ValueError("report_to must be none, wandb or tensorboard")
    if config.scheduler_steps <= config.num_warmup_steps:
        raise ValueError("scheduler_steps must be greater than num_warmup_steps")
    if config.total_steps < 1 or config.save_every < 1 or config.log_every < 1:
        raise ValueError("total_steps, save_every and log_every must be positive")


def parse_args():
    parser = argparse.ArgumentParser(description="Train M5-TTS with Accelerate")
    parser.add_argument("--config", required=True, help="Training configuration JSON file")
    parser.add_argument("--exp_dir", required=True, help="Directory for checkpoints and logs")
    parser.add_argument("--data_path", default=None, help="Override config.data_path")
    parser.add_argument("--tokenizer", default=None, help="Hugging Face repo ID or local tokenizer bundle")
    parser.add_argument("--init_model", default=None, help="Hugging Face repo ID or local bundle for weights-only initialization")
    parser.add_argument("--resume", default=None, help="Checkpoint directory for full-state resume")
    parser.add_argument("--project_name", default=None, help="Tracker run name / project name")
    parser.add_argument("--report_to", default=None, choices=("none", "wandb", "tensorboard"))
    parser.add_argument("--total_steps", type=int, default=None, help="Override config.total_steps")
    parser.add_argument("--save_every", type=int, default=None, help="Override config.save_every")
    parser.add_argument("--precision", default=None, choices=("bf16", "fp16", "no"))
    parser.add_argument("--attention_type", default=None, choices=("flash_attention_2", "sdpa"))
    parser.add_argument("--num_workers", type=int, default=None, help="Override config.num_workers")
    parser.add_argument("--seed", type=int, default=None, help="Global random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser()
    with config_path.open(encoding="utf-8") as stream:
        config_data = json.load(stream)
    unknown = set(config_data) - set(TrainingConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown configuration fields in {config_path}: {sorted(unknown)}")
    config = TrainingConfig(**config_data)

    overrides = {
        "data_path": args.data_path, "tokenizer": args.tokenizer, "init_model": args.init_model,
        "resume_from_checkpoint": args.resume, "project_name": args.project_name,
        "report_to": args.report_to, "total_steps": args.total_steps, "save_every": args.save_every,
        "precision": args.precision, "attention_type": args.attention_type,
        "num_workers": args.num_workers, "seed": args.seed,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)
    validate_config(config)

    if config.seed is not None:
        from accelerate.utils import set_seed
        set_seed(config.seed)
        log(f"Global seed set to {config.seed}", flush=True)

    exp_dir = Path(args.exp_dir).expanduser()
    exp_dir.mkdir(parents=True, exist_ok=True)
    resume_dir = find_resume_dir(exp_dir)
    if resume_dir:
        config.resume_from_checkpoint = str(resume_dir)
        log(f"Auto resume from {resume_dir}", flush=True)
    elif config.resume_from_checkpoint:
        resume_dir = Path(config.resume_from_checkpoint).expanduser()
        if not resume_dir.is_dir():
            raise FileNotFoundError(f"Resume checkpoint directory not found: {resume_dir}")

    tokenizer = load_tokenizer(config)
    pinyin_vocab = load_pinyin_vocab(config.pinyin_vocab_path) if config.pinyin_vocab_path else None
    model = create_model(config, len(tokenizer))

    # Full-state resume loads the same weights that accelerator.load_state restores;
    # weights-only initialization is used when no training state is being resumed.
    if config.init_model and resume_dir is None:
        load_initial_weights(model, config, tokenizer)
    elif config.init_model:
        log("init_model ignored because training state is being resumed", flush=True)

    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    decay = [p for _, p in parameters.items() if p.dim() >= 2]
    no_decay = [p for _, p in parameters.items() if p.dim() < 2]
    log(f"Decayed tensors: {len(decay)}, non-decayed tensors: {len(no_decay)}", flush=True)
    optimizer = AdamW(
        [
            {"params": decay, "weight_decay": 1e-2},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate, betas=(0.9, 0.95),
    )

    train_loader = build_dataloader(config, tokenizer, pinyin_vocab)
    model_config = {
        "backbone_flavor": config.backbone_flavor,
        "decoder_flavor": config.decoder_flavor,
        "text_vocab_size": len(tokenizer),
        "audio_vocab_size": AUDIO_VOCAB_SIZE,
        "audio_num_codebooks": config.audio_num_codebooks,
        "use_codebooks": config.audio_num_codebooks,
        "attention_type": config.attention_type,
    }
    trainer = Trainer(
        model=model, optimizer=optimizer, train_loader=train_loader,
        num_warmup_steps=config.num_warmup_steps, scheduler_steps=config.scheduler_steps,
        min_lr=config.min_lr, decoder_fraction=config.decoder_fraction,
        checkpoint_dir=str(exp_dir), text_tokenizer=tokenizer, model_config=model_config,
        accelerate_kwargs={"mixed_precision": config.precision},
        report_to=config.report_to, project_name=config.project_name,
    )
    trainer.train(
        total_steps=config.total_steps, save_every=config.save_every, log_every=config.log_every,
        resume_from_checkpoint=str(resume_dir) if resume_dir else None,
    )


if __name__ == "__main__":
    main()
