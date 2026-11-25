"""Accelerate-based training loop for M5-TTS."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from conditioning import save_conditioning_bundle, validate_conditioning_bundle


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        *,
        num_warmup_steps: int,
        scheduler_steps: int,
        min_lr: float,
        decoder_fraction: float,
        checkpoint_dir: str,
        text_tokenizer,
        model_config: dict[str, Any],
        accelerate_kwargs: dict | None = None,
        report_to: str = "none",
        project_name: str = "m5-tts",
        max_grad_norm: float = 1.0,
    ):
        if report_to not in ("none", "wandb", "tensorboard"):
            raise ValueError("report_to must be none, wandb or tensorboard")
        self.report_to = report_to
        self.project_name = project_name
        self.max_grad_norm = max_grad_norm
        self.decoder_fraction = decoder_fraction
        self.num_warmup_steps = num_warmup_steps
        self.text_tokenizer = text_tokenizer
        self.model_config = model_config

        kwargs = dict(accelerate_kwargs or {})
        if report_to != "none":
            kwargs["log_with"] = report_to
        self.accelerator = Accelerator(
            dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
            **kwargs,
        )
        self.print = self.accelerator.print

        self.optimizer = optimizer
        self.create_scheduler(scheduler_steps, num_warmup_steps, min_lr)
        self.model, self.optimizer, self.scheduler, self.train_loader = self.accelerator.prepare(
            model, self.optimizer, self.scheduler, train_loader
        )

        self.checkpoint_dir = Path(checkpoint_dir).expanduser()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if (self.checkpoint_dir / "conditioning.json").is_file():
            validate_conditioning_bundle(self.checkpoint_dir, text_tokenizer)
        elif any(self.checkpoint_dir.glob("model_*")):
            raise ValueError(
                "Existing checkpoints have no conditioning bundle. Use a new exp_dir and "
                "resume_from_checkpoint=model.safetensors for weights-only continuation."
            )
        elif self.accelerator.is_main_process:
            save_conditioning_bundle(self.checkpoint_dir, text_tokenizer)
        self.accelerator.wait_for_everyone()

        def num_params(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        self.print(f"Backbone parameter count: {num_params(model.backbone):,}")
        self.print(f"Decoder parameter count: {num_params(model.decoder):,}")
        self.print(f"Text embeddings parameter count: {num_params(model.text_embeddings):,}")
        self.print(f"Audio embeddings parameter count: {num_params(model.audio_embeddings):,}")
        self.print(f"Codebook 0 head parameter count: {num_params(model.codebook0_head):,}")
        self.print(f"*Total parameter count*: {num_params(model):,}")

    def create_scheduler(self, scheduler_steps: int, warmup_steps: int, min_lr: float):
        """Linear warmup, cosine decay to min_lr, then a constant tail."""
        if scheduler_steps <= warmup_steps:
            raise ValueError("scheduler_steps must be greater than num_warmup_steps")
        processes = self.accelerator.num_processes
        cosine_steps = (scheduler_steps - warmup_steps) * processes
        warmup_steps_actual = warmup_steps * processes
        constant_ratio = min_lr / self.optimizer.param_groups[0]["lr"]
        warmup = LinearLR(self.optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_steps_actual)
        cosine = CosineAnnealingLR(self.optimizer, T_max=cosine_steps, eta_min=min_lr)
        constant = LambdaLR(self.optimizer, lr_lambda=lambda _: constant_ratio)
        constant.total_iters = 100_000_000_000
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine, constant],
            milestones=[warmup_steps_actual, warmup_steps_actual + cosine_steps],
        )

    def safe_backward_and_step(self, loss, global_step: int) -> bool:
        try:
            self.accelerator.backward(loss)
            if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scheduler.step()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return True
        except RuntimeError as error:
            if "timeout" in str(error).lower() or "nccl" in str(error).lower():
                self.print(f"NCCL timeout at step {global_step}, skipping: {error}")
                self.optimizer.zero_grad()
                return False
            raise

    def load_checkpoint(self, checkpoint_path: str):
        path = Path(checkpoint_path).expanduser()
        if not path.is_dir():
            self.print(f"Checkpoint not found: {path}")
            return None, None
        self.accelerator.load_state(str(path))
        self.print(f"Loaded training state from {path}")
        parts = path.name.split("_")
        if len(parts) >= 3 and parts[0] == "model":
            try:
                return int(parts[1]), int(parts[2])
            except ValueError:
                pass
        return None, None

    def save_checkpoint(self, name: str):
        """Save training state plus a self-contained inference bundle."""
        path = self.checkpoint_dir / name
        self.accelerator.save_state(str(path))
        if self.accelerator.is_main_process:
            (path / "config.json").write_text(
                json.dumps(self.model_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            save_conditioning_bundle(path, self.text_tokenizer)
        self.accelerator.wait_for_everyone()
        self.print(f"Saved checkpoint: {path}")

    def train(
        self,
        total_steps: int,
        save_every: int,
        log_every: int = 10,
        resume_from_checkpoint: str | None = None,
    ):
        if total_steps < 1 or save_every < 1 or log_every < 1:
            raise ValueError("total_steps, save_every and log_every must be positive")
        global_step, epoch = 0, 1
        if resume_from_checkpoint:
            loaded_epoch, loaded_step = self.load_checkpoint(resume_from_checkpoint)
            epoch = loaded_epoch or epoch
            global_step = loaded_step or global_step

        hps = {
            "total_steps": total_steps, "num_warmup_steps": self.num_warmup_steps,
            "save_every": save_every, "log_every": log_every, **self.model_config,
        }
        if self.report_to != "none":
            self.accelerator.init_trackers(self.project_name, config=hps)

        total_tokens = 0
        consecutive_timeouts = 0
        max_consecutive_timeouts = 3
        last_saved_step = None
        empty_epochs = 0
        self.print(f"Start training from epoch {epoch}, step {global_step}, target {total_steps} steps")

        while global_step < total_steps:
            self.model.train()
            if hasattr(self.train_loader.batch_sampler, "set_epoch"):
                self.train_loader.batch_sampler.set_epoch(epoch)

            running_loss, num_batches = 0.0, 0
            progress = tqdm(
                self.train_loader, desc=f"Epoch {epoch}",
                disable=not self.accelerator.is_local_main_process,
            )
            for batch in progress:
                tokens, tokens_mask = batch["tokens"], batch["tokens_mask"]
                if tokens.ndim != 3 or tokens.numel() == 0:
                    continue
                if tokens.shape[-1] != self.model_config["audio_num_codebooks"] + 1:
                    raise ValueError(
                        f"Expected {self.model_config['audio_num_codebooks'] + 1} token columns, "
                        f"got {tokens.shape[-1]}"
                    )
                total_tokens += tokens.shape[0] * tokens.shape[1]

                with self.accelerator.autocast():
                    losses = self.model(tokens, tokens_mask, decoder_fraction=self.decoder_fraction)

                if not self.safe_backward_and_step(losses["loss"], global_step):
                    consecutive_timeouts += 1
                    if consecutive_timeouts >= max_consecutive_timeouts:
                        self.print("Max consecutive timeouts reached. Stopping training.")
                        break
                    continue
                consecutive_timeouts = 0
                running_loss += losses["loss"].item()
                num_batches += 1
                global_step += 1

                if global_step % log_every == 0 and self.accelerator.is_local_main_process:
                    self.accelerator.log({
                        "loss": losses["loss"].item(),
                        "backbone_loss": losses["backbone_loss"].item(),
                        "decoder_loss": losses["decoder_loss"].item(),
                        "c0_accuracy": losses["c0_accuracy"].item(),
                        "lr": self.scheduler.get_last_lr()[0],
                        "total_tokens": total_tokens,
                        "epoch": epoch,
                    }, step=global_step)
                    progress.set_postfix(
                        step=global_step, loss=f"{losses['loss'].item():.3f}",
                        c0_acc=f"{losses['c0_accuracy'].item():.3f}",
                        lr=f"{self.scheduler.get_last_lr()[0]:.6f}",
                    )

                if global_step % save_every == 0:
                    self.save_checkpoint(f"model_{epoch}_{global_step}")
                    last_saved_step = global_step
                if global_step >= total_steps:
                    break

            if num_batches:
                average = running_loss / num_batches
                self.print(f"Epoch {epoch} average loss: {average:.4f}")
                self.accelerator.log({"avg_epoch_loss": average}, step=global_step)
                empty_epochs = 0
            else:
                empty_epochs += 1
                if empty_epochs >= 3:
                    raise RuntimeError(
                        "No valid batches were produced. Check data_path and the JSONL record format."
                    )
            if consecutive_timeouts >= max_consecutive_timeouts:
                break
            epoch += 1

        # Avoid writing a second multi-GB state when the last step was just checkpointed.
        if last_saved_step != global_step:
            self.save_checkpoint(f"model_{epoch}_{global_step}_final")
        else:
            self.print(f"Final checkpoint already saved at step {global_step}")
        # Also destroys the distributed process group on multi-GPU runs.
        self.accelerator.end_training()
        self.print("Training complete.")
