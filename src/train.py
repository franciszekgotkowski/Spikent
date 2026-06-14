#!/usr/bin/env python3
"""Training script for a small Polish -> English Transformer."""

import argparse
import os
import random
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import MarianTokenizer

from config import ModelConfig, SmallModelConfig, TrainConfig
from src.dataset import (
    TranslationIterableDataset,
    build_dataloader,
    load_translation_dataset,
)
from src.model import TransformerTranslator


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_linear_warmup_cosine_decay_scheduler(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step + 1 - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265))).item()

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    accelerator: Accelerator,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    tokenizer,
    step: int,
    output_dir: str,
    is_best: bool = False,
):
    save_dir = Path(output_dir) / f"step_{step}"
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    state = {
        "step": step,
        "model_state_dict": unwrapped_model.state_dict(),
    }
    if accelerator.is_main_process:
        state["optimizer_state_dict"] = optimizer.state_dict()
        state["scheduler_state_dict"] = scheduler.state_dict()

    accelerator.save(state, save_dir / "checkpoint.pt")
    if accelerator.is_main_process:
        tokenizer.save_pretrained(save_dir)
        (save_dir / "DONE").touch()

        if is_best:
            best_dir = Path(output_dir) / "best"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(save_dir, best_dir)


def train(cfg: TrainConfig, model_cfg: ModelConfig):
    set_seed(cfg.seed)
    accelerator = Accelerator(
        mixed_precision="fp16" if cfg.use_amp else "no",
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    )

    tokenizer = MarianTokenizer.from_pretrained(cfg.tokenizer_name)

    model_cfg.src_vocab_size = len(tokenizer)
    model_cfg.tgt_vocab_size = len(tokenizer)
    model_cfg.max_seq_len = cfg.max_seq_len

    if accelerator.is_main_process:
        print(f"Tokenizer vocabulary size: {len(tokenizer)}")

    model = TransformerTranslator(model_cfg)
    if accelerator.is_main_process:
        print(f"Model parameters: {count_parameters(model):,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    dataset = load_translation_dataset(
        cfg.dataset_name,
        cfg.dataset_subset,
        split="train",
        streaming=True,
    )

    train_dataset = TranslationIterableDataset(
        dataset,
        tokenizer,
        cfg.src_column,
        cfg.tgt_column,
        cfg.max_seq_len,
        max_samples=cfg.max_samples,
        lang_score_column=cfg.lang_score_column,
    )
    train_loader = build_dataloader(
        train_dataset,
        tokenizer,
        cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    # Estimate total steps from max_samples / effective batch size.
    effective_batch_size = cfg.batch_size * cfg.gradient_accumulation_steps
    total_samples = cfg.max_samples or 1_000_000
    total_steps = (total_samples // effective_batch_size) * cfg.num_epochs

    scheduler = get_linear_warmup_cosine_decay_scheduler(
        optimizer, cfg.warmup_steps, total_steps
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=-100,
        label_smoothing=cfg.label_smoothing,
    )

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    global_step = 0

    for epoch in range(1, cfg.num_epochs + 1):
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch}/{cfg.num_epochs}")

        progress = tqdm(
            disable=not accelerator.is_main_process,
            desc=f"Epoch {epoch}",
            unit="step",
        )

        model.train()
        for batch in train_loader:
            with accelerator.accumulate(model):
                src = batch["input_ids"]
                tgt = batch["labels"]

                # Decoder input: labels shifted right; prepend BOS.
                # Labels use -100 for ignored positions, so we replace those with
                # the real pad token id before feeding them to the model.
                decoder_input = torch.cat(
                    [
                        torch.full(
                            (tgt.size(0), 1),
                            tokenizer.pad_token_id,
                            dtype=torch.long,
                            device=tgt.device,
                        ),
                        tgt[:, :-1].masked_fill(
                            tgt[:, :-1] == -100, tokenizer.pad_token_id
                        ),
                    ],
                    dim=1,
                )

                # The model expects key-padding masks where True means "ignore".
                src_key_padding_mask = src == tokenizer.pad_token_id
                tgt_key_padding_mask = decoder_input == tokenizer.pad_token_id

                logits = model(
                    src=src,
                    tgt=decoder_input,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                )

                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt.reshape(-1),
                )

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            progress.update(1)
            progress.set_postfix({"loss": loss.item()})

            if global_step % cfg.log_every_n_steps == 0 and accelerator.is_main_process:
                lr = scheduler.get_last_lr()[0]
                print(f"  step {global_step} | loss: {loss.item():.4f} | lr: {lr:.2e}")

            if global_step % cfg.save_every_n_steps == 0:
                save_checkpoint(
                    accelerator,
                    model,
                    optimizer,
                    scheduler,
                    tokenizer,
                    global_step,
                    cfg.output_dir,
                )

        progress.close()

    # Final checkpoint
    save_checkpoint(
        accelerator,
        model,
        optimizer,
        scheduler,
        tokenizer,
        global_step,
        cfg.output_dir,
        is_best=True,
    )
    if accelerator.is_main_process:
        print(f"Training complete. Checkpoints saved to {cfg.output_dir}")

    # PyArrow streaming threads outlive Python shutdown and trigger a GIL
    # crash if we let the interpreter finalize normally. os._exit() terminates
    # the process immediately after all checkpoints are flushed, bypassing
    # Python finalizers and avoiding the crash.
    os._exit(0)


def main():
    parser = argparse.ArgumentParser(description="Train small pl->en Transformer")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=TrainConfig.max_samples,
        help="Maximum training samples to stream",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TrainConfig.num_epochs,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=TrainConfig.batch_size,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=TrainConfig.output_dir,
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use SmallModelConfig (d_model=128, 2 layers) for CPU/laptop runs",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed-precision (required on CPU)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=TrainConfig.warmup_steps,
        help="LR warmup steps",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=TrainConfig.log_every_n_steps,
        help="Log loss every N steps",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=TrainConfig.save_every_n_steps,
        help="Save checkpoint every N steps",
    )
    args = parser.parse_args()

    use_amp = not args.no_amp and torch.cuda.is_available()
    model_cfg = SmallModelConfig() if args.small else ModelConfig()
    train_cfg = TrainConfig(
        max_samples=args.max_samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        use_amp=use_amp,
        warmup_steps=args.warmup_steps,
        log_every_n_steps=args.log_every,
        save_every_n_steps=args.save_every,
        max_seq_len=model_cfg.max_seq_len,
    )
    train(train_cfg, model_cfg)


if __name__ == "__main__":
    main()
