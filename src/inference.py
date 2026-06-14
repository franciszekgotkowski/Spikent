#!/usr/bin/env python3
"""Inference script for the trained Polish -> English model."""

import argparse

import torch
from transformers import MarianTokenizer

from config import ModelConfig, SmallModelConfig
from src.model import TransformerTranslator


def load_checkpoint(checkpoint_dir: str, device: str | None = None, small: bool = False):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = MarianTokenizer.from_pretrained(checkpoint_dir)

    base_cfg = SmallModelConfig if small else ModelConfig
    cfg = base_cfg(
        src_vocab_size=len(tokenizer),
        tgt_vocab_size=len(tokenizer),
    )
    model = TransformerTranslator(cfg)

    ckpt = torch.load(
        f"{checkpoint_dir}/checkpoint.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, tokenizer, device


def translate_text(
    model: TransformerTranslator,
    tokenizer: MarianTokenizer,
    text: str,
    device: str,
    max_len: int = 128,
) -> str:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    )
    src = inputs["input_ids"].to(device)

    bos_id = tokenizer.bos_token_id or tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id

    output = model.translate(src, bos_id=bos_id, eos_id=eos_id, max_len=max_len)
    output = output[0].tolist()

    # Cut at first EOS if present.
    if eos_id in output:
        output = output[: output.index(eos_id)]
    return tokenizer.decode(output, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    parser.add_argument("--text", default="Dzień dobry. Jak się masz?")
    parser.add_argument("--device", default=None)
    parser.add_argument("--small", action="store_true", help="Use SmallModelConfig")
    args = parser.parse_args()

    model, tokenizer, device = load_checkpoint(args.checkpoint, args.device, args.small)
    translation = translate_text(model, tokenizer, args.text, device)
    print(f"PL: {args.text}")
    print(f"EN: {translation}")


if __name__ == "__main__":
    main()
