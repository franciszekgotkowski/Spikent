"""Dataset loading and tokenization helpers."""

from typing import Iterator

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import MarianTokenizer


def load_translation_dataset(
    dataset_name: str,
    subset: str,
    split: str = "train",
    streaming: bool = True,
):
    """Load the FineTranslations subset from Hugging Face."""
    return load_dataset(
        dataset_name,
        name=subset,
        split=split,
        streaming=streaming,
    )


def tokenize_example(
    example: dict,
    tokenizer: MarianTokenizer,
    src_column: str,
    tgt_column: str,
    max_seq_len: int,
) -> dict:
    """Tokenize a source/target pair for seq2seq."""
    src = example[src_column]
    tgt = example[tgt_column]

    src_tokens = tokenizer(
        src,
        truncation=True,
        max_length=max_seq_len,
        padding=False,
    )
    tgt_tokens = tokenizer(
        tgt,
        truncation=True,
        max_length=max_seq_len,
        padding=False,
    )

    return {
        "input_ids": src_tokens["input_ids"],
        "attention_mask": src_tokens["attention_mask"],
        "labels": tgt_tokens["input_ids"],
    }


class TranslationIterableDataset(IterableDataset):
    """Streaming iterable dataset that tokenizes on the fly."""

    def __init__(
        self,
        hf_dataset,
        tokenizer: MarianTokenizer,
        src_column: str,
        tgt_column: str,
        max_seq_len: int,
        max_samples: int | None = None,
        lang_score_column: str | None = None,
        min_lang_score: float = 0.9,
    ):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.src_column = src_column
        self.tgt_column = tgt_column
        self.max_seq_len = max_seq_len
        self.max_samples = max_samples
        self.lang_score_column = lang_score_column
        self.min_lang_score = min_lang_score

    def __iter__(self) -> Iterator[dict]:
        count = 0
        for example in self.dataset:
            if self.max_samples is not None and count >= self.max_samples:
                break

            # Basic filtering
            if not example.get(self.src_column) or not example.get(self.tgt_column):
                continue

            if (
                self.lang_score_column
                and example.get(self.lang_score_column, 1.0) < self.min_lang_score
            ):
                continue

            tokenized = tokenize_example(
                example,
                self.tokenizer,
                self.src_column,
                self.tgt_column,
                self.max_seq_len,
            )

            # Skip empty or extremely short examples
            if len(tokenized["input_ids"]) < 3 or len(tokenized["labels"]) < 3:
                continue

            count += 1
            yield tokenized


def collate_fn(batch: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Pad a batch of tokenized examples."""
    max_src = max(len(item["input_ids"]) for item in batch)
    max_tgt = max(len(item["labels"]) for item in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        src = item["input_ids"]
        tgt = item["labels"]
        src_pad_len = max_src - len(src)
        tgt_pad_len = max_tgt - len(tgt)

        input_ids.append(src + [pad_token_id] * src_pad_len)
        attention_mask.append([1] * len(src) + [0] * src_pad_len)
        # Use -100 for label padding so it is ignored by loss.
        labels.append(tgt + [-100] * tgt_pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def build_dataloader(
    dataset: IterableDataset,
    tokenizer: MarianTokenizer,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
