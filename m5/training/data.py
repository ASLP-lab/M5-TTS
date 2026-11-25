"""Streaming JSONL shards and length-bucketed dynamic batching for M5 training.

Each training record is a JSON object with at least ``text`` and ``token``:
``token`` is a ``[frames, 16]`` array of Qwen 12 Hz codec codes in 0..2047.
Optional conditioning fields are ``utt``, ``caption``, ``accent``, ``lang`` and
``speaker``; ``wav``, ``duration``, ``start_time`` and ``end_time`` are ignored.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Iterable, Iterator, List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset

from conditioning import format_conditioned_text
from m5.training.text import Segment, _tokenize_segment

LOGGER = logging.getLogger(__name__)


def iter_jsonl(shards: List[str]) -> Iterator[dict[str, Any]]:
    """Stream records from JSONL shards, skipping malformed lines."""
    for shard in shards:
        skipped = 0
        with open(shard, "rb") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    sample = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    skipped += 1
                    LOGGER.warning("Skipping invalid JSONL record at %s:%d: %s", shard, line_number, error)
                    continue
                if not isinstance(sample, dict):
                    skipped += 1
                    LOGGER.warning("Skipping invalid JSONL record at %s:%d: expected a JSON object", shard, line_number)
                    continue
                yield sample
        if skipped:
            LOGGER.warning("Skipped %d invalid JSONL records in %s", skipped, shard)


class BucketDynamicBatch(IterableDataset):
    """Group samples of similar estimated length into token-bounded batches."""

    def __init__(
        self,
        dataset: Iterable[Any],
        max_tokens_per_bucket: int = 13_000,
        max_tokens: int = 6_000,
        num_buckets: int = 100,
        drop_oversized_samples: bool = True,
        text_tokenizer=None,
        max_batch_size: int | None = None,
    ):
        super().__init__()
        if not hasattr(dataset, "__iter__"):
            raise ValueError("dataset must be iterable")
        if max_tokens_per_bucket <= 0 or max_tokens <= 0 or num_buckets <= 0:
            raise ValueError("max_tokens, max_tokens_per_bucket and num_buckets must be positive")
        if max_tokens // num_buckets == 0:
            raise ValueError("max_tokens must be at least num_buckets")
        if max_batch_size is not None and (type(max_batch_size) is not int or max_batch_size < 1):
            raise ValueError("max_batch_size must be a positive integer or None")

        self.dataset = dataset
        self.max_tokens_per_bucket = max_tokens_per_bucket
        self.max_tokens = max_tokens
        self.num_buckets = num_buckets
        self.drop_oversized = drop_oversized_samples
        self.text_tokenizer = text_tokenizer
        self.max_batch_size = max_batch_size
        self.buckets: List[List[Any]] = [[] for _ in range(num_buckets)]
        self.bucket_tokens: List[int] = [0] * num_buckets

    def _bucket_index(self, length: int) -> int:
        step = self.max_tokens // self.num_buckets
        return min(length // step, self.num_buckets - 1)

    def _flush(self, index: int) -> List[Any]:
        samples = self.buckets[index]
        self.buckets[index] = []
        self.bucket_tokens[index] = 0
        return samples

    def __iter__(self) -> Iterator[List[Any]]:
        for sample in self.dataset:
            try:
                audio_len = len(sample.get("token") or [])
                text = sample.get("text", "")
                text_len = len(text) if isinstance(text, str) else 0
                if self.text_tokenizer is not None:
                    # Add the conditioning prefix length for an accurate bucket estimate.
                    prefix = format_conditioned_text(
                        "", speaker=1, caption=sample.get("caption"),
                        accent=sample.get("accent"), lang=sample.get("lang"),
                    )
                    text_len += len(self.text_tokenizer.encode(prefix, add_special_tokens=False))
                sample_len = audio_len + text_len
            except Exception as error:
                LOGGER.warning("Skipping sample: %s", error)
                continue

            if sample_len <= 0:
                LOGGER.warning("Skipping sample with non-positive length")
                continue
            if sample_len > self.max_tokens_per_bucket:
                if not self.drop_oversized:
                    yield [sample]
                else:
                    LOGGER.warning("Dropping oversized sample (%d > %d)", sample_len, self.max_tokens_per_bucket)
                continue

            index = self._bucket_index(sample_len)
            if self.bucket_tokens[index] + sample_len > self.max_tokens and self.buckets[index]:
                yield self._flush(index)
            self.buckets[index].append(sample)
            self.bucket_tokens[index] += sample_len
            if self.max_batch_size is not None and len(self.buckets[index]) >= self.max_batch_size:
                yield self._flush(index)

        for index in range(self.num_buckets):
            if self.buckets[index]:
                yield self._flush(index)

    def __len__(self):
        raise TypeError(f"{self.__class__.__name__} does not have a defined length.")


def tokenize_batch(
    segments: List[Segment],
    text_tokenizer,
    audio_num_codebooks: int = 16,
    pinyin_vocab: set[str] | None = None,
    pinyin_aug_ratio: float = 0.3,
):
    tokens, masks = [], []
    for segment in segments:
        try:
            frame, mask = _tokenize_segment(
                segment, text_tokenizer, audio_num_codebooks=audio_num_codebooks,
                pinyin_vocab=pinyin_vocab, pinyin_aug_ratio=pinyin_aug_ratio,
            )
        except Exception as error:
            LOGGER.warning("Skipping segment: %s", error)
            continue
        tokens.append(frame)
        masks.append(mask)
    if not tokens:
        return (
            torch.empty(0, 0, audio_num_codebooks + 1, dtype=torch.long),
            torch.empty(0, 0, audio_num_codebooks + 1, dtype=torch.bool),
        )
    return (
        pad_sequence(tokens, batch_first=True, padding_value=0),
        pad_sequence(masks, batch_first=True, padding_value=False),
    )


def make_collate_fn(
    text_tokenizer,
    audio_num_codebooks: int = 16,
    pinyin_vocab: set[str] | None = None,
    pinyin_aug_ratio: float = 0.3,
) -> Callable[[List[List[dict]]], dict[str, torch.Tensor]]:
    """Build a collate function for the batches produced by BucketDynamicBatch.

    DataLoader wraps each yielded bucket in a one-element list, so the outer
    list is flattened before individual samples are tokenized.
    """

    def collate_fn(batch: List[List[dict]]) -> dict[str, torch.Tensor]:
        segments: List[Segment] = []
        for group in batch:
            for sample in group:
                text, codes = sample.get("text"), sample.get("token")
                if not isinstance(text, str) or not isinstance(codes, (list, tuple)):
                    LOGGER.warning("Skipping record without raw text or audio codes: %s", sample.get("utt"))
                    continue
                try:
                    audio = torch.as_tensor(codes, dtype=torch.long).transpose(0, 1)
                except (ValueError, RuntimeError) as error:
                    LOGGER.warning("Skipping record %s with invalid audio codes: %s", sample.get("utt"), error)
                    continue
                if audio.shape[0] != audio_num_codebooks:
                    LOGGER.warning(
                        "Skipping record %s: expected %d codebooks, got %d",
                        sample.get("utt"), audio_num_codebooks, audio.shape[0],
                    )
                    continue
                caption = sample.get("caption")
                segments.append(Segment(
                    text=text.strip(),
                    audio=audio + 1,  # 0 is reserved for EOS
                    speaker=sample.get("speaker", 0),
                    accent=sample.get("accent", "None"),
                    lang=sample.get("lang", "None"),
                    caption="None" if caption is None else caption,
                ))

        tokens, tokens_mask = tokenize_batch(
            segments, text_tokenizer, audio_num_codebooks, pinyin_vocab, pinyin_aug_ratio
        )
        return {"tokens": tokens, "tokens_mask": tokens_mask}

    return collate_fn
