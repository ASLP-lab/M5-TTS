"""Offline contract tests for the training pipeline (no weights or network needed)."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from m5.training.data import BucketDynamicBatch, iter_jsonl, make_collate_fn
from m5.training.text import load_pinyin_vocab, random_replace_chinese_with_pinyin
from train import TrainingConfig, log, resolve_source, validate_config

HAS_PINYIN = all(importlib.util.find_spec(name) for name in ("jieba", "pypinyin"))


class FakeTokenizer:
    """Minimal tokenizer standing in for the conditioning tokenizer."""

    _conditioning_base_vocab_size = 100

    def encode(self, text, add_special_tokens=True):
        tokens = [ord(char) % 100 + 2 for char in text]
        return [1] + tokens + [2] if add_special_tokens else tokens


def record(frames=4, text="你好"):
    return {
        "utt": "sample_1", "text": text,
        "token": [[(frame + book) % 2048 for book in range(16)] for frame in range(frames)],
        "caption": "None", "accent": "None", "lang": "Chinese", "speaker": "0",
    }


class TrainingConfigTests(unittest.TestCase):
    def test_example_config_loads(self):
        path = Path(__file__).resolve().parents[1] / "m5" / "configs" / "train_example.json"
        config = TrainingConfig(**json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(config.audio_num_codebooks, 16)
        self.assertEqual(config.attention_type, "flash_attention_2")
        validate_config(config)

    def test_invalid_settings_are_rejected(self):
        for field, value in (("audio_num_codebooks", 8), ("precision", "fp8"),
                             ("report_to", "unknown"), ("total_steps", 0)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_config(TrainingConfig(data_path="data", **{field: value}))

    def test_local_source_is_resolved_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(resolve_source(tmp, ["tokenizer/*"]), Path(tmp).resolve())

    def test_log_only_prints_on_main_process(self):
        with patch("builtins.print") as printer, patch.dict(os.environ, {"RANK": "0"}):
            log("hello", flush=True)
        printer.assert_called_once()
        with patch("builtins.print") as printer, patch.dict(os.environ, {"RANK": "1"}):
            log("hello", flush=True)
        printer.assert_not_called()


class DataPipelineTests(unittest.TestCase):
    def test_iter_jsonl_skips_malformed_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            shard = Path(tmp) / "part.jsonl"
            shard.write_text(json.dumps(record()) + "\n{not json}\n\n[1, 2]\n", encoding="utf-8")
            self.assertEqual(len(list(iter_jsonl([str(shard)]))), 1)

    def test_collate_shapes_and_offsets(self):
        collate = make_collate_fn(FakeTokenizer(), 16)
        batch = collate([[record(frames=4)]])
        tokens, mask = batch["tokens"], batch["tokens_mask"]
        self.assertEqual(tokens.shape[0], 1)
        self.assertEqual(tokens.shape[2], 17)  # 16 audio codebooks + 1 text channel
        self.assertEqual(tokens.shape, mask.shape)
        self.assertTrue(bool(mask[0, 0, -1]))  # first frame is text
        self.assertFalse(bool(mask[0, 0, 0]))
        self.assertFalse(bool(mask[0, -1, -1]))  # EOS frame has no text target
        self.assertEqual(int(tokens[0, -1, 0]), 0)  # EOS frame uses code 0
        self.assertGreater(int(tokens[0, -2, 0]), 0)  # audio codes are shifted by +1

    def test_empty_batch_is_handled(self):
        batch = make_collate_fn(FakeTokenizer(), 16)([[]])
        self.assertEqual(batch["tokens"].ndim, 3)
        self.assertEqual(batch["tokens"].shape[0], 0)

    def test_oversized_samples_are_dropped(self):
        samples = [record(frames=50)]
        batches = list(BucketDynamicBatch(samples, max_tokens_per_bucket=20, max_tokens=20, num_buckets=2))
        self.assertEqual(batches, [])

    def test_missing_pinyin_vocab_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(FileNotFoundError):
            load_pinyin_vocab(Path(tmp) / "missing.lst")

    @unittest.skipUnless(HAS_PINYIN, "optional jieba/pypinyin dependencies")
    def test_pinyin_augmentation_uses_allowlist(self):
        result = random_replace_chinese_with_pinyin("你好", {"NI3", "HAO3"}, 1.0)
        self.assertNotEqual(result, "你好")
        self.assertTrue(any(token in result for token in ("NI3", "HAO3")))

    @unittest.skipUnless(HAS_PINYIN, "optional jieba/pypinyin dependencies")
    def test_collate_with_pinyin_augmentation(self):
        collate = make_collate_fn(FakeTokenizer(), 16, {"NI3", "HAO3"}, 1.0)
        batch = collate([[record(frames=4)]])
        self.assertEqual(batch["tokens"].shape[0], 1)
        self.assertEqual(batch["tokens"].shape[2], 17)


if __name__ == "__main__":
    unittest.main()
