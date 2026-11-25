"""Text and audio tokenization for M5 training.

The conditioning format is shared with inference through ``conditioning.py``.
Training records carry Qwen 12 Hz codec codes as ``[frames, 16]`` integer
arrays in 0..2047. A +1 offset is applied before tokenization so that 0 can
serve as the EOS frame.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch

from conditioning import encode_conditioned_text


@dataclass
class Segment:
    text: str
    audio: torch.Tensor  # [codebooks, frames], already shifted by +1
    speaker: int | str = 0
    accent: str | None = "None"
    lang: str | None = "None"
    caption: str | None = "None"


def load_pinyin_vocab(path) -> set[str]:
    """Load an optional user-provided pinyin allowlist, one syllable per line."""
    file = Path(path).expanduser()
    if not file.is_file():
        raise FileNotFoundError(
            f"Pinyin vocabulary not found: {file}. Set pinyin_vocab_path=null to disable augmentation."
        )
    return {line.strip().upper() for line in file.read_text(encoding="utf-8").splitlines() if line.strip()}


def _pinyin_tools():
    try:
        import jieba
        from pypinyin import Style, pinyin
    except ImportError as error:
        raise ImportError(
            "Pinyin augmentation requires optional dependencies: pip install jieba pypinyin"
        ) from error
    return jieba, pinyin, Style


def _is_chinese_char(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def _correct_pinyin_for_text(text: str) -> dict[int, str | None]:
    """Segment first so polyphonic characters get context-dependent readings."""
    jieba, pinyin, Style = _pinyin_tools()
    char2pinyin: dict[int, str | None] = {}
    offset = 0
    for word in jieba.cut(text, cut_all=False):
        chars = text[offset:offset + len(word)]
        if chars and all(_is_chinese_char(char) for char in chars):
            readings = pinyin(chars, style=Style.TONE3, neutral_tone_with_five=True, heteronym=False)
            for index, char in enumerate(chars):
                char2pinyin[offset + index] = readings[index][0].upper()
        else:
            for index in range(len(chars)):
                char2pinyin[offset + index] = None
        offset += len(word)
    return char2pinyin


def random_replace_chinese_with_pinyin(
    text: str, pinyin_vocab: set[str], max_replace_ratio: float = 0.3
) -> str:
    """Replace a random subset of Chinese characters with their upper-case pinyin."""
    if not text or max_replace_ratio <= 0:
        return text
    characters = list(text)
    readings = _correct_pinyin_for_text(text)
    chinese_indices = [i for i, char in enumerate(characters) if _is_chinese_char(char)]
    if not chinese_indices:
        return text
    max_replace = max(int(len(chinese_indices) * max_replace_ratio), 1)
    replace_num = random.randint(1, min(max_replace, len(chinese_indices)))
    for index in random.sample(chinese_indices, replace_num):
        reading = readings.get(index)
        if reading and reading in pinyin_vocab:
            characters[index] = reading
    return "".join(characters)


def _tokenize_text_segment(
    text_tokenizer,
    text: str,
    speaker: int | str,
    accent: str | None,
    audio_num_codebooks: int = 16,
    lang: str | None = "None",
    caption: str | None = "None",
    pinyin_vocab: set[str] | None = None,
    pinyin_aug_ratio: float = 0.3,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(text, str):
        raise TypeError("Conditioning requires raw text, not cached token IDs")
    if pinyin_vocab is not None and (accent is None or str(accent).strip().lower() in ("", "none", "null")):
        text = random_replace_chinese_with_pinyin(text, pinyin_vocab, pinyin_aug_ratio)

    text_tokens = torch.tensor(
        encode_conditioned_text(
            text_tokenizer, text, speaker=speaker, accent=accent, lang=lang, caption=caption
        ),
        dtype=torch.long,
    )
    frame = torch.zeros(len(text_tokens), audio_num_codebooks + 1, dtype=torch.long)
    mask = torch.zeros(len(text_tokens), audio_num_codebooks + 1, dtype=torch.bool)
    frame[:, -1] = text_tokens
    mask[:, -1] = True
    return frame.to(device), mask.to(device)


def _tokenize_audio(
    audio: torch.Tensor, audio_num_codebooks: int = 16, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    if audio.ndim != 2 or audio.shape[0] != audio_num_codebooks:
        raise ValueError(
            f"Expected audio codes shaped [{audio_num_codebooks}, frames], got {tuple(audio.shape)}"
        )
    tokens = audio.to(device)
    eos = torch.zeros(tokens.shape[0], 1, dtype=tokens.dtype, device=device)
    tokens = torch.cat([tokens, eos], dim=1)  # [codebooks, frames + 1]
    frame = torch.zeros(tokens.shape[1], audio_num_codebooks + 1, dtype=torch.long, device=device)
    mask = torch.zeros(tokens.shape[1], audio_num_codebooks + 1, dtype=torch.bool, device=device)
    frame[:, :-1] = tokens.transpose(0, 1)
    mask[:, :-1] = True
    return frame, mask


def _tokenize_segment(
    segment: Segment,
    text_tokenizer,
    audio_num_codebooks: int = 16,
    pinyin_vocab: set[str] | None = None,
    pinyin_aug_ratio: float = 0.3,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return frame tokens and masks shaped [frames, audio_num_codebooks + 1]."""
    text_frame, text_mask = _tokenize_text_segment(
        text_tokenizer, segment.text, segment.speaker, segment.accent,
        audio_num_codebooks=audio_num_codebooks, lang=segment.lang, caption=segment.caption,
        pinyin_vocab=pinyin_vocab, pinyin_aug_ratio=pinyin_aug_ratio, device=device,
    )
    audio_frame, audio_mask = _tokenize_audio(segment.audio, audio_num_codebooks, device)
    return torch.cat([text_frame, audio_frame], dim=0), torch.cat([text_mask, audio_mask], dim=0)
