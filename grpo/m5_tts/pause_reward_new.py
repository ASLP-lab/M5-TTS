"""
Character-level pause reward for GRPO.

Scores pause quality at every character boundary and averages across
all boundaries. This normalizes for varying punctuation density —
a sentence with 8 commas and a sentence with 1 comma produce
comparable scores.

Key difference from the old word-level approach:
  - word-level:  scores at word boundaries → N_words-1 scores → mean
  - char-level:  scores at speech-char boundaries → includes ALL character
                 transitions, not just word boundaries. Within-word
                 character gaps are ~0 (ideal for non-punct).

Range [0, 1]. Higher = more natural pausing.
"""

import numpy as np
from typing import List, Tuple, Dict

# ── Punctuation sets (Chinese + English) ──────────────────────
_STRONG_PUNCT = set("。！？.!?")
_WEAK_PUNCT = set("，、；：,;:")

# All punctuation to strip when matching (boundary + decorative)
_ALL_PUNCT = _STRONG_PUNCT | _WEAK_PUNCT | set("""'"".。！？.?!，、；：,;:（）「」『』""''…—·-()[]<>""")

# Space-like characters that separate words in non-CJK text
_SPACE_CHARS = set(" \t\r\n")

# CJK Unicode ranges
_CJK_RANGES = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
    (0x20000, 0x2A6DF), # CJK Unified Ideographs Extension B
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0x3040, 0x309F),   # Hiragana
    (0x30A0, 0x30FF),   # Katakana
    (0xAC00, 0xD7AF),   # Hangul Syllables
]


def _is_cjk(ch: str) -> bool:
    """Check if a character is CJK (Chinese/Japanese/Korean)."""
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _strip_punctuation(text: str) -> str:
    return ''.join(ch for ch in text if ch not in _ALL_PUNCT)


def _build_char_map(
    align_words: List,
    full_text: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[str]]:
    """
    Build speech-unit-level mapping from word alignment + original text.

    For CJK text: each character = one speech unit (character-level pause scoring).
    For Latin text: each word = one speech unit (word-level is the natural unit,
      since individual letters have no phonetic meaning).

    Punctuation is stripped from timestamps but used to annotate boundary types.
    Spaces are treated as word separators (stripped, no boundary annotation).

    Returns
    -------
    unit_starts   : np.ndarray  (n_units,)  start time per speech unit
    unit_ends     : np.ndarray  (n_units,)  end time per speech unit
    unit_bdry     : np.ndarray  (n_units,)  boundary type after this unit
                                              0=none, 1=weak, 2=strong
    unit_indices  : List[int]   (n_units,)  start index of this unit in full_text
    unit_texts    : List[str]   (n_units,)  the speech unit text
    """
    # ── Step 1: split text into speech units (CJK-aware) ──
    # Skip: punctuation (annotation only), spaces (word separators)
    # Keep: CJK chars → each is a unit; Latin letters + digits → grouped into words
    _SKIP = _ALL_PUNCT | _SPACE_CHARS

    unit_indices = []   # start index in full_text
    unit_texts = []     # text of this unit
    i = 0
    while i < len(full_text):
        ch = full_text[i]
        if ch in _SKIP:
            i += 1
            continue

        if _is_cjk(ch):
            # CJK character → standalone unit
            unit_indices.append(i)
            unit_texts.append(ch)
            i += 1
        else:
            # Latin letter / digit → group consecutive non-CJK, non-skip chars
            start = i
            buf = []
            while i < len(full_text) and full_text[i] not in _SKIP and not _is_cjk(full_text[i]):
                buf.append(full_text[i])
                i += 1
            if buf:
                unit_indices.append(start)
                unit_texts.append(''.join(buf))

    n_units = len(unit_indices)
    if n_units == 0:
        return (
            np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int32),
            [], []
        )

    # ── Step 2: match alignment words to speech units ──
    # clean_full = concatenation of speech unit texts (no spaces/punct)
    clean_full = ''.join(unit_texts)

    word_unit_ranges = []  # [(start, end) in clean_full / unit indices]
    pos = 0
    for w in align_words:
        wtext = w.text
        idx = clean_full.find(wtext, pos)
        if idx == -1:
            idx = clean_full.lower().find(wtext.lower(), pos)
        if idx == -1:
            if word_unit_ranges:
                word_unit_ranges.append((word_unit_ranges[-1][1], word_unit_ranges[-1][1]))
            else:
                word_unit_ranges.append((0, 0))
            continue
        end = idx + len(wtext)
        pos = end
        word_unit_ranges.append((idx, end))

    # ── Step 3: assign timestamps to each speech unit ──
    unit_starts = np.zeros(n_units, dtype=np.float64)
    unit_ends = np.zeros(n_units, dtype=np.float64)

    for wi, w in enumerate(align_words):
        if wi >= len(word_unit_ranges):
            break
        c_start, c_end = word_unit_ranges[wi]
        span = c_end - c_start  # how many chars in clean_full this word covers
        if span <= 0:
            continue

        w_start = w.start_time
        w_end = w.end_time
        w_dur = w_end - w_start

        # Map the clean_full character range back to speech unit indices
        # by finding which units contribute to this span
        unit_pos = 0
        for ui, utext in enumerate(unit_texts):
            unit_len = len(utext)
            unit_end_pos = unit_pos + unit_len

            # Overlap between [unit_pos, unit_end_pos) and [c_start, c_end)
            overlap_start = max(unit_pos, c_start)
            overlap_end = min(unit_end_pos, c_end)
            if overlap_start < overlap_end:
                # This unit overlaps with the alignment word
                frac_start = (overlap_start - c_start) / span
                frac_end = (overlap_end - c_start) / span
                unit_starts[ui] = w_start + frac_start * w_dur
                unit_ends[ui] = w_start + frac_end * w_dur

            unit_pos = unit_end_pos
            if unit_pos >= c_end:
                break

    # ── Step 4: determine boundary type between speech units ──
    # bdry[i] = what punctuation (if any) is between unit i and unit i+1
    unit_bdry = np.zeros(n_units, dtype=np.int32)

    for i in range(n_units - 1):
        # Look at characters between unit i's end and unit i+1's start in full_text
        end_i = unit_indices[i] + len(unit_texts[i])
        start_j = unit_indices[i + 1]
        for j in range(end_i, start_j):
            ch = full_text[j]
            if ch in _STRONG_PUNCT:
                unit_bdry[i] = max(unit_bdry[i], 2)
            elif ch in _WEAK_PUNCT:
                unit_bdry[i] = max(unit_bdry[i], 1)

    # Check trailing punctuation after last unit
    last_end = unit_indices[-1] + len(unit_texts[-1])
    for j in range(last_end, len(full_text)):
        ch = full_text[j]
        if ch in _STRONG_PUNCT:
            unit_bdry[-1] = max(unit_bdry[-1], 2)
        elif ch in _WEAK_PUNCT:
            unit_bdry[-1] = max(unit_bdry[-1], 1)

    return unit_starts, unit_ends, unit_bdry, unit_indices, unit_texts


def compute_char_level_pause_reward(
    align: List,
    text: str,
    strong_range: Tuple[float, float] = (0.15, 0.80),
    weak_range: Tuple[float, float] = (0.03, 0.35),
    nonpunct_max: float = 0.10,
) -> Tuple[float, Dict]:
    """
    Character-level pause reward.

    For each speech-character boundary in the input text:
      - Determine expected pause type from punctuation between chars
      - Measure actual gap in the audio
      - Score how well they match
    Then average across ALL speech-character boundaries.

    Parameters
    ----------
    align : list of alignment items  (.text, .start_time, .end_time)
        Forced alignment output.
    text : str
        Original text WITH punctuation marks.
    strong_range : (float, float)
        Expected pause range after 。！？ (seconds).
    weak_range : (float, float)
        Expected pause range after ，、；： (seconds).
    nonpunct_max : float
        Maximum acceptable gap for non-punctuation boundaries (seconds).

    Returns
    -------
    reward  : float  [0, 1] — higher = more natural pausing
    details : dict
    """
    n_words = len(align)
    if n_words < 1:
        return 1.0, {"error": "empty alignment", "n_speech_units": 0, "n_boundaries": 0}

    unit_starts, unit_ends, unit_bdry, unit_indices, unit_texts = \
        _build_char_map(align, text)

    n_units = len(unit_texts)
    if n_units < 2:
        return 1.0, {"n_speech_units": n_units, "n_boundaries": 0}

    # ── Compute actual gap at each speech-unit boundary ──
    # gap[i] = time between unit i and unit i+1
    n_boundaries = n_units - 1
    gaps = unit_starts[1:] - unit_ends[:-1]
    gaps = np.maximum(gaps, 0.0)  # clip negative overlaps to 0

    # bdry[i] applies to gap[i]
    bdry = unit_bdry[:n_boundaries]

    # ── Score each boundary ──
    scores = np.ones(n_boundaries, dtype=np.float64)

    for i in range(n_boundaries):
        gap = gaps[i]
        b = bdry[i]

        if b == 2:  # strong boundary
            lo, hi = strong_range
            if gap < lo:
                scores[i] = max(0.0, 1.0 - (lo - gap) / lo)
            elif gap > hi:
                scores[i] = max(0.0, 1.0 - (gap - hi) / hi)
            else:
                scores[i] = 1.0
        elif b == 1:  # weak boundary
            lo, hi = weak_range
            if gap < lo:
                scores[i] = max(0.0, 1.0 - (lo - gap) / lo)
            elif gap > hi:
                scores[i] = max(0.0, 1.0 - (gap - hi) / hi)
            else:
                scores[i] = 1.0
        else:  # no punctuation — only penalise abnormal pauses (gap > nonpunct_max)
            if gap > nonpunct_max:
                scores[i] = max(0.0, 1.0 - (gap - nonpunct_max) / nonpunct_max)
            else:
                scores[i] = 1.0

    # ── Aggregate: mean score across all speech-unit boundaries ──
    mean_score = float(np.mean(scores))
    reward = float(mean_score)

    # ── Per-category breakdown ──
    strong_mask = bdry == 2
    weak_mask = bdry == 1
    none_mask = bdry == 0

    # ── Build punctuation info for gap_details ──
    # For each boundary, extract the punctuation between the two speech units
    boundary_puncts = []
    for i in range(n_boundaries):
        puncts = []
        end_i = unit_indices[i] + len(unit_texts[i])
        start_j = unit_indices[i + 1]
        for j in range(end_i, start_j):
            ch = text[j]
            if ch in _STRONG_PUNCT | _WEAK_PUNCT:
                puncts.append(ch)
        boundary_puncts.append(''.join(puncts))

    gap_details = []
    for i in range(n_boundaries):
        gap_details.append({
            "unit_idx": unit_indices[i],
            "unit_before": unit_texts[i],
            "unit_after": unit_texts[i + 1],
            "punct_between": boundary_puncts[i],
            "gap_ms": round(float(gaps[i]) * 1000, 1),
            "boundary": int(bdry[i]),
            "score": round(float(scores[i]), 4),
        })

    details = {
        "reward": reward,
        "mean_score": mean_score,
        "n_speech_units": n_units,
        "n_boundaries": n_boundaries,
        "n_strong": int(strong_mask.sum()),
        "n_weak": int(weak_mask.sum()),
        "n_none": int(none_mask.sum()),
        "mean_score_strong": float(np.mean(scores[strong_mask])) if strong_mask.sum() > 0 else None,
        "mean_score_weak": float(np.mean(scores[weak_mask])) if weak_mask.sum() > 0 else None,
        "mean_score_none": float(np.mean(scores[none_mask])) if none_mask.sum() > 0 else None,
        "mean_gap_strong_ms": float(np.mean(gaps[strong_mask]) * 1000) if strong_mask.sum() > 0 else None,
        "mean_gap_weak_ms": float(np.mean(gaps[weak_mask]) * 1000) if weak_mask.sum() > 0 else None,
        "mean_gap_none_ms": float(np.mean(gaps[none_mask]) * 1000) if none_mask.sum() > 0 else None,
        "gap_details": gap_details,
    }

    return reward, details


# ═══════════════════════════════════════════════════════════
#  Convenience
# ═══════════════════════════════════════════════════════════
