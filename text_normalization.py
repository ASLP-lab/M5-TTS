"""Target-text normalization using WeTextProcessing"""
from functools import lru_cache
import os
from pathlib import Path
import re
import threading

_LOCK = threading.RLock()
_LANGUAGES = {"zh": "zh", "chinese": "zh", "en": "en", "english": "en"}
_RESERVED_TOKEN = re.compile(r"<\|[^>\n]*\|>")


def _cache_path(cache_dir):
    if cache_dir is None:
        cache_dir = os.environ.get("M5_TN_CACHE")
    if cache_dir is None:
        cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "m5-tts" / "wetext"
    return str(Path(cache_dir).expanduser().resolve())


@lru_cache(maxsize=8)
def _normalizer(language, cache_dir):
    try:
        if language == "zh":
            from tn.chinese.normalizer import Normalizer
            return Normalizer(cache_dir=cache_dir, remove_interjections=False, remove_erhua=False,
                              traditional_to_simple=False, remove_puncts=False, full_to_half=False)
        from tn.english.normalizer import Normalizer
        return Normalizer(cache_dir=cache_dir)
    except ImportError as error:
        raise ImportError("TN requires WeTextProcessing==1.3.0 and pynini==2.1.7. Install requirements.txt or run: pip install WeTextProcessing==1.3.0 pynini==2.1.7") from error


@lru_cache(maxsize=1)
def _traditional_converter():
    try:
        from opencc import OpenCC
    except ImportError as error:
        raise ImportError("Cantonese text conversion requires opencc-python-reimplemented==0.1.7. Install requirements.txt.") from error
    return OpenCC("s2hk")


def normalize_text(text, *, lang="Chinese", accent=None, cache_dir=None):
    """Normalize target text only; unsupported languages/dialects are passed through.

    Chinese and Sichuan use Chinese TN, preserving punctuation and interjections.
    Cantonese uses Chinese TN followed by Hong Kong traditional conversion, not
    Cantonese-specific verbalization. English uses WeTextProcessing's English TN.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("TN requires nonempty text")
    if _RESERVED_TOKEN.search(text):
        raise ValueError("Do not place model control tokens in text passed to TN")
    language = _LANGUAGES.get(lang.strip().lower() if isinstance(lang, str) else "")
    dialect = str(accent).strip().lower() if accent is not None else ""
    supported_dialect = dialect in ("", "none", "null") or (language == "zh" and dialect in ("chuan", "yue"))
    if language is None or not supported_dialect:
        return text
    with _LOCK:
        result = _normalizer(language, _cache_path(cache_dir)).normalize(text)
        if not isinstance(result, str) or not result.strip():
            raise ValueError("WeTextProcessing returned empty or invalid normalized text")
        if language == "zh" and dialect == "yue":
            result = _traditional_converter().convert(result)
            if not isinstance(result, str) or not result.strip():
                raise ValueError("OpenCC returned empty or invalid converted text")
    return result
