"""Shared, versioned conditioning format for M5 training and inference.

Vocabulary IDs are append-only. This module never rewrites pretrained assets.
New delimiter embeddings still need CPT; vocabulary migration alone does not
teach instruction following or preserve accent behavior under a changed format.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

CONDITIONING_FORMAT = 'm5-caption-accent-speaker-lang-v3'
CONTROL_TOKENS = (
    '<|caption_begin|>', '<|caption_end|>',
    '<|lang_begin|>', '<|lang_end|>',
    '<|accent_begin|>', '<|accent_end|>',
    '<|speaker_start|>', '<|speaker_end|>',
)
LEGACY_CONTROL_TOKENS = CONTROL_TOKENS[:-2]


def normalize_condition(value, name='condition') -> str:
    """Missing means unspecified, not neutral. Do not stringify containers."""
    if value is None:
        return ''
    if not isinstance(value, str):
        raise TypeError(f'{name} must be a string or null, got {type(value).__name__}')
    value = value.strip()
    if value.lower() in {'none', 'null'}:
        return ''
    if any(marker in value for marker in CONTROL_TOKENS):
        raise ValueError(f'{name} contains a reserved conditioning delimiter')
    return value


def require_lang(value) -> str:
    """Keep the pretraining language slot populated; never infer it from an ID."""
    lang = normalize_condition(value, 'lang')
    if not lang:
        raise ValueError('lang must be an explicit non-empty string matching pretraining (for example "zh" or "en")')
    return lang


def format_conditioned_text(text, speaker=0, accent=None, lang=None, caption=None) -> str:
    if not isinstance(text, str):
        raise TypeError('Conditioned text requires raw text, not cached legacy token IDs')
    if any(marker in text for marker in CONTROL_TOKENS):
        raise ValueError('Spoken text contains a reserved conditioning delimiter')
    caption = normalize_condition(caption, 'caption')
    accent = normalize_condition(accent, 'accent')
    lang = require_lang(lang)
    if isinstance(speaker, bool) or not isinstance(speaker, (str, int, type(None))):
        raise TypeError('speaker must be an integer, string, or null')
    speaker_text = normalize_condition(None if speaker is None else str(speaker), 'speaker')
    if speaker_text in ('', '0'):
        speaker_text = 'None'
    return (
        f'<|caption_begin|>{caption}<|caption_end|>'
        f'<|accent_begin|>{accent}<|accent_end|>'
        f'<|speaker_start|>{speaker_text}<|speaker_end|>'
        f'<|lang_begin|>{lang}<|lang_end|>{text}'
    )


def encode_conditioned_text(tokenizer, text, speaker=0, accent=None, lang=None, caption=None):
    """One encode call: BOS/EOS are applied exactly once by the postprocessor."""
    return tokenizer.encode(format_conditioned_text(text, speaker, accent, lang, caption))


def load_conditioning_tokenizer(tokenizer_path):
    from tokenizers import AddedToken
    from tokenizers.processors import TemplateProcessing
    from transformers import AutoTokenizer

    path = str(Path(tokenizer_path).expanduser())
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    before = tokenizer.get_vocab()
    existing = [token in before for token in CONTROL_TOKENS]
    if existing not in ([False] * len(CONTROL_TOKENS),
                        [True] * len(LEGACY_CONTROL_TOKENS) + [False] * 2,
                        [True] * len(CONTROL_TOKENS)):
        raise ValueError('Partially installed conditioning tokens; use the original tokenizer or a complete saved bundle')
    tokenizer.add_special_tokens(
        {'additional_special_tokens': [AddedToken(t, special=True, normalized=False,
                                                 lstrip=False, rstrip=False)
                                       for t in CONTROL_TOKENS]},
        replace_additional_special_tokens=False,
    )
    after = tokenizer.get_vocab()
    if any(after.get(token) != token_id for token, token_id in before.items()):
        raise ValueError('Adding conditioning tokens changed a pre-existing token ID')
    base_size = len(tokenizer) - len(CONTROL_TOKENS)
    expected_ids = list(range(base_size, len(tokenizer)))
    if [after[t] for t in CONTROL_TOKENS] != expected_ids:
        raise ValueError('Conditioning tokens must occupy the final eight vocabulary IDs in the declared order')
    if tokenizer.bos_token is None or tokenizer.eos_token is None:
        raise ValueError('M5 requires defined BOS and EOS tokens')
    bos, eos = tokenizer.bos_token, tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f'{bos}:0 $A:0 {eos}:0',
        pair=f'{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1',
        special_tokens=[(bos, tokenizer.bos_token_id), (eos, tokenizer.eos_token_id)],
    )
    for token in CONTROL_TOKENS:
        if tokenizer.encode(token, add_special_tokens=False) != [after[token]]:
            raise ValueError(f'Conditioning marker is not atomic: {token}')
    tokenizer._conditioning_base_vocab_size = base_size
    return tokenizer


def conditioning_metadata(tokenizer) -> dict:
    metadata = _conditioning_metadata(tokenizer, CONTROL_TOKENS, CONDITIONING_FORMAT,
                                      ['caption', 'accent', 'speaker', 'lang', 'text'])
    metadata['missing_condition'] = {
        'caption': 'empty_between_delimiters',
        'accent': 'empty_between_delimiters',
        'speaker': 'literal_None',
        'lang': 'empty_between_delimiters',
    }
    return metadata


def _conditioning_metadata(tokenizer, control_tokens, format_name, field_order, vocab_limit=None) -> dict:
    """Use the same metadata algorithm for v2 and the verified v1 prefix."""
    vocab = tokenizer.get_vocab()
    vocab_size = len(tokenizer) if vocab_limit is None else vocab_limit
    if vocab_limit is not None:
        vocab = {token: token_id for token, token_id in vocab.items() if token_id < vocab_limit}
    base_size = vocab_size - len(control_tokens)
    ids = [vocab[t] for t in control_tokens]
    if len(vocab) != vocab_size or ids != list(range(base_size, vocab_size)):
        raise ValueError('Conditioning token IDs are inconsistent with the append-only vocabulary')
    payload = json.dumps(sorted(vocab.items(), key=lambda item: item[1]),
                         ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    # Include the complete processing pipeline, not just spelling-to-ID mapping.
    pipeline = json.loads(tokenizer.backend_tokenizer.to_str())
    if vocab_limit is not None:
        pipeline['added_tokens'] = [token for token in pipeline['added_tokens']
                                    if token['id'] < vocab_limit]
    # Padding/truncation can be transient request state, not training format.
    pipeline['padding'] = None
    pipeline['truncation'] = None
    pipeline_payload = json.dumps(pipeline, ensure_ascii=False, sort_keys=True,
                                  separators=(',', ':')).encode('utf-8')
    return {
        'format': format_name,
        'field_order': field_order,
        'missing_condition': 'empty_between_delimiters',
        'base_vocab_size': base_size,
        'vocab_size': vocab_size,
        'control_token_ids': dict(zip(control_tokens, ids)),
        'vocab_sha256': hashlib.sha256(payload).hexdigest(),
        'tokenizer_pipeline_sha256': hashlib.sha256(pipeline_payload).hexdigest(),
        'bos_token_id': tokenizer.bos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }


def save_conditioning_bundle(directory, tokenizer):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = conditioning_metadata(tokenizer)
    metadata_file = directory / 'conditioning.json'
    if metadata_file.exists():
        validate_conditioning_bundle(directory, tokenizer)
    tokenizer.save_pretrained(str(directory / 'tokenizer'))
    temporary = directory / 'conditioning.json.tmp'
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(metadata_file)
    return metadata


def validate_conditioning_bundle(directory, tokenizer):
    path = Path(directory) / 'conditioning.json'
    if not path.is_file():
        raise FileNotFoundError(f'Missing conditioning metadata: {path}. Provide a complete M5 conditioning bundle.')
    stored = json.loads(path.read_text(encoding='utf-8'))
    expected = conditioning_metadata(tokenizer)
    if stored != expected:
        differing = sorted(k for k in set(stored) | set(expected) if stored.get(k) != expected.get(k))
        raise ValueError(f'Checkpoint/tokenizer conditioning mismatch: {differing}')
    return stored
