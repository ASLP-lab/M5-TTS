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


def format_conditioned_text(text, speaker=0, accent=None, lang=None, caption=None) -> str:
    if not isinstance(text, str):
        raise TypeError('Conditioned text requires raw text, not cached legacy token IDs')
    if any(marker in text for marker in CONTROL_TOKENS):
        raise ValueError('Spoken text contains a reserved conditioning delimiter')
    caption = normalize_condition(caption, 'caption')
    accent = normalize_condition(accent, 'accent')
    lang = normalize_condition(lang, 'lang')
    if isinstance(speaker, bool) or not isinstance(speaker, (str, int, type(None))):
        raise TypeError('speaker must be an integer, string, or null')
    speaker_text = normalize_condition(None if speaker is None else str(speaker), 'speaker')
    if speaker_text in ('', '0', 0):
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


def resolve_tokenizer_path(checkpoint_path=None, tokenizer_path=None) -> str:
    if tokenizer_path is not None:
        return str(Path(tokenizer_path).expanduser())
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path).expanduser()
        directory = checkpoint if checkpoint.is_dir() else checkpoint.parent
        for candidate in (directory, directory.parent):
            if (candidate / 'tokenizer' / 'tokenizer_config.json').is_file():
                return str(candidate / 'tokenizer')
    raise FileNotFoundError('Provide tokenizer_path or a checkpoint bundle containing tokenizer/ and conditioning.json')


def load_conditioning_tokenizer(tokenizer_path=None):
    from tokenizers import AddedToken
    from tokenizers.processors import TemplateProcessing
    from transformers import AutoTokenizer

    path = resolve_tokenizer_path(tokenizer_path=tokenizer_path)
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
        raise FileNotFoundError(f'Missing conditioning metadata: {path}. Use a new experiment directory for CPT from legacy weights.')
    stored = json.loads(path.read_text(encoding='utf-8'))
    expected = conditioning_metadata(tokenizer)
    if stored != expected:
        differing = sorted(k for k in set(stored) | set(expected) if stored.get(k) != expected.get(k))
        raise ValueError(f'Checkpoint/tokenizer conditioning mismatch: {differing}')
    return stored


def _validate_six_token_checkpoint(checkpoint_path, tokenizer, old_size):
    """Require a real v1 bundle whose vocabulary and pipeline equal the v2 prefix."""
    from transformers import AutoTokenizer

    if checkpoint_path is None:
        raise ValueError('Six-token checkpoint expansion requires checkpoint_path and its verified v1 bundle')
    checkpoint = Path(checkpoint_path).expanduser()
    directory = checkpoint if checkpoint.is_dir() else checkpoint.parent
    for candidate in (directory, directory.parent):
        metadata_path = candidate / 'conditioning.json'
        if metadata_path.is_file():
            break
    else:
        raise FileNotFoundError(f'Missing six-token conditioning bundle for checkpoint: {checkpoint_path}')
    stored = json.loads(metadata_path.read_text(encoding='utf-8'))
    legacy = AutoTokenizer.from_pretrained(str(candidate / 'tokenizer'), local_files_only=True)
    if len(legacy) != old_size or any(token in legacy.get_vocab() for token in CONTROL_TOKENS[-2:]):
        raise ValueError('Checkpoint bundle must contain exactly the saved six-token vocabulary')
    legacy_args = (LEGACY_CONTROL_TOKENS, 'm5-caption-accent-lang-v1', ['caption', 'accent', 'lang', 'text'])
    if stored != _conditioning_metadata(legacy, *legacy_args):
        raise ValueError('Six-token checkpoint metadata does not match its saved tokenizer')
    if stored != _conditioning_metadata(tokenizer, *legacy_args, vocab_limit=old_size):
        raise ValueError('Six-token checkpoint vocabulary or pipeline differs from the current tokenizer prefix')


def load_conditioning_weights(model, weights, tokenizer, allow_vocab_expansion=True, checkpoint_path=None):
    """Strict load, allowing legacy or verified six-token weights -> eight-token CPT.

    Optimizer/scheduler state is deliberately not handled here. Expansion means
    weight-only initialization with a fresh optimizer, never full-state resume.
    """
    import torch

    key = 'text_embeddings.weight'
    if key not in weights:
        raise KeyError(f'Checkpoint is missing {key}')
    source = weights[key]
    target = model.text_embeddings.weight
    if source.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError(f'Text embedding hidden dimension mismatch: {tuple(source.shape)} vs {tuple(target.shape)}')
    metadata = conditioning_metadata(tokenizer)
    old_size, new_size = source.shape[0], target.shape[0]
    if new_size != metadata['vocab_size']:
        raise ValueError(f'Model vocabulary {new_size} does not match tokenizer {metadata["vocab_size"]}')
    expanded = old_size != new_size
    state = dict(weights)
    if expanded:
        if not allow_vocab_expansion:
            raise ValueError('Legacy checkpoint requires CPT before inference with new conditioning tokens')
        if old_size == metadata['base_vocab_size'] + len(LEGACY_CONTROL_TOKENS):
            _validate_six_token_checkpoint(checkpoint_path, tokenizer, old_size)
        elif old_size != metadata['base_vocab_size']:
            raise ValueError(f'Expected legacy vocabulary {metadata["base_vocab_size"]} or its verified six-token extension, got {old_size}')
        # Preserve checkpoint precision. New rows keep the model initializer.
        enlarged = target.detach().to(device=source.device, dtype=source.dtype).clone()
        enlarged[:old_size].copy_(source)
        state[key] = enlarged
    # strict=True also checks every other parameter name and shape.
    model.load_state_dict(state, strict=True)
    if not torch.equal(model.text_embeddings.weight[:old_size].detach().cpu(),
                       source.detach().to(dtype=target.dtype, device='cpu')):
        raise RuntimeError('Legacy text embeddings were not preserved during checkpoint loading')
    return {'expanded': expanded, 'old_vocab_size': old_size, 'new_vocab_size': new_size}
