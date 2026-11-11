"""Strict CPU loading of local/Hugging Face M5 and Qwen checkpoint bundles."""
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_CODEC = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
CODEC_FILES = ("codec_encoder.onnx", "codec_decoder.onnx", "codec_decoder_cached.onnx",
               "codec_cached.json", "codec_identity.json")


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def resolve(source, *, revision='main', cache_dir=None, local_files_only=False, patterns=None):
    path = Path(source).expanduser()
    if path.is_dir():
        return path.resolve()
    if path.exists() or path.is_absolute() or str(source).startswith(('.', '~')):
        raise FileNotFoundError(f'Expected a local model directory: {source}')
    return Path(snapshot_download(str(source), revision=revision, cache_dir=cache_dir,
                                  local_files_only=local_files_only, allow_patterns=patterns))


def load_checkpoint(bundle, config_path=None):
    import torch
    from safetensors.torch import load_file
    from conditioning import load_conditioning_tokenizer, validate_conditioning_bundle, load_conditioning_weights
    from m5.model.model import CSM, ModelArgs

    config_path = Path(config_path) if config_path else bundle/'config.json'
    for path in [bundle/'model.safetensors', config_path, bundle/'conditioning.json', bundle/'tokenizer/tokenizer_config.json']:
        if not path.is_file():
            raise FileNotFoundError(f'Missing M5 bundle asset: {path}')
    config = json.loads(config_path.read_text())
    if config.get('audio_vocab_size',2051) != 2051 or config.get('audio_num_codebooks',16) != 16 or config.get('use_codebooks',16) != 16:
        raise ValueError('The current runtime requires audio_vocab_size=2051 and 16 codebooks')
    tokenizer = load_conditioning_tokenizer(bundle/'tokenizer')
    validate_conditioning_bundle(bundle, tokenizer)
    if config.get('text_vocab_size',len(tokenizer)) != len(tokenizer):
        raise ValueError('Tokenizer/config vocabulary mismatch')
    model = CSM(ModelArgs(backbone_flavor=config['backbone_flavor'], decoder_flavor=config['decoder_flavor'],
                         text_vocab_size=len(tokenizer), audio_vocab_size=2051, audio_num_codebooks=16,
                         use_codebooks=16, attention_type='sdpa'))
    weights = load_file(str(bundle/'model.safetensors'),device='cpu')
    load_conditioning_weights(model,weights,tokenizer,allow_vocab_expansion=False)
    return model.eval().to(device='cpu',dtype=torch.float32), tokenizer


def load_codec(path):
    import torch
    from m5.codec_qwen12hz import Qwen3TTSTokenizer

    for name in ('config.json','preprocessor_config.json','model.safetensors'):
        if not (path/name).is_file():
            raise FileNotFoundError(f'Missing audio codec asset: {path/name}')
    config = json.loads((path/'config.json').read_text())
    if config.get('model_type') != 'qwen3_tts_tokenizer_12hz':
        raise ValueError('Only the Qwen 12Hz speech tokenizer is supported')
    tokenizer = Qwen3TTSTokenizer.from_pretrained(str(path),device_map='cpu',dtype=torch.float32,local_files_only=True)
    return tokenizer.model.eval().to(device='cpu',dtype=torch.float32)
