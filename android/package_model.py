#!/usr/bin/env python3
"""Package a compatible FP32 ONNX directory and prepared voices for Android."""
import argparse
import hashlib
import json
import shutil
import struct
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_CONTAINER_HEADER = b'M5MODEL-ZIP-V1\r\n'
ENCODER_SHA256 = 'ae0364957dd52df283e88baea8ea09d1f1f34b579cd53d2806b38290ebc03b57'


def package_for_browser(source, destination):
    """Keep the ZIP payload intact while preventing browser archive auto-extraction."""
    temporary = destination.with_suffix(destination.suffix + '.part')
    with source.open('rb') as src, temporary.open('wb') as dst:
        dst.write(MODEL_CONTAINER_HEADER)
        shutil.copyfileobj(src, dst, 4 * 1024 * 1024)
    temporary.replace(destination)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def export_tokenizer(source, destination):
    data = json.loads(source.read_text())
    model = data['model']
    if data['normalizer'] is not None or model['type'] != 'BPE' or not model['ignore_merges']:
        raise ValueError('Unsupported tokenizer pipeline')
    pre = data['pre_tokenizer']['pretokenizers']
    if (len(pre) != 2 or pre[0]['type'] != 'Split' or pre[0]['behavior'] != 'Isolated'
            or pre[0]['invert'] or pre[1]['type'] != 'ByteLevel'
            or pre[1]['use_regex'] or pre[1]['add_prefix_space']):
        raise ValueError('Unsupported pre-tokenizer')
    for token in data['added_tokens']:
        if any(token[k] for k in ['single_word', 'lstrip', 'rstrip', 'normalized']):
            raise ValueError('Unsupported added-token flags')
    post = data['post_processor']
    if [x.get('SpecialToken', {}).get('id') for x in [post['single'][0], post['single'][-1]]] != [
            '<|begin_of_text|>', '<|end_of_text|>'] or len(post['single']) != 3:
        raise ValueError('Unsupported post-processor')
    vocab = model['vocab']
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    byte_unicode = dict(zip(byte_values, byte_values))
    for b in range(256):
        if b not in byte_unicode:
            byte_unicode[b] = 256 + sum(v >= 256 for v in byte_unicode.values())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('wb') as f:
        def integer(v):
            f.write(struct.pack('>i', v))

        def string(s):
            encoded = s.encode('utf8'); integer(len(encoded)); f.write(encoded)

        f.write(b'M5TK1')
        for name in ['<|begin_of_text|>', '<|end_of_text|>']:
            integer(post['special_tokens'][name]['ids'][0])
        string(pre[0]['pattern']['Regex'])
        integer(len(vocab))
        for token, token_id in vocab.items():
            string(token); integer(token_id)
        for b in range(256):
            integer(byte_unicode[b]); integer(vocab[chr(byte_unicode[b])])
        integer(len(model['merges']))
        for pair in model['merges']:
            a, b = pair if isinstance(pair, list) else pair.split(' ')
            integer(vocab[a]); integer(vocab[b]); integer(vocab[a + b])
        integer(len(data['added_tokens']))
        for token in data['added_tokens']:
            string(token['content']); integer(token['id'])


def main():
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True, help='Compatible FP32 ONNX directory')
    parser.add_argument('--voices', type=Path, required=True, help='Directory of prepared, authorized .npz voices')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/android_package')
    args = parser.parse_args()
    if not args.voices.is_dir() or not any(args.voices.glob('*.npz')):
        parser.error('--voices must contain prepared, authorized .npz voices')
    identity = json.loads((args.model_dir / 'codec_identity.json').read_text())['encoder_sha256']
    if identity != ENCODER_SHA256:
        parser.error('Codec encoder identity does not match this Android app; use its compatible codec assets')
    meta = json.loads((args.model_dir / 'manifest.json').read_text())
    meta['checkpoint'] = 'model.safetensors'
    if not meta.get('checkpoint_sha256'):
        raise ValueError('Model manifest must record checkpoint_sha256 at export time')
    if args.output.exists():
        parser.error('--output must not already exist; choose a new directory')
    args.output.mkdir(parents=True)
    extra = args.output / 'bundle_metadata'; extra.mkdir()
    export_tokenizer(args.model_dir / 'tokenizer.json', extra / 'tokenizer.bin')
    (extra / 'manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    voices = []
    for path in sorted(args.voices.glob('*.npz')):
        with np.load(path, allow_pickle=False) as voice:
            info = json.loads(str(voice['metadata'].item())); codes = voice['codes']
            if info['encoder_sha256'] != identity or codes.ndim != 2 or codes.shape[1] != 16:
                raise ValueError(f'Incompatible voice: {path}')
            if not info['reference_text'].strip() or codes.min() < 0 or codes.max() > 2047:
                raise ValueError(f'Invalid voice: {path}')
            voices.append(dict(name=f'音色 {len(voices) + 1}', id=path.stem,
                               reference_text=info['reference_text'], codes=codes.astype(int).tolist()))
    if not voices:
        raise ValueError('No prepared voices')
    (extra / 'voices.json').write_text(json.dumps(dict(encoder_sha256=identity, voices=voices), ensure_ascii=False) + '\n')
    files = {name: args.model_dir / name for name in [
        'backbone.onnx', 'depth_frame.onnx', 'codec_decoder_cached.onnx', 'codec_cached.json',
        'text_embeddings.npy', 'audio_embeddings.npy']}
    files.update({name: extra / name for name in ['tokenizer.bin', 'voices.json', 'manifest.json']})
    manifest = dict(format='m5-android-offline-v1', precision='fp32', runtime='onnxruntime-1.23.2',
                    checkpoint_sha256=meta['checkpoint_sha256'], files={})
    for name, path in files.items():
        manifest['files'][name] = dict(size=path.stat().st_size, sha256=sha256(path))
        print('Verified', name, path.stat().st_size, flush=True)
    bundle = args.output / 'm5-v20-offline-model.zip'
    temporary = bundle.with_suffix('.zip.part')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        archive.writestr('bundle.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        for name, path in files.items():
            archive.write(path, name)
    temporary.replace(bundle)
    (args.output / 'bundle.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print('Ready', bundle, bundle.stat().st_size, 'sha256', sha256(bundle), flush=True)
    container = args.output / 'm5-v20-offline-model.m5model'
    package_for_browser(bundle, container)
    print('Browser-safe package', container, container.stat().st_size, 'sha256', sha256(container), flush=True)


if __name__ == '__main__':
    main()
