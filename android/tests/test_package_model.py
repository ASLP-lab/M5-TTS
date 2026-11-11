import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import package_model


class PackagingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / 'model'
        self.voices = self.root / 'voices'
        self.output = self.root / 'output'
        self.model.mkdir()
        self.voices.mkdir()
        np.savez(self.voices / 'voice.npz', codes=np.zeros((2, 16), dtype=np.int64),
                 metadata=json.dumps({'reference_text': '测试', 'encoder_sha256': package_model.ENCODER_SHA256}))
        self.write_json('codec_identity.json', {'encoder_sha256': package_model.ENCODER_SHA256})
        self.write_json('manifest.json', {'checkpoint': '/internal/experiment/model.safetensors',
                                          'checkpoint_sha256': 'test-checkpoint-digest'})
        for name in ('backbone.onnx', 'depth_frame.onnx', 'codec_decoder_cached.onnx',
                     'codec_cached.json', 'text_embeddings.npy', 'audio_embeddings.npy'):
            (self.model / name).write_bytes(b'packaging-test-fixture-not-a-model')
        values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        alphabet = values + list(range(256, 256 + 256 - len(values)))
        self.write_json('tokenizer.json', {
            'normalizer': None,
            'model': {'type': 'BPE', 'ignore_merges': True, 'vocab': {chr(c): i for i, c in enumerate(alphabet)}, 'merges': []},
            'pre_tokenizer': {'pretokenizers': [
                {'type': 'Split', 'behavior': 'Isolated', 'invert': False, 'pattern': {'Regex': '.+'}},
                {'type': 'ByteLevel', 'use_regex': False, 'add_prefix_space': False}]},
            'added_tokens': [],
            'post_processor': {
                'single': [{'SpecialToken': {'id': '<|begin_of_text|>'}},
                           {'Sequence': {'id': 'A'}}, {'SpecialToken': {'id': '<|end_of_text|>'}}],
                'special_tokens': {'<|begin_of_text|>': {'ids': [256]}, '<|end_of_text|>': {'ids': [257]}}},
        })

    def write_json(self, name, data):
        (self.model / name).write_text(json.dumps(data), encoding='utf-8')

    def run_packager(self, arguments=None):
        if arguments is None:
            arguments = ['--model-dir', str(self.model), '--voices', str(self.voices), '--output', str(self.output)]
        with patch.object(sys, 'argv', ['package_model.py', *arguments]), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            package_model.main()

    def test_container_layout_checksums_and_path_redaction(self):
        self.run_packager()
        data = (self.output / 'm5-v20-offline-model.m5model').read_bytes()
        self.assertTrue(data.startswith(package_model.MODEL_CONTAINER_HEADER))
        with zipfile.ZipFile(io.BytesIO(data[len(package_model.MODEL_CONTAINER_HEADER):])) as archive:
            bundle = json.loads(archive.read('bundle.json'))
            self.assertEqual(len(bundle['files']), 9)
            self.assertEqual(set(archive.namelist()), set(bundle['files']) | {'bundle.json'})
            for name, spec in bundle['files'].items():
                content = archive.read(name)
                self.assertEqual(len(content), spec['size'])
                self.assertEqual(hashlib.sha256(content).hexdigest(), spec['sha256'])
            self.assertTrue(archive.read('tokenizer.bin').startswith(b'M5TK1'))
            self.assertEqual(json.loads(archive.read('manifest.json'))['checkpoint'], 'model.safetensors')
            self.assertEqual(json.loads(archive.read('voices.json'))['encoder_sha256'], package_model.ENCODER_SHA256)

    def test_incompatible_codec_is_rejected_without_output(self):
        self.write_json('codec_identity.json', {'encoder_sha256': 'incompatible'})
        with self.assertRaises(SystemExit):
            self.run_packager()
        self.assertFalse(self.output.exists())

    def test_existing_output_is_not_modified(self):
        self.output.mkdir()
        marker = self.output / 'keep.txt'
        marker.write_text('keep')
        with self.assertRaises(SystemExit):
            self.run_packager()
        self.assertEqual(marker.read_text(), 'keep')
        self.assertEqual(list(self.output.iterdir()), [marker])

    def test_inputs_must_be_explicit(self):
        with self.assertRaises(SystemExit):
            self.run_packager([])


if __name__ == '__main__':
    unittest.main()
