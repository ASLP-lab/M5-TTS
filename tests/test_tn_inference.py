from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

import inference
import infer_onnx

ROOT = Path(__file__).resolve().parents[1]


class TorchTNTests(unittest.TestCase):
    def setUp(self):
        self.generator = SimpleNamespace(sample_rate=24000, device=torch.device('cpu'),
                                          generate=Mock(return_value=torch.ones(4000)))
        self.model = inference.M5TTS(self.generator)

    def test_default_normalizes_with_real_rules(self):
        self.model.infer('12元', speaker_id=1)
        self.assertEqual(self.generator.generate.call_args.kwargs['text'], '十二元')

    def test_speaker_normalization(self):
        with patch('text_normalization.normalize_text', return_value='十二元') as tn:
            self.model.infer('12元', speaker_id=1)
            tn.assert_called_once_with('12元', lang='Chinese', accent=None)
        call = self.generator.generate.call_args.kwargs
        self.assertEqual(call['text'], '十二元')
        self.assertEqual(call['context'][0].speaker, 1)

    def test_clone_normalizes_target_not_reference_or_conditions(self):
        with patch('text_normalization.normalize_text', return_value='十二元') as tn, \
             patch('inference.prepare_prompt', return_value=torch.ones(1000)):
            self.model.infer('12元', prompt_wav='reference.wav', prompt_text='准确的3句转写', caption='快乐2分')
            tn.assert_called_once_with('12元', lang='Chinese', accent=None)
        segment = self.generator.generate.call_args.kwargs['context'][0]
        self.assertEqual(segment.text, '准确的3句转写十二元')
        self.assertEqual(segment.caption, '快乐2分')

    def test_dialect_clone_normalizes_only_target(self):
        for accent, expected in [('chuan', '这件衣服十二元。'), ('yue', '這件衣服十二元。')]:
            with self.subTest(accent=accent), patch('inference.prepare_prompt', return_value=torch.ones(1000)):
                self.model.infer('这件衣服12元。', prompt_wav='reference.wav',
                                 prompt_text='参考3元。', accent=accent, caption='快乐2分')
                segment = self.generator.generate.call_args.kwargs['context'][0]
                self.assertEqual(segment.text, '参考3元。' + expected)
                self.assertEqual(segment.accent, accent)
                self.assertEqual(segment.lang, 'Chinese')
                self.assertEqual(segment.caption, '快乐2分')

    def test_dialect_speaker_normalization(self):
        for accent, expected in [('chuan', '这件衣服十二元。'), ('yue', '這件衣服十二元。')]:
            with self.subTest(accent=accent):
                self.model.infer('这件衣服12元。', accent=accent)
                call = self.generator.generate.call_args.kwargs
                self.assertEqual(call['text'], expected)
                self.assertEqual(call['context'][0].accent, accent)

    def test_multilingual_target_only(self):
        cases = [('Japanese', 'りんごを12個買いました。', 'りんごを十二個買いました。'),
                 ('Korean', '가격은 12000원입니다.', '가격은 만이천 원입니다.'),
                 ('German', 'Ich habe 12 Äpfel gekauft.', 'Ich habe zwölf Äpfel gekauft.'),
                 ('French', 'J’ai acheté 12 pommes.', 'J’ai acheté douze pommes.')]
        for lang, text, expected in cases:
            for cloning in (False, True):
                with self.subTest(lang=lang, cloning=cloning), \
                     patch('inference.prepare_prompt', return_value=torch.ones(1000)):
                    kwargs = dict(prompt_wav='reference.wav', prompt_text='reference 3.5 ') if cloning else {}
                    self.model.infer(text, lang=lang, caption='style 2', **kwargs)
                    call = self.generator.generate.call_args.kwargs
                    self.assertEqual(call['context'][0].text if cloning else call['text'],
                                     'reference 3.5' + expected if cloning else expected)
                    self.assertEqual(call['context'][0].lang, lang)
                    self.assertEqual(call['context'][0].caption, 'style 2')

    def test_unsupported_language_preserves_raw_text(self):
        self.model.infer('12 mele', lang='Italian')
        self.assertEqual(self.generator.generate.call_args.kwargs['text'], '12 mele')

    def test_tn_failure_does_not_generate(self):
        with patch('text_normalization.normalize_text', side_effect=ImportError('TN missing')):
            with self.assertRaises(ImportError):
                self.model.infer('12元')
        self.generator.generate.assert_not_called()

    def test_cli_normalizes_without_extra_flags(self):
        for name, extra in [('infer_clone.py', ['--prompt-wav','reference.wav','--prompt-text','参考3元']),
                            ('infer_speaker.py', ['--speaker-id','1'])]:
            with self.subTest(name=name), patch.object(sys, 'argv', [name,'--text','12元',*extra]), \
                 patch.object(inference, 'load_from_args', return_value=self.model), \
                 patch.object(self.model, 'save', return_value='output.wav'), \
                 patch('inference.prepare_prompt', return_value=torch.ones(1000)):
                runpy.run_path(str(ROOT/name), run_name='__main__')
                call = self.generator.generate.call_args.kwargs
                self.assertEqual(call['context'][0].text if name == 'infer_clone.py' else call['text'],
                                 '参考3元十二元' if name == 'infer_clone.py' else '十二元')
        self.assertNotIn('--tn', inference.inference_parser('test').format_help())


class OnnxTNTests(unittest.TestCase):
    def runtime(self):
        model = object.__new__(infer_onnx.M5Onnx)
        model.sample_rate = 24000
        model.provider = 'CPUExecutionProvider'
        model.device, model.device_id = 'cpu', 0
        model.session_providers = {}
        model.provider_options = {}
        model.threads = 1
        model.codec_mode = 'window'
        model.meta = dict(hop_length=1920,max_sequence=2048,layers=1,kv_heads=1,head_dim=2,audio_vocab=2051,min_frames=6)
        model.offsets = np.arange(16)*2051
        model.audio_embeddings = np.zeros((16*2051,2),np.float32)
        model.text_embeddings = np.zeros((8,2),np.float32)
        model.tokenizer = Mock()
        model.tokenizer.encode.return_value = SimpleNamespace(ids=[1,2])
        model.encode_reference = Mock(return_value=np.zeros((1,16),np.int64))
        model.sessions = {'backbone':Mock(), 'depth_frame':Mock()}
        model.sessions['backbone'].run.return_value = (np.zeros((1,2),np.float32),np.empty((1,),np.float32))
        model.sessions['depth_frame'].run.side_effect = [(np.full((1,16),2),np.zeros((1,1,2),np.float32)),
                                                       (np.full((1,16),3),np.zeros((1,1,2),np.float32)),
                                                       (np.zeros((1,16),np.int64),np.zeros((1,1,2),np.float32))]
        model._decode = Mock(side_effect=lambda codes:np.ones(len(codes)*1920,np.float32))
        return model

    def test_target_only_and_stats(self):
        model = self.runtime()
        with patch('infer_onnx.read_reference', return_value=np.ones(24000,np.float32)), \
             patch('text_normalization.normalize_text', return_value='十二元') as tn:
            audio = model.infer('12元','ref.wav','参考3元')
        tn.assert_called_once_with('12元', lang='Chinese')
        text = model.tokenizer.encode.call_args.args[0]
        self.assertTrue(text.endswith('参考3元十二元'))
        self.assertEqual(model.stats['normalized_text'],'十二元')
        self.assertEqual(model.stats['original_text'],'12元')
        self.assertTrue(model.stats['eos'])
        self.assertGreater(len(audio),0)

    def test_default_normalizes_with_real_rules(self):
        model = self.runtime()
        with patch('infer_onnx.read_reference', return_value=np.ones(24000,np.float32)):
            model.infer('12元','ref.wav','参考3元')
        self.assertTrue(model.tokenizer.encode.call_args.args[0].endswith('参考3元十二元'))
        self.assertEqual(model.stats['normalized_text'], '十二元')
        self.assertEqual(model.stats['original_text'], '12元')

    def test_multilingual_text_and_stats(self):
        cases = [('Japanese', '12個', '十二個'), ('Korean', '12000원', '만이천원'),
                 ('German', '12 Äpfel', 'zwölf Äpfel'), ('French', '12 pommes', 'douze pommes')]
        for lang, text, expected in cases:
            model = self.runtime()
            with self.subTest(lang=lang), patch('infer_onnx.read_reference', return_value=np.ones(24000,np.float32)):
                model.infer(text, 'ref.wav', 'reference 3.5 ', lang=lang)
                conditioned = model.tokenizer.encode.call_args.args[0]
                self.assertTrue(conditioned.endswith('reference 3.5' + expected))
                self.assertIn('<|lang_begin|>' + lang + '<|lang_end|>', conditioned)
                self.assertEqual(model.stats['original_text'], text)
                self.assertEqual(model.stats['normalized_text'], expected)

    def test_normalizer_runs_before_reference_encoding(self):
        model = self.runtime()
        with patch('text_normalization.normalize_text',side_effect=ImportError('TN missing')):
            with self.assertRaises(ImportError):
                list(model.generate('12元','ref.wav','参考'))
        model.encode_reference.assert_not_called()


if __name__=='__main__':
    unittest.main()
