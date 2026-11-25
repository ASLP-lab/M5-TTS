import unittest
from unittest.mock import Mock, patch

import gradio as gr
import torch

from app import DemoBackend, DemoSettings, REFERENCE_AUDIO, REFERENCE_TEXT, create_demo
from inference import M5TTS
from text_normalization import normalize_text


class WebTNTests(unittest.TestCase):
    def setUp(self):
        self.backend = DemoBackend(DemoSettings(device='cpu', attention='sdpa'))
        self.inputs = [str(REFERENCE_AUDIO), REFERENCE_TEXT, '这件衣服99.5元。', 'zh', 'None', .9, 10, 42]

    def test_web_leaves_normalization_to_inference(self):
        fake = Mock(sample_rate=24000)
        fake.infer.return_value = torch.ones(4000) * .1
        with patch('app.M5TTS.from_pretrained', return_value=fake), \
             patch('text_normalization.normalize_text') as normalizer:
            audio, info = self.backend.synthesize(*self.inputs)
        normalizer.assert_not_called()
        self.assertEqual(fake.infer.call_args.args[0], self.inputs[2])
        self.assertEqual(fake.infer.call_args.kwargs['prompt_text'], REFERENCE_TEXT)
        self.assertNotIn('tn', fake.infer.call_args.kwargs)
        self.assertNotIn('tn_requested', info)
        self.assertEqual(audio[0],24000)

    def test_default_normalizes_once_inside_inference(self):
        generator = Mock(sample_rate=24000, device=torch.device('cpu'))
        generator.generate.return_value = torch.ones(4000) * .1
        self.backend._model = M5TTS(generator)
        with patch('inference.prepare_prompt', return_value=torch.ones(1000)), \
             patch('text_normalization.normalize_text', wraps=normalize_text) as normalizer:
            self.backend.synthesize(*self.inputs)
        normalizer.assert_called_once_with(self.inputs[2], lang='Chinese', accent=None)
        segment = generator.generate.call_args.kwargs['context'][0]
        self.assertEqual(segment.text, REFERENCE_TEXT + '这件衣服九十九点五元。')

    def test_tn_failure_does_not_generate(self):
        generator = Mock(sample_rate=24000)
        self.backend._model = M5TTS(generator)
        with patch('text_normalization.normalize_text', side_effect=ValueError('TN failed')):
            with self.assertRaisesRegex(gr.Error, 'TN failed'):
                self.backend.synthesize(*self.inputs)
        generator.generate.assert_not_called()

    def test_multilingual_normalizes_once_inside_inference(self):
        cases = [('ja', 'Japanese', 'りんごを12個買いました。', 'りんごを十二個買いました。'),
                 ('ko', 'Korean', '가격은 12000원입니다.', '가격은 만이천 원입니다.'),
                 ('de', 'German', 'Ich habe 12 Äpfel gekauft.', 'Ich habe zwölf Äpfel gekauft.'),
                 ('fr', 'French', 'J’ai acheté 12 pommes.', 'J’ai acheté douze pommes.')]
        generator = Mock(sample_rate=24000, device=torch.device('cpu'))
        generator.generate.return_value = torch.ones(4000) * .1
        self.backend._model = M5TTS(generator)
        for code, lang, text, expected in cases:
            self.inputs[2:4] = [text, code]
            with self.subTest(lang=lang), patch('inference.prepare_prompt', return_value=torch.ones(1000)), \
                 patch('text_normalization.normalize_text', wraps=normalize_text) as normalizer:
                self.backend.synthesize(*self.inputs)
                normalizer.assert_called_once_with(text, lang=lang, accent=None)
                segment = generator.generate.call_args.kwargs['context'][0]
                self.assertEqual(segment.text, REFERENCE_TEXT + expected)
                self.assertEqual(segment.lang, lang)

    def test_web_has_no_tn_checkbox_or_callback_argument(self):
        with patch('app.M5TTS.from_pretrained') as loader:
            demo=create_demo(DemoSettings(device='cpu',attention='sdpa'))
            config=demo.get_config_file()
            self.assertFalse(any(c['type']=='checkbox' and c['props'].get('label','').startswith('Text normalization')
                                 for c in config['components']))
            callback=next(d for d in config['dependencies'] if d.get('api_name')=='synthesize')
            self.assertEqual(len(callback['inputs']),8)
            loader.assert_not_called()
            demo.close()


if __name__=='__main__':
    unittest.main()
