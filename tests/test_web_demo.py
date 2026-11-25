import unittest
from unittest.mock import Mock, patch

import gradio as gr
import numpy as np
import torch

from app import DemoBackend, DemoSettings, REFERENCE_AUDIO, REFERENCE_TEXT, validate_inputs


class WebDemoTests(unittest.TestCase):
    def setUp(self):
        self.inputs = [str(REFERENCE_AUDIO), REFERENCE_TEXT, "你好，这是网页测试。",
                       "zh", "None", 0.9, 10, 42]
        self.backend = DemoBackend(DemoSettings(device="cpu", attention="sdpa"))

    def test_default_reference_is_valid(self):
        validate_inputs(*self.inputs)

    def test_invalid_inputs_do_not_load_model(self):
        cases = [(0, None), (1, ""), (1, "你" * 501), (2, ""), (2, "你" * 251),
                 (3, "invalid"), (4, "invalid"), (5, float("nan")), (5, 0),
                 (6, 1.5), (6, 51), (7, -1), (7, float("inf")), (7, True)]
        with patch("app.M5TTS.from_pretrained") as loader:
            for index, value in cases:
                with self.subTest(index=index, value=value):
                    inputs = self.inputs.copy()
                    inputs[index] = value
                    with self.assertRaises(gr.Error):
                        self.backend.synthesize(*inputs)
            loader.assert_not_called()

    def test_invalid_recording_is_rejected(self):
        with patch("app.sf.info", side_effect=RuntimeError("bad recording")):
            with self.assertRaisesRegex(gr.Error, "Cannot read"):
                validate_inputs(*self.inputs)
        for duration, channels, samplerate in [(31, 1, 24000), (0.1, 1, 24000),
                                               (2, 3, 24000), (2, 1, 384000)]:
            info = Mock(duration=duration, channels=channels, samplerate=samplerate)
            with self.subTest(duration=duration, channels=channels, samplerate=samplerate):
                with patch("app.sf.info", return_value=info):
                    with self.assertRaises(gr.Error):
                        validate_inputs(*self.inputs)

    def test_loading_reuse_seed_and_parameter_forwarding(self):
        model = Mock(sample_rate=24000)
        model.infer.side_effect = lambda *args, **kwargs: torch.rand(2400) - 0.5
        rng = torch.random.get_rng_state().clone()
        with patch("app.M5TTS.from_pretrained", return_value=model) as loader:
            first, metadata = self.backend.synthesize(*self.inputs)
            second, _ = self.backend.synthesize(*self.inputs)
            loader.assert_called_once()
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[1].dtype, np.int16)
        self.assertEqual(first[0], 24000)
        self.assertEqual(metadata["seed"], 42)
        self.assertEqual(metadata["audio_duration_seconds"], 0.1)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(model.infer.call_args.kwargs["lang"], "Chinese")
        self.assertIsNone(model.infer.call_args.kwargs["accent"])
        self.assertEqual(model.infer.call_args.kwargs["prompt_text"], REFERENCE_TEXT)
        self.assertEqual(model.infer.call_args.kwargs["max_audio_length_ms"], 40000)

    def test_load_failure_can_be_retried(self):
        with patch("app.M5TTS.from_pretrained", side_effect=RuntimeError("offline")):
            with self.assertLogs("app", level="ERROR"):
                with self.assertRaisesRegex(gr.Error, "Model loading failed"):
                    self.backend.synthesize(*self.inputs)
        self.assertIsNone(self.backend._model)
        self.assertFalse(self.backend._lock.locked())


if __name__ == "__main__":
    unittest.main()
