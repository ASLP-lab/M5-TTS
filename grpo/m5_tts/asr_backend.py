"""Frozen ASR adapters. Imports do not load weights or start CUDA work."""
from pathlib import Path
from threading import Lock
from .reward_language import canonical_language


class QwenASRBackend:
    """Adapt Qwen transcription + independent alignment to the reward API.

    No reference/target text is passed to ASR or its timestamp aligner.
    Qwen times are seconds; the public adapter contract uses milliseconds.
    The shared aligner lock also protects the optional pause-reward path.
    """
    def __init__(self, model_path, aligner_path, device, max_batch_size=1,
                 max_new_tokens=2048, language='Chinese', aligner_lock=None):
        import torch
        from qwen_asr import Qwen3ASRModel
        self.model_path, self.aligner_path = str(model_path), str(aligner_path)
        self.language = language
        self.aligner_lock = aligner_lock or Lock()
        dtype = torch.float32 if str(device) == 'cpu' else torch.bfloat16
        self.asr = Qwen3ASRModel.from_pretrained(
            self.model_path, dtype=dtype, device_map=str(device),
            attn_implementation='sdpa', max_inference_batch_size=max_batch_size,
            max_new_tokens=max_new_tokens, forced_aligner=self.aligner_path,
            forced_aligner_kwargs=dict(dtype=dtype, device_map=str(device),
                                       attn_implementation='sdpa'))
        self.asr.model.eval().requires_grad_(False)
        self.asr.forced_aligner.model.eval().requires_grad_(False)

    @property
    def forced_aligner(self):
        return self.asr.forced_aligner

    def generate(self, input, language=None, **_compat_kwargs):
        paths = [input] if isinstance(input, (str, Path)) else list(input)
        langs = language if isinstance(language, list) else [language or self.language] * len(paths)
        if len(langs) != len(paths):
            raise ValueError('Language batch mismatch')
        langs = [canonical_language(value) for value in langs]
        # Split the two official calls so pause alignment shares a single lock
        # without serializing unrelated ASR decoding with that model.
        results = self.asr.transcribe(audio=[str(x) for x in paths], context='',
                                      language=langs, return_time_stamps=False)
        if len(results) != len(paths):
            raise ValueError('ASR returned a different batch length')
        aligned = [None] * len(results)
        nonempty = [i for i, r in enumerate(results) if r.text.strip()]
        if nonempty:
            with self.aligner_lock:
                values = self.forced_aligner.align(
                    audio=[str(paths[i]) for i in nonempty],
                    text=[results[i].text for i in nonempty], language=[langs[i] for i in nonempty])
            if len(values) != len(nonempty):
                raise ValueError('Aligner returned a different batch length')
            for i, value in zip(nonempty, values):
                aligned[i] = value
        output = []
        for path, result, alignment in zip(paths, results, aligned):
            units = [] if alignment is None else [
                {'text': str(u.text), 'start_ms': float(u.start_time) * 1000,
                 'end_ms': float(u.end_time) * 1000} for u in alignment]
            output.append({'key': str(path), 'text': result.text,
                           'language': result.language,
                           'words': [u['text'] for u in units],
                           'timestamp': [[u['start_ms'], u['end_ms']] for u in units],
                           'aligned_units': units, 'timestamp_unit': 'ms',
                           'backend': 'qwen3_asr', 'model': self.model_path,
                           'aligner_model': self.aligner_path})
        return output
