"""Minimal 12Hz codec adapter; upstream model/license attribution is in NOTICE."""
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import EncodecFeatureExtractor

from .modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model


class Qwen3TTSTokenizer:
    """The encode/decode subset used by M5; no qwen-tts distribution required."""

    @classmethod
    def from_pretrained(cls, source, **kwargs):
        instance = cls()
        feature_kwargs = {k: kwargs[k] for k in ('local_files_only','cache_dir','revision','token','force_download') if k in kwargs}
        instance.feature_extractor = EncodecFeatureExtractor.from_pretrained(source, **feature_kwargs)
        instance.model = Qwen3TTSTokenizerV2Model.from_pretrained(source, **kwargs)
        instance.config = instance.model.config
        instance.device = instance.model.device
        if instance.config.model_type != 'qwen3_tts_tokenizer_12hz':
            raise ValueError('Only the Qwen 12Hz codec is supported')
        return instance

    def encode(self, audios, sr=None, return_dict=True):
        import librosa
        import soundfile as sf
        if isinstance(audios, (np.ndarray, str, Path)):
            audios = [audios]
        if not audios:
            raise ValueError('At least one reference waveform is required')
        target = int(self.feature_extractor.sampling_rate)
        waveforms = []
        for value in audios:
            rate = sr
            if isinstance(value, (str, Path)):
                value, rate = sf.read(value, dtype='float32')
            if not isinstance(value, np.ndarray) or rate is None:
                raise ValueError('NumPy waveforms require their source sample rate sr')
            if value.ndim > 1:
                value = np.mean(value, axis=-1)
            if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
                raise ValueError('Expected a nonempty finite waveform')
            if int(rate) != target:
                value = librosa.resample(y=value.astype(np.float32), orig_sr=int(rate), target_sr=target)
            waveforms.append(value.astype(np.float32))
        inputs = self.feature_extractor(raw_audio=waveforms, sampling_rate=target, return_tensors='pt')
        inputs = inputs.to(self.device).to(self.model.dtype)
        with torch.inference_mode():
            return self.model.encode(inputs['input_values'].squeeze(1), inputs['padding_mask'].squeeze(1), return_dict=return_dict)

    def decode(self, encoded):
        if hasattr(encoded, 'audio_codes'):
            codes = encoded.audio_codes
        elif isinstance(encoded, dict):
            codes = encoded['audio_codes']
        elif isinstance(encoded, list) and encoded:
            codes = [row['audio_codes'] for row in encoded]
        else:
            raise TypeError('Expected encode output, audio_codes dict, or list of dicts')
        if isinstance(codes, torch.Tensor):
            codes = codes.unsqueeze(0) if codes.ndim == 2 else codes
        else:
            codes = pad_sequence([c if isinstance(c,torch.Tensor) else torch.from_numpy(np.asarray(c)).long()
                                  for c in codes], batch_first=True, padding_value=-1)
        if codes.ndim != 3 or codes.shape[-1] != 16:
            raise ValueError('Expected codes shaped (batch, frames, 16)')
        with torch.inference_mode():
            values = self.model.decode(codes.to(self.device), return_dict=True).audio_values
        return [wave.to(torch.float32).detach().cpu().numpy() for wave in values], int(self.model.get_output_sample_rate())
