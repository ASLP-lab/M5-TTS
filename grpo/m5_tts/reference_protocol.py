"""Native M5 reference-continuation protocol, shared by rollout and Seed eval."""
from dataclasses import replace
import numpy as np
import soundfile as sf
import torch
import torchaudio


def remove_reference_silence(audio, threshold=.015, min_silence_duration=.15, sample_rate=24000):
    # Keep the original infer_m5ablation preprocessing, including its 50 ms buffer.
    values = audio.detach().cpu().numpy()
    speech = np.abs(values) > threshold
    silence_samples = int(min_silence_duration * sample_rate)
    segments, active, start = [], False, 0
    for i in range(len(speech)):
        if speech[i] and not active:
            active, start = True, i
        elif not speech[i] and active:
            end = min(len(speech), i + silence_samples)
            if end - i == silence_samples and not speech[i:end].any():
                active = False
                segments.append((start, i))
    if active:
        segments.append((start, len(speech)))
    if not segments:
        return audio
    buffer = int(.05 * sample_rate)
    joined = np.concatenate([values[max(0, a-buffer):min(len(values), b+buffer)] for a, b in segments])
    return torch.from_numpy(joined.copy()).to(audio.device)


def load_reference_audio(path, sample_rate):
    values, rate = sf.read(path, dtype='float32', always_2d=True)
    audio = torch.from_numpy(values.mean(axis=1).copy())
    if not audio.numel() or not bool(torch.isfinite(audio).all()):
        raise ValueError('Empty/non-finite reference audio: ' + str(path))
    if rate != sample_rate:
        audio = torchaudio.functional.resample(audio, rate, sample_rate)
    audio = audio / (audio.abs().max() + 1e-8)
    threshold = .02 if audio.numel() / sample_rate > 10 else .015
    return remove_reference_silence(audio, threshold=threshold, sample_rate=sample_rate)


def continuation_context(reference, target):
    """One text block, followed by its partial audio. Never mutate a cached ref."""
    if not (reference.text or '').strip() or not (target.text or '').strip():
        raise ValueError('Reference continuation requires reference and target text')
    if reference.audio is None and reference.audio_codes is None:
        raise ValueError('Reference continuation requires reference audio or cached codes')
    if (reference.speaker, reference.accent, reference.lang, reference.caption) != (
            target.speaker, target.accent, target.lang, target.caption):
        raise ValueError('Reference and target conditioning must match')
    return [replace(reference, text=reference.text.strip() + target.text.strip())]
