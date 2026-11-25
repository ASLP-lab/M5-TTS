"""Validate the char_exp recipe and its frame-alignment contract."""
import math


def settings(conf):
    return conf['token_cer']


def scalar_weights(conf):
    return conf['reward_weight']


def _metadata(alignment, meta):
    if not isinstance(alignment, dict) or not isinstance(meta, dict):
        raise ValueError('Pure token CER requires localization and generation metadata; restart updated actors')
    if alignment.get('backend') != 'qwen3_asr':
        raise ValueError('Pure token CER requires the Qwen transcription/alignment backend')
    n, total = meta.get('num_frames'), meta.get('num_decisions')
    eos = meta.get('has_eos_decision')
    rate, stride = meta.get('sample_rate'), meta.get('frame_samples')
    if any(type(x) is not int or x <= 0 for x in [n, total, rate, stride]) or type(eos) is not bool:
        raise ValueError('Invalid generation decision metadata')
    if total != n + int(eos) or meta.get('stop_reason') not in ['eos', 'max_frames']:
        raise ValueError('Decision count or stopping reason mismatch')
    if eos != (meta['stop_reason'] == 'eos'):
        raise ValueError('Only sampled EOS may create an EOS decision')
    context = alignment.get('frame_context')
    if not isinstance(context, dict) or any(context.get(k) != meta[k] for k in ['num_frames', 'sample_rate', 'frame_samples']):
        raise ValueError('Localization and generated frame clocks differ')
    frame_ms = 1000.0 * stride / rate
    duration = context.get('observed_duration_ms')
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 0 <= duration <= n * frame_ms + 1000 / rate:
        raise ValueError('Invalid observed duration')
    return n, total, eos, frame_ms, duration
