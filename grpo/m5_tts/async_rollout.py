"""Collect synchronous rollout groups with deterministic resampling."""
import time
import torch
from .grpo_utils import prepare_rollout_advantages
from .performance import ROLLOUT_SUM_KEYS


def collect_rollout(adapter, records, reward_client, conf, rank, sequence):
    started = time.monotonic()
    aggregate = {key: 0.0 for key in ROLLOUT_SUM_KEYS}
    for attempt in range(int(conf.get('resample_max_attempts', 3))):
        seed = (int(conf['seed']) + 1000003 * sequence + 101 * attempt + rank) % (2**63 - 1)
        rng = torch.Generator(device=adapter.device).manual_seed(seed)
        episodes = adapter.rollout(records, int(conf['num_answers_per_question']),
                                   reward_client, conf, rank, sequence, rng=rng)
        for key in aggregate:
            aggregate[key] += adapter.last_rollout_metrics[key]
        normalized, has_signal = prepare_rollout_advantages(episodes, conf)
        if has_signal:
            break
    if not normalized or not sum(len(ep.generated_token_ids) for ep in normalized):
        raise ValueError('The local rollout batch has no generated frames')
    aggregate['rollout_seconds'] = time.monotonic() - started
    aggregate['rollout_micro_batch_size'] = min(int(conf.get('rollout_micro_batch_size', 1)), int(conf['num_answers_per_question']))
    return episodes, aggregate
