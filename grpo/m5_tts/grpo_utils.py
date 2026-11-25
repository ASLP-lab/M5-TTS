# Copyright (c) 2025 Zhipu AI Inc (authors: CogAudio Group Members)
# Modified for the M5 colocated verl GRPO recipe.
# Licensed under the Apache License, Version 2.0.
"""Group-wise outcome normalization and fixed-reference KL."""
import dataclasses
from collections import defaultdict
from typing import List
import numpy as np
import torch
from .data_types import Episode

def prepare_rollout_advantages(episodes: List[Episode], conf: dict):
    from .token_cer_char_exp import attach_group_advantages

    weights = conf['reward_weight']
    if any(weights.keys() - ep.reward_info.keys() for ep in episodes):
        raise ValueError('Rollout is missing a reward component')
    normalized, scalar_signal = _normalize_group_rewards(episodes, weights)
    normalized, token_signal = attach_group_advantages(normalized, conf)
    return normalized, scalar_signal or token_signal


def _normalize_group_rewards(episodes, reward_weights):
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode.group_token_ids].append(episode)

    output = []
    has_grad = True
    for group in groups.values():
        reward_keys = list(group[0].reward_info.keys())
        reward_array = {k: np.array([ep.reward_info[k] for ep in group]) for k in reward_keys}
        normed_rewards = {}

        for k in reward_keys:
            arr = reward_array[k]
            mean = arr.mean()
            std = arr.std()
            if std < 1e-2:
                normed_rewards[k] = [0.0 for v in arr]
            else:
                normed_rewards[k] = [(v - mean) / std for v in arr]

        summed = [
            sum(normed_rewards[key][i] * reward_weights[key] for key in reward_keys)
            for i in range(len(group))
        ]
        mean_summed = np.mean(summed)
        std_summed = np.std(summed)

        for idx, episode in enumerate(group):
            if std_summed < 1e-2:
                summed_norm = 0.0
                has_grad = False
            else:
                summed_norm = (summed[idx] - mean_summed) / std_summed

            if np.isscalar(summed_norm):
                summed_norm = [summed_norm] * len(episode.generated_token_ids)
            new_reward_info = episode.reward_info.copy()
            for k in reward_keys:
                new_reward_info[k] = normed_rewards[k][idx]
            episode_new = dataclasses.replace(episode, reward=summed_norm, reward_info=new_reward_info)
            output.append(episode_new)
    return output, has_grad


def compute_kl_loss(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_estimator: str = "k3",
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html
    """
    if kl_estimator == "k1":
        log_ratio = log_probs.float() - log_probs_base.float()
    elif kl_estimator == "k2":
        log_ratio = log_probs.float() - log_probs_base.float()
        log_ratio = log_ratio ** 2 / 2.0
    elif kl_estimator == "k3":
        log_ratio = log_probs.float() - log_probs_base.float()
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio
    else:
        raise ValueError(f"Unknown kl_estimator: {kl_estimator}")

    log_ratio = log_ratio.clamp(min=-10, max=10)
    return log_ratio
