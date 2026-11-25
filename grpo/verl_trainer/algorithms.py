"""Register the M5 GRPO objective with verl."""
from collections import defaultdict

import numpy as np
from verl.trainer.ppo.core_algos import register_adv_est, register_policy_loss

from m5_tts.grpo_utils import prepare_rollout_advantages
from m5_tts.token_cer_char_exp import apply_character_cer

from .colocated_loss import batch_forward_m5


@register_adv_est('m5_char_exp')
def m5_advantage(episodes, conf):
    normalized, signal = prepare_rollout_advantages(episodes, conf)
    prepared, metrics = apply_character_cer(normalized, conf)
    return prepared, metrics, signal


register_policy_loss('m5_c0_depth_reference')(batch_forward_m5)


def schedule_clip(conf, step):
    for side in ('low', 'high'):
        schedule = conf['dynamic_clip_eps_' + side]
        fraction = min(step / schedule['final_step'], 1.0)
        conf['clip_eps_' + side] = schedule['start'] + (schedule['end'] - schedule['start']) * fraction


def reward_metrics(episodes):
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode.group_token_ids].append(episode)

    result = {}
    for key in ('cer_reward', 'sim_reward', 'pause_reward'):
        values = [episode.reward_info[key] for episode in episodes]
        result[key] = np.mean(values)
        result[key + '_std'] = np.std(values)
        result[key + '_group_var'] = np.mean([
            np.var([episode.reward_info[key] for episode in group]) for group in groups.values()
        ])
    return result
