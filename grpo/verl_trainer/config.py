"""Paths and fixed defaults for the single synchronous GRPO recipe."""
import json
import math
from pathlib import Path

import yaml


WORLD_SIZE = 8
VIRTUAL_STREAMS = 2
PROMPTS_PER_STREAM = 16


def load_config(path):
    source = Path(path).expanduser().resolve()
    config = yaml.safe_load(source.read_text(encoding='utf-8'))

    def resolve(section, key, kind=None):
        value = config[section][key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{section}.{key} must be a path')
        result = (source.parent / Path(value).expanduser()).resolve()
        if kind == 'file' and not result.is_file():
            raise FileNotFoundError(result)
        if kind == 'dir' and not result.is_dir():
            raise FileNotFoundError(result)
        config[section][key] = str(result)
        return result

    checkpoint = resolve('m5_model', 'checkpoint')
    if checkpoint.is_dir():
        checkpoint /= 'model.safetensors'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config['m5_model']['checkpoint'] = str(checkpoint)
    resolve('m5_model', 'codec_path', 'dir')
    for key in ('asr_model', 'aligner_model', 's3prl_checkout'):
        resolve('assets', key, 'dir')
    for key in ('wavlm_checkpoint', 'sim_checkpoint'):
        resolve('assets', key, 'file')
    resolve('verl', 'prompt_file', 'file')
    resolve('verl', 'run_root')
    resolve('train_conf', 'checkpoint_root')
    resolve('train_conf', 'reference_pool', 'file')

    conf = config['train_conf']
    weights = conf['reward_weight']
    if set(weights) != {'cer_reward', 'sim_reward', 'pause_reward'} or weights['cer_reward'] != 0:
        raise ValueError('Expected scalar CER weight 0 and SIM/Pause weights')
    token = conf['token_cer']
    if set(token) != {'weight'}:
        raise ValueError('token_cer accepts only weight in the char_exp recipe')
    for weight in (weights['sim_reward'], weights['pause_reward'], token['weight']):
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
            raise ValueError('Reward weights must be finite and non-negative')
    for key in ('final_step', 'save_per_step', 'data_repeats', 'shuffle_buffer'):
        if type(conf[key]) is not int or conf[key] < 1:
            raise ValueError(f'train_conf.{key} must be a positive integer')
    conf.update(
        dtype='bf16', num_answers_per_question=8, rollout_micro_batch_size=8,
        resample_max_attempts=3, clip_eps_low=0.2, clip_eps_high=0.3,
        dynamic_clip_eps_low=dict(start=0.2, end=0.3, final_step=100000),
        dynamic_clip_eps_high=dict(start=0.3, end=0.5, final_step=100000),
        kl_weight=0.01, decoder_fraction=0.125, grad_clip=1.0,
        generation_conf=dict(temperature=1.0, topk=50, max_audio_length_ms=40000, max_seq_len=2048),
    )
    token.update(mode='char_exp', enabled=True, ambiguous='use_path', advantage_clip=None)
    port = config['verl'].get('reward_port_base', 19881)
    if type(port) is not int or not 1024 <= port <= 65536 - WORLD_SIZE:
        raise ValueError('verl.reward_port_base must leave space for eight local ports')
    config['verl']['reward_port_base'] = port
    return config


def read_reference_pool(path):
    path = Path(path)
    references = json.loads(path.read_text(encoding='utf-8'))
    if not references or len(references) % VIRTUAL_STREAMS:
        raise ValueError('Reference pool must have a positive even number of entries')
    if len({record['speaker_id'] for record in references}) != len(references):
        raise ValueError('Reference pool must have one entry per speaker')
    for record in references:
        record['wav_path'] = str(path.parent / record['wav_path'])
        if not record['prompt_text'].strip():
            raise ValueError('A reference waveform needs its transcript in prompt_text')
    return references
