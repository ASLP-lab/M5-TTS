"""Waveform rollout and local reward service for each colocated GPU."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import requests
import torch
from verl.workers.rollout.base import BaseRollout

from m5_tts.async_rollout import collect_rollout
from m5_tts.model_adapter import M5Adapter
from m5_tts.reward_func import RewardClient

from .protocol import packet


class M5WaveformRollout(BaseRollout):
    def __init__(self, config, rank, url, run_root):
        self.conf = config['train_conf']
        self.rank = rank
        self.version = None
        self.adapter = M5Adapter(config['m5_model'], torch.device('cuda', 0))
        self.client = RewardClient(url, Path(run_root) / 'reward_audio' / str(rank))

    async def resume(self, tags):
        pass

    async def release(self):
        self.client.close()

    async def update_weights(self, weights, **kwargs):
        self.adapter.model.load_state_dict(dict(weights), strict=True)
        torch.cuda.synchronize()

    def generate_sequences(self, data):
        version = int(data.meta_info['version'])
        sequence = int(data.meta_info['sequence'])
        if version != self.version:
            raise RuntimeError('Rollout weights are out of sync')
        records = list(data.non_tensor_batch['records'])
        episodes, metrics = collect_rollout(self.adapter, records, self.client, self.conf, self.rank, sequence)
        groups = [[] for _ in records]
        for episode in episodes:
            groups[episode.prompt_index].append(episode)
        if any(len(group) != self.conf['num_answers_per_question'] for group in groups):
            raise RuntimeError('Incomplete GRPO rollout group')
        result = packet('groups', groups, version=version, sequence=sequence)
        for key, value in metrics.items():
            result.batch['rollout/' + key] = torch.full((len(records),), float(value) / len(records))
        return result


class RewardService:
    def __init__(self, assets, port, log, reference):
        self.url = f'http://127.0.0.1:{port}'
        python_path = assets['s3prl_checkout']
        if os.environ.get('PYTHONPATH'):
            python_path += os.pathsep + os.environ['PYTHONPATH']
        env = dict(
            os.environ,
            LOCAL_RANK='0', RANK='0', WORLD_SIZE='1',
            M5_GRPO_ASR_MODEL=assets['asr_model'],
            M5_GRPO_ALIGNER_MODEL=assets['aligner_model'],
            M5_GRPO_WAVLM_CHECKPOINT=assets['wavlm_checkpoint'],
            M5_GRPO_SIM_CHECKPOINT=assets['sim_checkpoint'],
            PYTHONPATH=python_path,
            TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
        )
        with log.open('ab', buffering=0) as stream:
            self.process = subprocess.Popen(
                [sys.executable, '-m', 'uvicorn', 'm5_tts.reward_server:app',
                 '--host', '127.0.0.1', '--port', str(port)],
                env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        try:
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f'Reward service exited. See {log}')
                try:
                    if requests.get(self.url + '/health', timeout=2).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError(f'Reward startup timed out. See {log}')

            response = requests.post(
                self.url + '/reward',
                json=dict(audio_path=reference['wav_path'], target_audio=reference['wav_path'],
                          uttid='warmup', ref_text=reference['prompt_text'],
                          language=reference.get('lang') or 'Chinese'),
                timeout=600,
            )
            response.raise_for_status()
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process.poll() is not None:
            return
        os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()
