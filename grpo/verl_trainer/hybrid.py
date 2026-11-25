"""Each GPU runs one verl worker with synchronized training and rollout phases."""
import asyncio
from datetime import timedelta
from importlib import import_module
import os
from pathlib import Path
import random
import time

import numpy as np
import ray
import torch
import torch.distributed as dist
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.trainer.ppo.core_algos import get_adv_estimator_fn, get_policy_loss_fn
from verl.workers.engine.base import EngineRegistry

from .algorithms import reward_metrics, schedule_clip
from .protocol import groups_to_episodes, packet
from .workers import M5WaveformRollout, RewardService


@ray.remote
class M5HybridWorker(Worker):
    def __init__(self, config, run_root, reference):
        super().__init__()
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        seed = config['train_conf']['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        dist.init_process_group('nccl', timeout=timedelta(minutes=30), device_id=torch.device('cuda', 0))
        # Each Ray worker needs its own verl engine registration.
        import_module('.engine', __package__)

        self.engine = EngineRegistry.new('m5_multicodebook', 'm5_fsdp2', config, run_root)
        self.engine.initialize()
        self.reward = RewardService(
            config['assets'], config['verl']['reward_port_base'] + self.rank,
            Path(run_root) / 'logs' / f'reward_{self.rank}.log', reference,
        )
        try:
            self.rollout = M5WaveformRollout(config, self.rank, self.reward.url, run_root)
        except BaseException:
            self.reward.close()
            raise

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def ready(self):
        return dict(rank=self.rank, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), step=self.engine.step)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, blocking=False)
    def generate_sequences(self, data):
        step = self.engine.step
        if int(data.meta_info['version']) != step:
            raise RuntimeError('Stale rollout request')
        if self.rollout.version != step:
            weights, _ = self.engine.get_per_tensor_param()
            asyncio.run(self.rollout.update_weights(weights))
            self.rollout.version = step
        with self.engine.eval_mode():
            return self.rollout.generate_sequences(data)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_policy(self, data):
        started = time.monotonic()
        version = int(data.meta_info['version'])
        if version != self.engine.step:
            raise RuntimeError('Rollout policy version differs from learner')
        conf = self.engine.conf
        schedule_clip(conf, self.engine.step)
        raw = groups_to_episodes(data)
        prepared, metrics, signal = get_adv_estimator_fn('m5_char_exp')(raw, conf)
        with self.engine.train_mode():
            result = self.engine.train_batch(
                {'episodes': prepared}, get_policy_loss_fn('m5_c0_depth_reference'))
        metrics.update(result['metrics'])
        metrics.update(reward_metrics(raw))
        metrics.update({
            key: data.batch[key].sum().item() for key in data.batch.keys() if key.startswith('rollout/')
        })
        metrics.update(step=self.engine.step, has_signal=float(signal), policy_lag=self.engine.step - 1 - version,
                       lr=self.engine.lr_scheduler_step(), learner_update_seconds=time.monotonic() - started)
        return packet('metrics', [{key: float(value) for key, value in metrics.items()}])

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, path):
        self.engine.save_checkpoint(path, global_step=self.engine.step)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path):
        self.engine.load_checkpoint(path)
        return self.engine.step

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def close(self):
        try:
            asyncio.run(self.rollout.release())
        finally:
            self.reward.close()
            dist.destroy_process_group()
