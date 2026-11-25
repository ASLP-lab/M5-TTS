"""Real verl BaseEngine; PyTorch FSDP2, FP32 master weights, no DeepSpeed."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import json
import os
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.checkpoint.state_dict import (
    get_state_dict, set_state_dict, StateDictOptions)
from safetensors.torch import save_file
from verl.workers.engine.base import BaseEngine, EngineRegistry
from verl.utils.fsdp_utils import fsdp2_clip_grad_norm_
from m5_tts.generator_m5 import load_m5_model
from m5_tts.telemetry import entropy_metrics
from .config import VIRTUAL_STREAMS


@EngineRegistry.register(model_type='m5_multicodebook', backend='m5_fsdp2', device='cuda')
class M5FSDP2Engine(BaseEngine):
    def __init__(self, config, run_root):
        self.config, self.conf = config, config['train_conf']
        self.device = torch.device('cuda', 0)
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.virtual_group = None
        self.virtual_subranks = 1
        virtual_size = VIRTUAL_STREAMS
        assert self.world % virtual_size == 0
        if virtual_size != self.world:
            self.virtual_subranks=self.world//virtual_size
            for start in range(0,self.world,self.virtual_subranks):
                ranks=list(range(start,start+self.virtual_subranks))
                group=dist.new_group(ranks)
                if self.rank in ranks:self.virtual_group=group
        self.step = 0

    @property
    def is_param_offload_enabled(self): return False
    @property
    def is_optimizer_offload_enabled(self): return False

    def initialize(self):
        model, self.tokenizer = load_m5_model(self.config['m5_model'], self.device)
        self.reference = deepcopy(model).eval().requires_grad_(False)
        # load_m5_model rounds to the original BF16 policy first. Native FSDP2
        # then maintains FP32 master parameters rather than BF16 Adam moments.
        model.float().train()
        self.model = fully_shard(model, mesh=init_device_mesh('cuda', (self.world,)),
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
            reshard_after_forward=False)
        opt = self.conf['optim_conf']
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(opt['lr']),
            betas=(0.9, 0.999), weight_decay=float(opt['weight_decay']))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)

    @contextmanager
    def train_mode(self, **kwargs):
        self.model.train()
        yield

    @contextmanager
    def eval_mode(self, **kwargs):
        previous = self.model.training
        self.model.eval()
        try: yield
        finally: self.model.train(previous)

    def optimizer_zero_grad(self): self.optimizer.zero_grad(set_to_none=True)

    def forward_backward_batch(self, data, loss_function, forward_only=False):
        episodes = data['episodes']
        conf = dict(self.conf, collect_policy_entropy=True)
        if self.virtual_group is not None:
            conf.update(_virtual_dp_group=self.virtual_group,_virtual_subranks=self.virtual_subranks)
        values = loss_function(self.model, self.reference, episodes, conf, self.device)
        loss = values.pop('loss')
        values.update(entropy_metrics(values, self.device))
        values['loss'] = float(loss.detach())
        if not forward_only:
            loss.backward()
        return {'metrics': values}

    def optimizer_step(self):
        # FSDP2 gradients are sharded DTensors; use verl's native global clipping.
        bad = torch.zeros((),device=self.device,dtype=torch.int64)
        for p in self.model.parameters():
            if p.grad is not None:
                g=p.grad.to_local() if isinstance(p.grad,DTensor) else p.grad
                bad += (~torch.isfinite(g)).sum()
        dist.all_reduce(bad)
        assert int(bad)==0, 'Non-finite policy gradient'
        norm = fsdp2_clip_grad_norm_(self.model.parameters(), float(self.conf['grad_clip']))
        self.optimizer.step();self.step += 1
        return float(norm)

    def lr_scheduler_step(self):
        self.scheduler.step();return self.scheduler.get_last_lr()[0]

    def get_per_tensor_param(self):
        def stream():
            for name,p in self.model.state_dict().items():
                full=p.full_tensor() if isinstance(p,DTensor) else p
                yield name,full.detach().to(dtype=torch.bfloat16,device='cpu').contiguous()
        return stream(), None

    def export_rollout(self, path):
        stream,_=self.get_per_tensor_param();state={k:v for k,v in stream}
        if self.rank==0:
            path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
            tmp=path.with_suffix('.tmp');save_file(state,str(tmp));os.replace(tmp,path)
        dist.barrier()
        return str(path)

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None, **kwargs):
        path = Path(local_path)
        if self.rank == 0:
            path.mkdir(parents=True, exist_ok=True)
            (path / 'complete.json').unlink(missing_ok=True)
        dist.barrier()
        options=StateDictOptions(full_state_dict=True,cpu_offload=True)
        model_state,optim_state=get_state_dict(self.model,self.optimizer,options=options)
        rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(self.device),
                 python=random.getstate(),numpy=np.random.get_state())
        torch.save(rng,path/f'rng_rank{self.rank}.pt')
        if self.rank==0:
            torch.save(dict(model=model_state,optimizer=optim_state,scheduler=self.scheduler.state_dict(),
                            step=self.step),path/'training_state.pt')
        self.export_rollout(path/'model.safetensors')
        if self.rank==0:
            marker=dict(step=self.step,train_conf=self.conf,world_size=self.world,
                initial_checkpoint=self.config['m5_model']['checkpoint'],engine='verl0.9.1/m5_fsdp2')
            tmp=path/'complete.tmp';tmp.write_text(json.dumps(marker,indent=2));os.replace(tmp,path/'complete.json')
        dist.barrier()

    def load_checkpoint(self, local_path, **kwargs):
        path=Path(local_path)
        state=torch.load(path/'training_state.pt',map_location='cpu',weights_only=False)
        set_state_dict(self.model,self.optimizer,model_state_dict=state['model'],
            optim_state_dict=state['optimizer'],options=StateDictOptions(full_state_dict=True))
        self.scheduler.load_state_dict(state['scheduler']);self.step=int(state['step'])
        rng = torch.load(path / f'rng_rank{self.rank}.pt', map_location='cpu', weights_only=False)
        torch.set_rng_state(rng['torch'])
        torch.cuda.set_rng_state(rng['cuda'], self.device)
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        dist.barrier()

    def get_data_parallel_size(self): return self.world
    def get_data_parallel_rank(self): return self.rank
    def get_data_parallel_group(self): return dist.group.WORLD
    def is_mp_src_rank_with_outputs(self): return True
