"""Run colocated synchronous M5 GRPO on one eight-GPU node."""
import argparse
from itertools import islice
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import traceback


from .config import PROMPTS_PER_STREAM, VIRTUAL_STREAMS, WORLD_SIZE, load_config, read_reference_pool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', help='Path to a complete step directory')
    parser.add_argument('--max-steps', type=int, help='Stop at this absolute step')
    args = parser.parse_args()
    config = load_config(args.config)
    conf = config['train_conf']
    references = read_reference_pool(conf['reference_pool'])
    resume = Path(args.resume).expanduser().resolve() if args.resume else None
    step = 0
    if resume:
        marker = json.loads((resume / 'complete.json').read_text(encoding='utf-8'))
        if marker['world_size'] != WORLD_SIZE:
            raise ValueError('Resume requires an eight-GPU checkpoint')
        if marker['initial_checkpoint'] != config['m5_model']['checkpoint']:
            raise ValueError('Resume requires the original initial checkpoint')
        step = marker['step']
    target = conf['final_step'] if args.max_steps is None else min(conf['final_step'], args.max_steps)
    if not 0 <= step < target:
        raise ValueError('The requested final step must be greater than the resume step')

    for key, value in {
        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'TOKENIZERS_PARALLELISM': 'false', 'VERL_USE_EXTERNAL_PLUGINS': 'none',
        'RAY_ENABLE_UV_RUN_RUNTIME_ENV': '0',
    }.items():
        os.environ.setdefault(key, value)
    with TemporaryDirectory(prefix='m5-ray-') as ray_dir:
        train(config, references, resume, step, target, ray_dir)


def train(config, references, resume, step, target, ray_dir):
    import ray
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

    from m5_tts.data_loader import PromptDataset

    from .hybrid import M5HybridWorker
    from .protocol import packet

    if torch.cuda.device_count() != WORLD_SIZE:
        raise ValueError('Expose exactly eight CUDA devices for this recipe')
    conf = config['train_conf']
    root = Path(config['verl']['run_root'])
    ckroot = Path(conf['checkpoint_root'])
    for directory in (root / 'logs', ckroot):
        directory.mkdir(parents=True, exist_ok=True)
    state_path = root / 'state.json'
    if state_path.exists() and not resume:
        raise FileExistsError(f'Choose a fresh run directory or resume from a checkpoint: {root}')
    state = dict(status='starting', step=step, target_step=target, world_size=WORLD_SIZE)

    def publish(**changes):
        state.update(changes, updated=time.time())
        temp = state_path.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temp.replace(state_path)
        print(json.dumps(changes), flush=True)

    publish(status='loading_workers')
    writer = SummaryWriter(str(root / 'tensorboard'))
    group = None
    try:
        ray.init(address='local', num_gpus=WORLD_SIZE, num_cpus=24, include_dashboard=False,
                 _temp_dir=ray_dir, object_store_memory=2 * 1024**3)
        pool = RayResourcePool(process_on_nodes=[WORLD_SIZE], use_gpu=True, max_colocate_count=1,
                               name_prefix='m5-grpo')
        group = RayWorkerGroup(
            resource_pool=pool, name_prefix='m5-grpo',
            ray_cls_with_init=RayClassWithInitArgs(M5HybridWorker, config, str(root), references[0]),
        )
        workers = group.ready()
        if len({worker['gpu'] for worker in workers}) != WORLD_SIZE:
            raise RuntimeError('Each worker must own a distinct GPU')
        if resume and any(restored != step for restored in group.load_checkpoint(str(resume))):
            raise RuntimeError('Checkpoint contained an inconsistent step')

        streams = [
            iter(PromptDataset([config['verl']['prompt_file']], rank, VIRTUAL_STREAMS,
                               seed=conf['seed'], shuffle_buffer=conf['shuffle_buffer'],
                               repeats=conf['data_repeats'], references=references))
            for rank in range(VIRTUAL_STREAMS)
        ]
        for stream in streams:
            for _ in range(step * PROMPTS_PER_STREAM):
                next(stream)

        first_step = step + 1
        last_saved = resume
        while step < target and not (root / 'STOP').exists():
            started = time.monotonic()
            publish(status='running', step=step, phase='rollout')
            records = [record for stream in streams for record in islice(stream, PROMPTS_PER_STREAM)]
            if len(records) != PROMPTS_PER_STREAM * VIRTUAL_STREAMS:
                raise ValueError('Prompt stream exhausted. Increase train_conf.data_repeats.')
            data = group.generate_sequences(packet('records', records, version=step, sequence=step)).get()
            update_started = time.monotonic()
            publish(phase='train')
            result = group.update_policy(data)
            rows = list(result.non_tensor_batch['metrics'])
            if any(row['policy_lag'] != 0 or row['step'] != step + 1 for row in rows):
                raise RuntimeError('Rollout and learner policy versions diverged')
            metrics = {key: sum(row[key] for row in rows) / len(rows)
                       for key in set.intersection(*(set(row) for row in rows))}
            step += 1
            metrics.update(step=step, step_seconds=time.monotonic() - started,
                           rollout_wait_seconds=update_started - started)
            for key, value in metrics.items():
                writer.add_scalar('train/' + key, value, step)
            writer.flush()
            with (root / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(metrics) + '\n')
            publish(step=step, last_metrics=metrics)

            if step == first_step or step % conf['save_per_step'] == 0 or step == target or (root / 'STOP').exists():
                last_saved = ckroot / f'step_{step:06d}'
                group.save_checkpoint(str(last_saved))
                publish(last_checkpoint=str(last_saved))

        final_path = ckroot / f'step_{step:06d}'
        if step >= first_step and last_saved != final_path:
            group.save_checkpoint(str(final_path))
            last_saved = final_path
        publish(status='stopped' if step < target else 'complete',
                last_checkpoint=str(last_saved) if last_saved else None)
    except BaseException as error:
        publish(status='failed', error=repr(error))
        raise
    finally:
        if group is not None:
            try:
                ray.get([worker.close.remote() for worker in group._workers], timeout=45)
            except Exception:
                traceback.print_exc()
        writer.close()
        ray.shutdown()


if __name__ == '__main__':
    main()
