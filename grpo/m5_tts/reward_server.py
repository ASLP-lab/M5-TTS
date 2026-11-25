# Copyright (c) 2025 Zhipu AI Inc (authors: CogAudio Group Members)
# Modified for the M5 colocated verl GRPO recipe.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Loopback service for batched Qwen ASR, global SIM and pause scoring."""
import logging
import faulthandler
import os
import time
from threading import Lock

import numpy as np
import torch
import torchaudio
import zhconv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from .pause_reward_new import compute_char_level_pause_reward
from .performance import ServiceProfile, timed_stage
from .reward_batching import DynamicBatcher
from .asr_backend import QwenASRBackend
from .asr_localization import localize_errors
from .reward_language import canonical_language
from .run_sim import get_sim_model, verification2
from .run_wer import process_one


app = FastAPI()
profile = ServiceProfile()

@app.middleware('http')
async def arrival_time(request: Request, call_next):
    request.state.arrived_at = time.monotonic()
    return await call_next(request)

@app.get('/metrics')
async def metrics():
    # Health/metrics must not queue behind synchronous GPU reward handlers in
    # Starlette's shared thread pool. Snapshot locks protect only small CPU data.
    return dict(profile.snapshot(), batching={name: batcher.snapshot()
                for name, batcher in _batchers.items()})

@app.on_event('shutdown')
def close_profile():
    for batcher in _batchers.values():
        batcher.close()
    profile.close()

sample_rate = 24000
local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('WORKER_ID', '0')))
_asr_model = _sim_model = _pause_aligner = None
_models_loaded = False
_model_lock, asr_lock, sim_lock, pause_lock = (Lock() for _ in range(4))
_batchers = {}
REWARD_BATCH_SIZE = 8


def _asr_batch(paths):
    files, languages = zip(*paths)
    with torch.cuda.device(local_rank), torch.inference_mode():
        return _asr_model.generate(input=list(files), language=list(languages), batch_size=len(paths), batch_size_s=300)


def _align_batch(items):
    with torch.cuda.device(local_rank), torch.inference_mode():
        return _pause_aligner.align(audio=[item[0] for item in items],
                                    text=[item[1] for item in items], language=[item[2] for item in items])


class RewardRequest(BaseModel):
    audio_path: str
    uttid: str
    target_audio: str | None = None
    ref_text: str
    frame_metadata: dict | None = None
    language: str = 'Chinese'




def _ensure_models_loaded():
    global _asr_model, _sim_model, _pause_aligner, _models_loaded
    if _models_loaded:
        return
    with _model_lock:
        if _models_loaded:
            return
        faulthandler.dump_traceback_later(120, repeat=True)
        try:
            device = torch.device('cuda', local_rank)
            logging.warning('Loading Qwen ASR and aligner')
            _asr_model = QwenASRBackend(
                os.environ['M5_GRPO_ASR_MODEL'],
                os.environ['M5_GRPO_ALIGNER_MODEL'],
                device=device, max_batch_size=REWARD_BATCH_SIZE, aligner_lock=pause_lock,
            )
            _pause_aligner = _asr_model.forced_aligner
            logging.warning('Loading WavLM/ECAPA')
            _sim_model = get_sim_model(
                device, os.environ['M5_GRPO_SIM_CHECKPOINT'], os.environ['M5_GRPO_WAVLM_CHECKPOINT'])
            settings = dict(max_wait_seconds=0.03, max_queue_size=64, max_batch_seconds=160)
            _batchers['asr'] = DynamicBatcher(
                'asr', _asr_batch, max_batch_size=REWARD_BATCH_SIZE, model_lock=asr_lock, **settings)
            # A single alignment per batch preserves pause-threshold decisions.
            _batchers['pause_align'] = DynamicBatcher(
                'pause_align', _align_batch, max_batch_size=1, model_lock=pause_lock, **settings)
            _models_loaded = True
        finally:
            faulthandler.cancel_dump_traceback_later()
        logging.warning('All reward models loaded')


@app.get('/health')
async def health():
    return {'models_loaded': _models_loaded, 'asr_batch_size': REWARD_BATCH_SIZE,
            'aligner_batch_size': 1}


@app.post('/reward')
def get_reward(req: RewardRequest, request: Request):
    started = time.monotonic()
    arrived = getattr(request.state, 'arrived_at', started)
    timings = {'handler_queue_seconds': started - arrived}
    success = False
    pending = []
    profile.change('active_requests', 1)
    try:
        language = canonical_language(req.language)
        _ensure_models_loaded()
        tick = time.monotonic()
        response_audio, sr = torchaudio.load(req.audio_path)
        if sr != sample_rate:
            response_audio = torchaudio.functional.resample(response_audio, sr, sample_rate)
        if response_audio.shape[0] > 1:
            response_audio = response_audio.mean(dim=0, keepdim=True)

        timings['audio_load_seconds'] = time.monotonic() - tick
        duration = response_audio.shape[-1] / sample_rate
        asr_future = _batchers['asr'].submit((req.audio_path, language), duration)
        pending.append(asr_future)
        align_future = None
        if req.ref_text:
            align_future = _batchers['pause_align'].submit(
                (req.audio_path, req.ref_text, language), duration)
            pending.append(align_future)

        sim_reward = 0.0
        if req.target_audio and req.target_audio != 'None':
            tick = time.monotonic()
            target_audio, sr = torchaudio.load(req.target_audio)
            if sr != sample_rate:
                target_audio = torchaudio.functional.resample(target_audio, sr, sample_rate)
            if target_audio.shape[0] > 1:
                target_audio = target_audio.mean(dim=0, keepdim=True)
            timings['target_audio_load_seconds'] = time.monotonic() - tick
            # Match Cosy's guard and feed 24 kHz to verification2, which
            # performs its one and only 24 -> 16 kHz resampling internally.
            if min(response_audio.shape[-1], target_audio.shape[-1]) > int(sample_rate * .1):
                with timed_stage(timings, 'sim', sim_lock, profile):
                    sim_reward = float((1 + verification2(
                        response_audio, target_audio, _sim_model).item()) / 2)

        result = asr_future.result()
        timings.update(result.timings('asr'))
        asr_value = result.value
        asr_text = asr_value['text']
        transcription = zhconv.convert(asr_text, 'zh-cn') if language == 'Chinese' else asr_text
        _, _, cer, _, _, _ = process_one(transcription, req.ref_text, 'zh')
        cer_reward = float(np.exp(-2.5 * cer))
        tick = time.monotonic()
        asr_alignment = localize_errors(asr_value, req.ref_text,
            response_audio.shape[-1] / sample_rate * 1000, req.frame_metadata, language=language)
        if abs(asr_alignment['cer'] - cer) > 1e-12:
            raise ValueError('Localization changed the historical CER definition')
        timings['asr_localization_seconds'] = time.monotonic() - tick
        pause_reward = 0.0
        if align_future is not None:
            result = align_future.result()
            timings.update(result.timings('pause_align'))
            with timed_stage(timings, 'pause_score'):
                pause_reward, _ = compute_char_level_pause_reward(result.value, text=req.ref_text)
                pause_reward = float(pause_reward)

        rewards = {'cer_reward': cer_reward, 'sim_reward': sim_reward,
                   'pause_reward': pause_reward}
        if not all(np.isfinite(value) for value in rewards.values()):
            raise ValueError('Non-finite reward component')
        # The learner applies configured component weights; profiling does not
        # alter the scoring or normalization path.
        timings['server_seconds'] = time.monotonic() - started
        timings['server_wall_seconds'] = time.monotonic() - arrived
        success = True
        return {'reward': sum(rewards.values()),
                'reward_info': rewards, 'timings': timings,
                'asr_alignment': asr_alignment}
    except Exception as exc:
        logging.exception('Reward computation failed for %s', req.uttid)
        # Infrastructure failure is not a low-quality training example.
        raise HTTPException(status_code=503, detail='Reward computation failed') from exc
    finally:
        # A sibling stage may still be reading the shared WAV after an error.
        # Finish accepted work before responding and letting the client unlink it.
        for future in pending:
            try:
                future.result()
            except Exception:
                pass
        timings['server_seconds'] = time.monotonic() - started
        timings['server_wall_seconds'] = time.monotonic() - arrived
        profile.change('active_requests', -1)
        profile.finish(req.uttid, timings, success)
