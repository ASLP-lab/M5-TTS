# Copyright (c) 2025 Zhipu AI Inc (authors: CogAudio Group Members)
# Modified for the M5 colocated verl GRPO recipe.
# Licensed under the Apache License, Version 2.0.
"""Bounded asynchronous shared-file HTTP rewards with thread-local sessions."""
from concurrent.futures import ThreadPoolExecutor
import math
import time
from .reward_transport import post_with_retry
from pathlib import Path
from threading import BoundedSemaphore, Lock, local
import uuid
import requests
import torch
import torchaudio


class RewardClient:
    def __init__(self, url, temp_dir, timeout=300, workers=8, max_inflight=16):
        self.url = url.rstrip('/')
        if not self.url.endswith('/reward'):
            self.url += '/reward'
        if workers < 1 or max_inflight < workers:
            raise ValueError('Require 1 <= reward_workers <= reward_max_inflight')
        self.temp_dir = Path(temp_dir)
        self.timeout = timeout
        self.reward_keys = ('cer_reward', 'sim_reward', 'pause_reward')
        self._slots = BoundedSemaphore(max_inflight)
        self._lock, self._local = Lock(), local()
        self._sessions = []
        self._pending = set()
        self._closed = False
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='grpo-reward')

    def _session(self):
        if not hasattr(self._local, 'session'):
            self._local.session = requests.Session()
            with self._lock:
                self._sessions.append(self._local.session)
        return self._local.session

    def _reset_session(self):
        session = getattr(self._local, 'session', None)
        if session is not None:
            session.close()
            del self._local.session

    def _request(self, waveform, uttid, target_audio, ref_text, timings=None, queued_at=None,
                 frame_metadata=None, language='Chinese'):
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        path = (self.temp_dir / (uuid.uuid4().hex + '.wav')).resolve()
        completed, retries = False, 0
        timings = dict(timings or {})
        timings['client_pool_wait_seconds'] = time.monotonic() - queued_at if queued_at else 0.0
        try:
            tick = time.monotonic()
            torchaudio.save(str(path), waveform, 24000)
            timings['client_audio_save_seconds'] = time.monotonic() - tick
            tick = time.monotonic()
            response, retries = post_with_retry(self._session, self._reset_session, self.url, {
                'audio_path': str(path), 'uttid': str(uttid),
                'target_audio': target_audio, 'ref_text': ref_text,
                'frame_metadata': frame_metadata, 'language': language,
            }, timeout=self.timeout)
            timings['client_http_seconds'] = time.monotonic() - tick
            timings['client_retry_count'] = retries
            completed = True
            response.raise_for_status()
            result = response.json()
            result.setdefault('timings', {}).update(timings)
            info = result['reward_info']
            if not math.isfinite(float(result['reward'])) or not all(
                    key in info and not isinstance(info[key], bool) and math.isfinite(float(info[key]))
                    for key in self.reward_keys):
                raise ValueError('Missing or invalid reward component in response')
            return result
        finally:
            # A timed-out handler can still be using this shared file.
            if completed and retries == 0:
                path.unlink(missing_ok=True)

    def _finished(self, future):
        with self._lock:
            self._pending.discard(future)
        self._slots.release()



    def submit(self, uttid, response_token, response_audio, target_audio, ref_text,
               frame_metadata=None, language='Chinese'):
        if frame_metadata is not None:
            frame_metadata = dict(frame_metadata)
            frames = frame_metadata['num_frames']
            decisions = frame_metadata.get('num_decisions', frames)
            eos = frame_metadata.get('has_eos_decision', False)
            if type(eos) is not bool or decisions != frames + int(eos) or decisions != len(response_token):
                raise ValueError('Frame metadata does not match generated decisions')
            if eos and any(response_token[-1][:16]):
                raise ValueError('EOS metadata does not match the sampled EOS row')
        started = time.monotonic()
        self._slots.acquire()
        timings = {'client_slot_wait_seconds': time.monotonic() - started}
        try:
            if self._closed:
                raise RuntimeError('Reward client is closed')
            # CUDA work stays on the actor stream; HTTP workers receive CPU data.
            tick = time.monotonic()
            waveform = response_audio.detach().to(device='cpu', dtype=torch.float32, copy=True).reshape(1, -1)
            timings['client_copy_seconds'] = time.monotonic() - tick
            queued_at = time.monotonic()
            future = self._pool.submit(self._request, waveform, uttid, target_audio, ref_text,
                                       timings, queued_at, frame_metadata, language)
            with self._lock:
                self._pending.add(future)
            future.add_done_callback(self._finished)
            return future
        except BaseException:
            self._slots.release()
            raise

    def close(self):
        self._closed = True
        self._pool.shutdown(wait=True, cancel_futures=True)
        for session in self._sessions:
            session.close()
