"""Bounded, CPU-only stage timing; work sums are not pipeline wall time."""
from collections import Counter, deque
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Lock, Thread
import time

REWARD_TIMING_KEYS = (
    'client_slot_wait_seconds', 'client_copy_seconds', 'client_pool_wait_seconds',
    'client_audio_save_seconds', 'client_http_seconds', 'client_retry_count',
    'handler_queue_seconds', 'audio_load_seconds', 'target_audio_load_seconds', 'sim_queue_seconds',
    'sim_compute_seconds', 'asr_queue_seconds', 'asr_compute_seconds',
    'pause_align_queue_seconds', 'pause_align_compute_seconds',
    'pause_score_seconds', 'server_seconds', 'server_wall_seconds',
    'asr_batch_seconds', 'asr_batch_size',
    'pause_align_batch_seconds', 'pause_align_batch_size',
)
ROLLOUT_SUM_KEYS = (
    'generation_seconds', 'reference_prepare_seconds',
    'reward_submit_backpressure_seconds', 'reward_tail_wait_seconds',
    'candidate_generation_batches', 'reward_requests',
) + tuple('reward_' + key for key in REWARD_TIMING_KEYS)

@contextmanager
def timed_stage(timings, name, lock=None, service=None):
    """Queue = time waiting for this model lock; compute = time holding it."""
    started = time.monotonic()
    if service:
        service.change(name + '_queued', 1)
    acquired = False
    try:
        if lock is not None:
            lock.acquire()
            acquired = True
        tick = time.monotonic()
        if lock is not None:
            timings[name + '_queue_seconds'] = tick - started
        if service:
            service.change(name + '_queued', -1)
            service.change(name + '_active', 1)
        try:
            yield
        finally:
            timings[name + ('_compute_seconds' if lock is not None else '_seconds')] = time.monotonic() - tick
            if service:
                service.change(name + '_active', -1)
    finally:
        if acquired:
            lock.release()

class ServiceProfile:
    def __init__(self):
        self.lock = Lock()
        self.counts = Counter()
        self.totals = Counter()
        self.recent = deque(maxlen=256)
        self.started = time.monotonic()
        self.path = os.environ.get('M5_GRPO_REWARD_PROFILE_PATH')
        self.queue = Queue(maxsize=4096)
        self.writer = None
        if self.path:
            self.writer = Thread(target=self._write, name='reward-profile-writer', daemon=True)
            self.writer.start()

    def change(self, name, delta):
        with self.lock:
            self.counts[name] += delta

    def finish(self, uttid, timings, success):
        entry = dict(time=datetime.now(timezone.utc).isoformat(), uttid=uttid,
                     success=success, timings=timings)
        with self.lock:
            self.counts['completed' if success else 'failed'] += 1
            self.totals.update(timings)
            self.recent.append(entry)
        if self.path:
            try:
                self.queue.put_nowait(entry)
            except Full:
                self.change('profile_dropped', 1)

    def _write(self):
        # NFS flushes never hold the metrics lock or a scoring model lock.
        try:
            with Path(self.path).open('a', buffering=65536) as file:
                flushed = time.monotonic()
                while True:
                    try:
                        entry = self.queue.get(timeout=0.5)
                    except Empty:
                        file.flush()
                        flushed = time.monotonic()
                        continue
                    if entry is None:
                        file.flush()
                        return
                    file.write(json.dumps(entry, allow_nan=False) + '\n')
                    if time.monotonic() - flushed >= 0.5:
                        file.flush()
                        flushed = time.monotonic()
        except OSError:
            self.change('profile_write_errors', 1)
            logging.exception('Reward profiling output failed')

    def close(self):
        if self.writer is not None and self.writer.is_alive():
            try:
                self.queue.put(None, timeout=2)
                self.writer.join(timeout=10)
            except Full:
                self.change('profile_dropped', 1)

    def snapshot(self):
        with self.lock:
            return dict(uptime_seconds=time.monotonic()-self.started,
                        counts=dict(self.counts), totals=dict(self.totals),
                        profile_pending=self.queue.qsize(),
                        recent=list(self.recent)[-16:])
