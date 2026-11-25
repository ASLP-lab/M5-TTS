"""Bounded dynamic batches for frozen reward models; no HTTP or CUDA dependency."""
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Lock, Thread
import time


@dataclass(frozen=True)
class BatchResult:
    value: object
    queue_seconds: float
    batch_seconds: float
    batch_size: int

    def timings(self, name):
        # Share batch work, not its whole duration, across request accounting.
        return {name + '_queue_seconds': self.queue_seconds,
                name + '_compute_seconds': self.batch_seconds / self.batch_size,
                name + '_batch_seconds': self.batch_seconds,
                name + '_batch_size': self.batch_size}


class DynamicBatcher:
    """FIFO microbatches bounded by count, padded audio seconds and queue size.

    Each caller retains its own Future, including when batches fail. A short
    fill deadline starts at the first request; it never waits for a full batch.
    Model calls stay on one thread; evaluation can share the supplied lock.
    """
    def __init__(self, name, compute, max_batch_size=8, max_wait_seconds=.02,
                 max_queue_size=64, max_batch_seconds=160., model_lock=None):
        if not (1 <= max_batch_size <= max_queue_size and max_wait_seconds >= 0
                and max_batch_seconds > 0):
            raise ValueError('Invalid bounded batch configuration')
        self.name, self.compute = name, compute
        self.max_batch_size, self.max_wait_seconds = max_batch_size, max_wait_seconds
        self.max_batch_seconds, self.model_lock = max_batch_seconds, model_lock
        self.queue = Queue(maxsize=max_queue_size)
        self.gate, self.stats_lock = Lock(), Lock()
        self.closed = False
        self.stats = {'batches': 0, 'items': 0, 'failed_batches': 0,
                      'batch_seconds': 0., 'max_observed_batch': 0}
        self.worker = Thread(target=self._run, name='reward-batch-' + name, daemon=True)
        self.worker.start()

    def submit(self, payload, audio_seconds=0.):
        if not 0 <= audio_seconds < float('inf'):
            raise ValueError('Audio duration must be finite and nonnegative')
        future = Future()
        with self.gate:
            if self.closed:
                raise RuntimeError(self.name + ' batcher is closed')
            try:
                self.queue.put_nowait((payload, float(audio_seconds), future, time.monotonic()))
            except Full as exc:
                raise RuntimeError(self.name + ' reward queue is full') from exc
        return future

    def _run(self):
        carried = None
        stopping = False
        while not stopping:
            first = carried if carried is not None else self.queue.get()
            carried = None
            if first is None:
                break
            batch = [first]
            longest = first[1]
            deadline = first[3] + self.max_wait_seconds
            while len(batch) < self.max_batch_size:
                try:
                    # Drain already available work even after the fill deadline.
                    item = self.queue.get_nowait()
                except Empty:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        item = self.queue.get(timeout=remaining)
                    except Empty:
                        break
                if item is None:
                    stopping = True
                    break
                padded_cost = max(longest, item[1]) * (len(batch) + 1)
                if padded_cost > self.max_batch_seconds:
                    carried = item
                    break
                batch.append(item)
                longest = max(longest, item[1])
            batch = [item for item in batch if item[2].set_running_or_notify_cancel()]
            if not batch:
                continue
            acquired = False
            tick = time.monotonic()
            failed = False
            try:
                if self.model_lock is not None:
                    self.model_lock.acquire()
                    acquired = True
                tick = time.monotonic()
                values = list(self.compute([item[0] for item in batch]))
                if len(values) != len(batch):
                    raise ValueError(self.name + ' returned a different batch length')
                elapsed = time.monotonic() - tick
                for item, value in zip(batch, values):
                    item[2].set_result(BatchResult(value, tick - item[3], elapsed, len(batch)))
            except Exception as exc:
                failed = True
                elapsed = time.monotonic() - tick
                for item in batch:
                    item[2].set_exception(exc)
            finally:
                if acquired:
                    self.model_lock.release()
                with self.stats_lock:
                    self.stats['batches'] += 1
                    self.stats['items'] += len(batch)
                    self.stats['failed_batches'] += int(failed)
                    self.stats['batch_seconds'] += elapsed
                    self.stats['max_observed_batch'] = max(self.stats['max_observed_batch'], len(batch))

    def snapshot(self):
        with self.stats_lock:
            return dict(self.stats, queued=self.queue.qsize(), closed=self.closed,
                        max_batch_size=self.max_batch_size,
                        max_batch_seconds=self.max_batch_seconds,
                        max_wait_seconds=self.max_wait_seconds,
                        max_queue_size=self.queue.maxsize)

    def close(self):
        with self.gate:
            if self.closed:
                return
            self.closed = True
        # Drain accepted work before shutdown, including a partial last batch.
        self.queue.put(None)
        self.worker.join()
