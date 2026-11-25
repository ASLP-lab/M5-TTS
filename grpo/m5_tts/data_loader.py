"""CPU-only JSONL pipeline: rank/worker shards and a bounded shuffle buffer."""
import json
import random
from .m5_conditioning import require_lang


class PromptDataset:
    def __init__(self, files, rank, world, seed, shuffle_buffer, repeats, references):
        self.file, = files
        self.rank, self.world = rank, world
        self.seed, self.shuffle_buffer, self.repeats = seed, shuffle_buffer, repeats
        self.references = list(references)
        random.Random(seed).shuffle(self.references)

    def __iter__(self):
        reference_position = self.rank
        for cycle in range(self.repeats):
            for record in self._iter_cycle(cycle):
                ref = self.references[reference_position % len(self.references)]
                reference_position += self.world
                record.update(reference_id=ref['reference_id'], reference_speaker_id=ref['speaker_id'],
                              wav_path=ref['wav_path'], prompt_text=ref['prompt_text'])
                yield record

    def _iter_cycle(self, cycle):
        rng = random.Random(self.seed + 104729 * cycle + self.rank)
        pool = []
        seen = 0
        with open(self.file, encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip() or (line_number - 1 + cycle) % self.world != self.rank:
                    continue
                record = json.loads(line)
                if not isinstance(record.get('text'), str) or not record['text'].strip():
                    raise ValueError(f'{self.file}:{line_number} needs non-empty text')
                record['lang'] = require_lang(record.get('lang'))
                record['utt'] = str(record.get('utt', record.get('uttid', f'{self.file}:{line_number}')))
                seen += 1
                pool.append(record)
                if len(pool) >= self.shuffle_buffer:
                    yield pool.pop(rng.randrange(len(pool)))
        while pool:
            yield pool.pop(rng.randrange(len(pool)))
        if not seen:
            raise ValueError(f'Empty prompt shard for rank {self.rank}')
