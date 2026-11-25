"""M5 conditioning and codebooks; reward futures overlap subsequent generation."""
from collections import OrderedDict
from contextlib import closing
import time
import json
import logging
from .performance import REWARD_TIMING_KEYS
import torch
from .reference_protocol import load_reference_audio, continuation_context
from .data_types import Episode
from .generator_m5 import Segment, load_m5_for_rl


class M5Adapter:
    def __init__(self, model_config, device):
        self.generator, self.tokenizer = load_m5_for_rl(model_config, device)
        self.device = device
        self.last_rollout_metrics = {}
        self._reference_cache = OrderedDict()

    @property
    def model(self):
        return self.generator._model

    def prepare_context(self, record, text, legacy_accent=None):
        """Use the same reference for generation and SIM; cache frozen codec IDs."""
        target = Segment(text, record.get('speaker', 0), None,
                         record.get('accent', legacy_accent), record['lang'], record.get('caption'))
        path = record.get('wav_path')
        if not path:
            return text, [target]
        transcript = record.get('prompt_text', '').strip()
        if not transcript:
            raise ValueError('Reference-conditioned rollout requires prompt_text with wav_path')
        key = (path, transcript, target.speaker, target.accent, target.lang, target.caption)
        if key not in self._reference_cache:
            audio = load_reference_audio(path, self.generator.sample_rate)
            codes = self.generator._audio_tokenizer.encode(audio).detach().cpu()
            reference = Segment(transcript, target.speaker, audio, target.accent,
                                target.lang, target.caption, audio_codes=codes)
            self._reference_cache[key] = reference
            while len(self._reference_cache) > 64:
                self._reference_cache.popitem(last=False)
        self._reference_cache.move_to_end(key)
        return None, continuation_context(self._reference_cache[key], target)

    @torch.no_grad()
    def rollout(self, records, n, reward_client, conf, rank, step, rng=None, stop=None):
        generation = conf['generation_conf']
        candidate_batch = conf.get('rollout_micro_batch_size', 1)
        if type(candidate_batch) is not int or not 1 <= candidate_batch <= 256:
            raise ValueError('rollout_micro_batch_size must be an integer in 1..256')
        candidate_batch = min(candidate_batch, n)
        was_training = self.model.training
        self.model.eval()
        jobs = []
        reward_timings = {'reward_' + key: 0.0 for key in REWARD_TIMING_KEYS}
        reference_seconds = 0.0
        generation_seconds = 0.0
        submit_seconds = 0.0
        generation_batches = 0
        started = time.monotonic()
        try:
            for prompt_index, record in enumerate(records):
                raw_text = record['text']
                if '<|endofprompt|>' in raw_text:
                    legacy_accent, text = raw_text.split('<|endofprompt|>', 1)
                else:
                    legacy_accent, text = None, raw_text
                group_started = tick = time.monotonic()
                generation_text, context = self.prepare_context(record, text, legacy_accent)
                reference_seconds += time.monotonic() - tick
                group_id = f'{rank}:{prompt_index}:{record["utt"]}'
                outputs = self.generator.iter_candidates(
                    text=generation_text, speaker=context[-1].speaker, context=context,
                    n=n, batch_size=candidate_batch,
                    max_audio_length_ms=generation.get('max_audio_length_ms', 40000),
                    temperature=generation.get('temperature', 1.0), topk=generation.get('topk', 50),
                    max_seq_len=generation.get('max_seq_len', 2048), rng=rng, stop=stop,
                    return_metadata=True)
                prefix_ids = None
                generation_batches += (n + candidate_batch - 1) // candidate_batch
                with closing(outputs):
                    for candidate in range(n):
                        tick = time.monotonic()
                        generated = next(outputs)
                        audio, codes, prefix = generated[:3]
                        stop_info = generated[3]
                        codes = codes.reshape(-1, 16)
                        ids = torch.cat([codes, codes.new_zeros((len(codes), 1))], dim=-1).tolist()
                        if stop_info.get('has_eos_decision', False):
                            ids.append([0] * 17)  # The EOS was sampled; not sent to the waveform decoder.
                        frame_metadata = dict(num_frames=len(codes),
                            sample_rate=self.generator.sample_rate,
                            frame_samples=self.generator._audio_tokenizer.frame_samples,
                            codebooks_per_frame=16, stop_reason=stop_info.get('stop_reason', 'not_recorded'))
                        frame_metadata.update(num_decisions=len(ids), has_eos_decision=stop_info['has_eos_decision'])
                        if prefix_ids is None:
                            prefix_ids = prefix.reshape(-1, 17).tolist()
                        generation_seconds += time.monotonic() - tick
                        tick = time.monotonic()
                        future = reward_client.submit(
                            uttid=f'{group_id}:{step}:{candidate}', response_token=ids,
                            response_audio=audio, target_audio=record.get('wav_path'),
                            ref_text=text,
                            frame_metadata=frame_metadata, language=record.get('lang') or 'Chinese')
                        jobs.append((prefix_ids, ids, group_id, prompt_index,
                                     frame_metadata, future))
                        submit_seconds += time.monotonic() - tick
                        for job in jobs:
                            if job[-1].done():
                                job[-1].result()
                logging.info('actor_progress %s', json.dumps(dict(
                    worker=rank, sequence=step, groups_submitted=prompt_index + 1,
                    groups_total=len(records), candidates_submitted=len(jobs),
                    rewards_completed=sum(job[-1].done() for job in jobs),
                    generation_seconds=generation_seconds,
                    reward_submit_backpressure_seconds=submit_seconds,
                    group_wall_seconds=time.monotonic() - group_started,
                    elapsed_seconds=time.monotonic() - started)))
            waited = time.monotonic()
            episodes = []
            for prefix_ids, ids, group_id, prompt_index, generation_metadata, future in jobs:
                result = future.result()
                for key in REWARD_TIMING_KEYS:
                    reward_timings['reward_' + key] += float(result.get('timings', {}).get(key, 0.0))
                episodes.append(Episode(prefix_ids, ids, group_id,
                                        result['reward'], result['reward_info'], prompt_index,
                                        asr_alignment=result.get('asr_alignment'),
                                        generation_metadata=generation_metadata))
            self.last_rollout_metrics = {
                **reward_timings,
                'reward_requests': len(jobs),
                'reference_prepare_seconds': reference_seconds,
                'generation_seconds': generation_seconds,
                'reward_submit_backpressure_seconds': submit_seconds,
                'reward_tail_wait_seconds': time.monotonic() - waited,
                'rollout_seconds': time.monotonic() - started,
                'candidate_generation_batches': generation_batches,
                'rollout_micro_batch_size': candidate_batch,
            }
            return episodes
        except BaseException:
            for job in jobs:
                job[-1].cancel()
            for job in jobs:
                try:
                    job[-1].result()
                except BaseException:
                    pass
            raise
        finally:
            self.model.train(was_training)
