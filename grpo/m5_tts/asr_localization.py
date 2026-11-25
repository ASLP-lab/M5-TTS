"""Diagnostic text edits -> acoustic intervals -> codec frames, never rewards."""
from collections import Counter
import math
import zhconv
from .run_wer import process_one, punctuation_all

_IGNORED = set(punctuation_all) - {"'"}


def chars_with_offsets(text):
    return [(c, i) for i, c in enumerate(text) if not c.isspace() and c not in _IGNORED]


def _canonical_unit(text, language):
    text = zhconv.convert(text, 'zh-cn') if language == 'Chinese' else text
    return ''.join(c for c, _ in chars_with_offsets(text))


def _frame_context(meta, duration_ms):
    if not meta:
        return None
    n, rate, stride = meta['num_frames'], meta['sample_rate'], meta['frame_samples']
    if any(type(v) is not int or v <= 0 for v in (n, rate, stride)):
        raise ValueError('Frame metadata requires positive integer count/rate/stride')
    frame_ms = 1000 * stride / rate
    decoded_ms = n * frame_ms
    if duration_ms > decoded_ms + 1000 / rate:
        raise ValueError('Observed audio is longer than declared codec frames')
    trim_ms = max(0., decoded_ms - duration_ms)
    return dict(num_frames=n, frame_ms=frame_ms, sample_rate=rate,
                frame_samples=stride, decoded_duration_ms=decoded_ms,
                observed_duration_ms=duration_ms, trimmed_tail_ms=trim_ms,
                stop_reason=meta.get('stop_reason', 'not_recorded'),
                codebooks_per_frame=meta.get('codebooks_per_frame', 16),
                first_partly_or_fully_unobserved_frame=(math.floor(duration_ms / frame_ms)
                    if trim_ms > 1000 / rate else None))


def _project(start, end, context):
    if context is None or start is None or end is None or end <= start:
        return None
    first = max(0, math.floor(start / context['frame_ms'] + 1e-9))
    last = min(context['num_frames'], math.ceil(end / context['frame_ms'] - 1e-9))
    return [first, last] if last > first else None


def localize_errors(asr_result, reference, duration_ms, frame_metadata=None, language='Chinese'):
    """Keep the historical CER normalization and mark uncertainty explicitly.

    No artificial durations for deletions. No positional zip after text cleanup.
    Repeated-text edit paths are diagnosed as potentially ambiguous, not truth.
    """
    if not math.isfinite(duration_ms) or duration_ms < 0:
        raise ValueError('Invalid audio duration')
    raw = asr_result['text']
    transcript = zhconv.convert(raw, 'zh-cn') if language == 'Chinese' else raw
    _, _, cer, ref, hyp, edits = process_one(transcript, reference, 'zh')
    # The offsets below index the simplified transcript, not the raw ASR string.
    normalized_hyp = chars_with_offsets(transcript)
    normalized_ref = chars_with_offsets(reference)
    if [x[0] for x in normalized_hyp] != hyp or [x[0] for x in normalized_ref] != ref:
        raise ValueError('CER normalization contract changed')
    context = _frame_context(frame_metadata, duration_ms)
    units = []
    previous_start = previous_end = -1.
    for original in asr_result.get('aligned_units', []):
        start, end = float(original['start_ms']), float(original['end_ms'])
        status = 'valid'
        if not math.isfinite(start) or not math.isfinite(end):
            status = 'non_finite';start = end = None
        elif start < 0 or end > duration_ms + 1e-6 or end < start:
            status = 'out_of_range'
        elif start < previous_start or end < previous_end:
            status = 'non_monotonic'
        elif end == start:
            status = 'zero_duration'
        if start is not None:
            previous_start, previous_end = start, end
        units.append(dict(text=original['text'], start_ms=start, end_ms=end,
                          status=status, frame_span=(_project(start, end, context)
                              if status == 'valid' else None)))
    unit_text = ''.join(_canonical_unit(u['text'], language) for u in units)
    mapping_ok = unit_text == ''.join(hyp)
    status = ('empty_transcript' if not hyp else 'missing_timestamps' if not units
              else 'unit_text_mismatch' if not mapping_ok else
              'invalid_timestamps' if any(u['status']!='valid' for u in units) else 'aligned')
    char_times = []
    if mapping_ok:
        for j, u in enumerate(units):
            chars = _canonical_unit(u['text'], language)
            for char in chars:
                char_times.append(dict(char=char, unit_index=j, start_ms=u['start_ms'],
                    end_ms=u['end_ms'], status=u['status'],
                    granularity='character' if len(chars)==1 else 'shared_word_interval'))
    else:
        char_times = [None] * len(hyp)
    if len(char_times) != len(hyp):
        raise ValueError('Timestamp mapping length mismatch')
    errors, operations = [], []
    counts = Counter(hyp)
    for chunk in edits:
        kind = chunk.type
        a,b,c,d = chunk.ref_start_idx,chunk.ref_end_idx,chunk.hyp_start_idx,chunk.hyp_end_idx
        op = dict(type=kind, ref_span=[a,b], hyp_span=[c,d],
                  expected=''.join(ref[a:b]), recognized=''.join(hyp[c:d]))
        operations.append(op)
        if kind == 'equal':
            continue
        event = dict(op, start_ms=None, end_ms=None, frame_span=None,
                     boundary_interval_ms=None, location_status='unlocated',
                     alignment_path='one_minimum_edit_path',
                     repetition_ambiguity=any(counts[x]>1 for x in hyp[max(0,c-1):min(len(hyp),d+1)]))
        if kind == 'delete':
            left = char_times[c-1] if c>0 and char_times else None
            right = char_times[c] if c<len(char_times) else None
            l = left['end_ms'] if left and left['status']=='valid' else (0. if c==0 else None)
            r = right['start_ms'] if right and right['status']=='valid' else (duration_ms if c==len(hyp) else None)
            event.update(boundary_interval_ms=[l,r],
                         location_status='deletion_boundary' if l is not None and r is not None else 'deletion_unlocated',
                         deletion_kind='whole_utterance' if not hyp else 'suffix' if c==len(hyp)
                            else 'prefix' if c==0 else 'internal')
            # Deliberately leave start/end/frame_span empty: absent speech is not a real span.
        else:
            selected = char_times[c:d]
            if selected and all(x and x['status']=='valid' for x in selected):
                start,end=selected[0]['start_ms'],selected[-1]['end_ms']
                if end>start:
                    event.update(start_ms=start,end_ms=end,frame_span=_project(start,end,context),
                                 location_status='located')
            if status in ('unit_text_mismatch','missing_timestamps'):event['location_status']=status
        errors.append(event)
    return dict(schema_version=1,backend=asr_result.get('backend','seaco'),
                model=asr_result.get('model'),aligner_model=asr_result.get('aligner_model'),
                raw_transcript=raw,transcript=transcript,target_text=reference,
                timestamp_unit='ms',status=status,cer=float(cer),
                reference_chars=ref,hypothesis_chars=hyp,
                reference_source_offsets=[i for _,i in normalized_ref],
                transcript_source_offsets=[i for _,i in normalized_hyp],
                aligned_units=units,char_intervals=char_times,edits=operations,errors=errors,
                frame_context=context,
                interpretation='Diagnostic ASR/edit alignment, not verified pronunciation or causal token blame')
