"""Inspectable CER preprocessing; no advantage estimator or decay law is chosen.

Consume the existing ASR/edit alignment. Preserve its character normalization,
time uncertainty and actual sampled-EOS contract. A caller may supply one
nonnegative curve per edit unit; otherwise only the observed local support is
used. These are diagnostic CER costs, not group-relative advantages.
"""
from bisect import bisect_left
import math

from .token_cer import _metadata


CLAUSE_BOUNDARIES = frozenset('，,。.!！?？;；:：\n\r…')


def split_clauses(text, reference_chars, source_offsets):
    """Half-open raw-source/ref spans. Do not split decimal points or renormalize."""
    if not isinstance(text, str) or len(reference_chars) != len(source_offsets):
        raise ValueError('Reference text/offset contract mismatch')
    if not reference_chars:
        raise ValueError('A nonempty CER reference is required')
    previous = -1
    for char, offset in zip(reference_chars, source_offsets):
        if (type(offset) is not int or not previous < offset < len(text)
                or text[offset] != char):
            raise ValueError('Reference source offsets must match original text')
        previous = offset
    ends = []
    for i, char in enumerate(text):
        decimal = (char == '.' and 0 < i < len(text) - 1
                   and text[i - 1].isdigit() and text[i + 1].isdigit())
        if char in CLAUSE_BOUNDARIES and not decimal:
            ends.append(i + 1)
    ends.append(len(text))
    clauses, cursor = [], 0
    for end in sorted(set(ends)):
        lo = bisect_left(source_offsets, cursor)
        hi = bisect_left(source_offsets, end)
        if hi > lo:
            clauses.append(dict(clause_id=len(clauses), source_span=[cursor, end],
                                ref_span=[lo, hi], text=text[cursor:end]))
            cursor = end
        elif clauses:
            clauses[-1]['source_span'][1] = end
            clauses[-1]['text'] = text[clauses[-1]['source_span'][0]:end]
            cursor = end
        # Leading punctuation stays attached to the first nonempty clause.
    return clauses


def _overlaps(start, end, n, total, frame_ms, duration):
    weights = [0.0] * total
    for t in range(n):
        overlap = max(0.0, min((t + 1) * frame_ms, end, duration)
                      - max(t * frame_ms, start))
        weights[t] = overlap / frame_ms
    return weights


def _time_at(times, index, duration):
    row = times[index] if 0 <= index < len(times) else None
    if not row or row.get('status') != 'valid':
        return None
    start, end = row.get('start_ms'), row.get('end_ms')
    if (isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
            or not math.isfinite(start) or not math.isfinite(end)
            or not 0 <= start < end <= duration + 1e-6):
        return None
    return row


def _atoms(alignment):
    """Split the existing minimum-edit path, without recomputing text alignment."""
    ref, hyp = alignment['reference_chars'], alignment['hypothesis_chars']
    errors = {(e['type'], tuple(e['ref_span']), tuple(e['hyp_span'])): e
              for e in alignment['errors']}
    ra = ha = 0
    for op in alignment['edits']:
        a, b = op['ref_span']; c, d = op['hyp_span']; kind = op['type']
        if (any(type(x) is not int for x in [a, b, c, d])
                or not (a == ra <= b <= len(ref) and c == ha <= d <= len(hyp))
                or (a == b and c == d)):
            raise ValueError('Edit path is not a complete monotonic partition')
        ra, ha = b, d
        if kind not in ('equal', 'replace', 'substitute', 'insert', 'delete'):
            raise ValueError('Unknown edit operation')
        if kind == 'equal' and (ref[a:b] != hyp[c:d] or b - a != d - c):
            raise ValueError('Equal edit contains different text')
        if kind == 'insert' and (a != b or c == d):
            raise ValueError('Invalid insertion')
        if kind == 'delete' and (c != d or a == b):
            raise ValueError('Invalid deletion')
        parent = {} if kind == 'equal' else errors.get((kind, (a, b), (c, d)))
        if parent is None:
            raise ValueError('Error record missing for edit operation')
        for k in range(max(b - a, d - c)):
            ri = a + k if k < b - a else None
            hi = c + k if k < d - c else None
            atomic_kind = ('insert' if ri is None else 'delete' if hi is None
                           else 'equal' if kind == 'equal' else 'substitute')
            yield dict(type=atomic_kind, ref_index=ri, hyp_index=hi,
                       ref_span=[ri, ri + 1] if ri is not None else [b, b],
                       hyp_span=[hi, hi + 1] if hi is not None else [d, d],
                       expected=ref[ri] if ri is not None else '',
                       recognized=hyp[hi] if hi is not None else '',
                       parent_ref_span=[a, b], parent_hyp_span=[c, d],
                       repetition_ambiguity=bool(parent.get('repetition_ambiguity')),
                       within_chunk_ambiguity=(kind in ('replace', 'substitute')
                                               and b - a != d - c))
    if ra != len(ref) or ha != len(hyp):
        raise ValueError('Edit path does not cover reference and hypothesis')


def build_structure(alignment, metadata, *, ambiguous='use_path'):
    """Stages 1-3: clauses, per-character localization and explicit error masks.

    Prefix support is eligibility only, not an assigned reward or decay curve.
    Insertion slots are reference gaps (0..N); deletions never acquire fake audio.
    """
    if ambiguous not in ('use_path', 'terminal'):
        raise ValueError('ambiguous must be use_path or terminal')
    n, total, eos, frame_ms, duration = _metadata(alignment, metadata)
    ref, hyp = alignment['reference_chars'], alignment['hypothesis_chars']
    clauses = split_clauses(alignment['target_text'], ref,
                            alignment['reference_source_offsets'])
    clause_of = {j: c['clause_id'] for c in clauses
                 for j in range(*c['ref_span'])}
    times = alignment['char_intervals']
    if len(times) != len(hyp):
        raise ValueError('Hypothesis timestamp count mismatch')
    units, error_units = [], []
    ref_error, hyp_error = [0] * len(ref), [0] * len(hyp)
    insertion_gaps = [0] * (len(ref) + 1)
    masks = {key: [0] * total for key in
             ('error', 'observed_error', 'surrogate', 'prefix_eligible')}
    for atom in _atoms(alignment):
        ri, hi, kind = atom['ref_index'], atom['hyp_index'], atom['type']
        index = ri if ri is not None else min(atom['ref_span'][0], len(ref) - 1)
        clause = clauses[clause_of[index]]
        gap = atom['ref_span'][0] if kind == 'insert' else None
        unit = dict(atom, unit_id=f'ref:{ri}' if ri is not None else f'gap:{gap}:hyp:{hi}',
                    clause_id=clause['clause_id'], insertion_gap=gap,
                    source_offset=alignment['reference_source_offsets'][ri] if ri is not None else None,
                    start_ms=None, end_ms=None, granularity=None, frame_span=None,
                    assignment='unlocated', local_weights=[0.0] * total,
                    prefix_support_weights=[0.0] * total, prefix_start_ms=None,
                    boundary_interval_ms=None, cer_mass=0.0 if kind == 'equal' else 1.0 / len(ref))
        stamp = _time_at(times, hi, duration) if hi is not None else None
        if alignment.get('status') in ('unit_text_mismatch', 'missing_timestamps'):
            stamp = None
        if stamp:
            unit.update(start_ms=stamp['start_ms'], end_ms=stamp['end_ms'],
                        granularity=stamp.get('granularity', 'unknown'))
        terminal_ambiguity = (ambiguous == 'terminal'
                              and (atom['repetition_ambiguity'] or atom['within_chunk_ambiguity']))
        if stamp and not terminal_ambiguity:
            unit['local_weights'] = _overlaps(stamp['start_ms'], stamp['end_ms'],
                                               n, total, frame_ms, duration)
            indices = [t for t, w in enumerate(unit['local_weights']) if w]
            unit['frame_span'] = [indices[0], indices[-1] + 1] if indices else None
            unit['assignment'] = 'acoustic_interval'
        if kind != 'equal':
            if ri is not None: ref_error[ri] = 1
            if hi is not None: hyp_error[hi] = 1
            if gap is not None: insertion_gaps[gap] += 1
            cut = unit['start_ms']
            if not any(unit['local_weights']):
                anchor, cut = (n if eos else total - 1), duration
                assignment = ('eos_deletion' if kind == 'delete' else 'eos_unlocated') if eos else 'cap_terminal_surrogate'
                if kind == 'delete' and not terminal_ambiguity:
                    boundary = atom['hyp_span'][0]
                    left = _time_at(times, boundary - 1, duration)
                    right = _time_at(times, boundary, duration)
                    lo = left['end_ms'] if left else (0.0 if boundary == 0 else None)
                    high = right['start_ms'] if right else (duration if boundary == len(hyp) else None)
                    unit['boundary_interval_ms'] = [lo, high]
                    unit['deletion_kind'] = ('whole_utterance' if not hyp else
                                             'suffix' if boundary == len(hyp) else
                                             'prefix' if boundary == 0 else 'internal')
                    if unit['deletion_kind'] in ('prefix', 'internal') and high is not None:
                        last = min(n - 1, max(0, math.ceil(duration / frame_ms) - 1))
                        anchor, cut = min(last, int(high // frame_ms)), high
                        assignment = 'deletion_boundary_surrogate'
                unit['local_weights'][anchor] = 1.0
                unit['assignment'] = assignment
            unit['prefix_cut_ms'] = cut
            for t, w in enumerate(unit['local_weights']):
                if w:
                    masks['error'][t] = 1
                    masks['observed_error' if unit['assignment'] == 'acoustic_interval' else 'surrogate'][t] = 1
            error_units.append(unit)
        units.append(unit)
    by_ref = {u['ref_index']: u for u in units if u['ref_index'] is not None}
    for unit in error_units:
        clause = clauses[unit['clause_id']]
        cut = unit['prefix_cut_ms']
        stop = unit['ref_span'][0]
        candidates = [by_ref[j]['start_ms'] for j in range(clause['ref_span'][0], stop)
                      if by_ref[j]['type'] == 'equal' and by_ref[j]['assignment'] == 'acoustic_interval'
                      and by_ref[j]['start_ms'] <= cut]
        start = 0.0 if unit['clause_id'] == 0 else min(candidates) if candidates else None
        # Shared-word timestamps can straddle punctuation. Do not advertise
        # earlier-clause frames as eligible in that uncertain case.
        previous_ends = [u['end_ms'] for u in units
                         if u['clause_id'] < unit['clause_id'] and u['end_ms'] is not None]
        if start is not None and previous_ends:
            start = max(start, max(previous_ends))
        if start is not None and start < cut:
            support = _overlaps(start, cut, n, total, frame_ms, duration)
            for t, w in enumerate(support):
                if unit['local_weights'][t]: support[t] = 0.0
                elif w: masks['prefix_eligible'][t] = 1
            unit['prefix_support_weights'] = support
            unit['prefix_start_ms'] = start
    cer = float(alignment['cer'])
    if not math.isfinite(cer) or abs(len(error_units) / len(ref) - cer) > 1e-8:
        raise ValueError('Atomic edit count disagrees with existing CER')
    return dict(schema_version=1, target_text=alignment['target_text'],
                transcript=alignment['transcript'], clauses=clauses,
                reference_units=[by_ref[j] for j in range(len(ref))],
                insertion_units=[u for u in units if u['type'] == 'insert'],
                error_units=error_units, reference_error_mask=ref_error,
                hypothesis_error_mask=hyp_error, insertion_gap_counts=insertion_gaps,
                masks=masks, num_audio_frames=n, num_decisions=total,
                eos_decision_index=n if eos else None, frame_ms=frame_ms,
                observed_duration_ms=duration, edit_count=len(error_units), cer=cer,
                interpretation='Diagnostic edit masks; prefix eligibility is not causal evidence')
