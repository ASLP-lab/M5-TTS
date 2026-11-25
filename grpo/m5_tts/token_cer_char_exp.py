"""Full-group character CER advantages with a fixed exp(-character distance) kernel.

Not a mass redistribution: a character retains its own advantage. Only negative
raw advantages also reach preceding reference characters in the SAME clause.
All statistics precede learner down-selection. No wall-clock decay is used.
"""
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import math

from .token_cer_stages import build_structure

MODE = 'char_exp'
RHO = math.exp(-1.0)
EPS = 1e-8


def character_group_advantages(costs):
    """r=-edit_count; population z-score independently at each reference position."""
    if len(costs) < 2 or not costs[0] or any(len(row) != len(costs[0]) for row in costs):
        raise ValueError('Character CER needs a rectangular group with >=2 candidates')
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           or not math.isfinite(x) or x < 0 for row in costs for x in row):
        raise ValueError('Character edit counts must be finite and nonnegative')
    means, stds = [], []
    for column in zip(*costs):
        mean = -math.fsum(column) / len(costs)
        variance = math.fsum((-x - mean) ** 2 for x in column) / len(costs)
        means.append(mean)
        stds.append(math.sqrt(variance))
    advantages = [[0.0 if std == 0 else (-x - mean) / (std + EPS)
                   for x, mean, std in zip(row, means, stds)] for row in costs]
    return advantages, means, stds


def decay_character_advantages(advantages, clause_ids):
    """O(N) evaluation of A_i + sum_{j>i,same clause} exp(-(j-i))*min(A_j,0)."""
    if len(advantages) != len(clause_ids) or not advantages:
        raise ValueError('Character/clause vector size mismatch')
    if any(not math.isfinite(x) for x in advantages):
        raise ValueError('Non-finite character advantage')
    if any(type(x) is not int or x < 0 for x in clause_ids):
        raise ValueError('Invalid clause ID')
    if any(b < a for a, b in zip(clause_ids, clause_ids[1:])):
        raise ValueError('Clause IDs must form ordered contiguous blocks')
    prefix = [0.0] * len(advantages)
    accumulated = 0.0
    for i in range(len(advantages) - 1, -1, -1):
        if i == len(advantages) - 1 or clause_ids[i] != clause_ids[i + 1]:
            accumulated = 0.0
        prefix[i] = RHO * accumulated
        accumulated = min(advantages[i], 0.0) + prefix[i]
    return [a + p for a, p in zip(advantages, prefix)], prefix


def character_costs_and_support(structure):
    """Fold insertions to next reference character (tail insertions to last).

    Preserve actual insertion intervals as additional support of that slot.
    Thus sum(costs)/N remains CER; insertions are not silently dropped. A slot
    may have >1 edit, so its advantage is relative, not an absolute correctness
    label. Alias intervals are merged with max rather than counted twice.
    """
    units = structure['reference_units']
    n, total = len(units), structure['num_decisions']
    if not n:
        raise ValueError('Empty reference')
    costs = [float(u['type'] != 'equal') for u in units]
    support = [list(u['local_weights']) for u in units]
    insertion_anchors = []
    for unit in structure['insertion_units']:
        anchor = min(unit['insertion_gap'], n - 1)
        costs[anchor] += 1.0
        support[anchor] = [max(a, b) for a, b in zip(support[anchor], unit['local_weights'])]
        insertion_anchors.append(dict(unit_id=unit['unit_id'], reference_index=anchor))
    if any(len(row) != total or any(not math.isfinite(w) or w < 0 or w > 1 for w in row)
           for row in support):
        raise ValueError('Invalid character frame support')
    if abs(math.fsum(costs) / n - structure['cer']) > 1e-8:
        raise ValueError('Character edit counts disagree with utterance CER')
    return costs, support, insertion_anchors


def project_to_decisions(advantages, support):
    """Overlap-weighted projection, not division by character duration.

    A full frame of a single character receives its full advantage. When word
    timestamps alias or character boundaries share a frame, take the weighted
    mean rather than multiply the same acoustic interval. Uncovered frames=0.
    """
    if len(advantages) != len(support) or not support:
        raise ValueError('Projection size mismatch')
    total = len(support[0])
    if any(len(row) != total for row in support):
        raise ValueError('Ragged decision support')
    numerator, denominator = [0.0] * total, [0.0] * total
    for a, row in zip(advantages, support):
        if not math.isfinite(a):
            raise ValueError('Non-finite projected advantage')
        for t, w in enumerate(row):
            if not math.isfinite(w) or not 0 <= w <= 1:
                raise ValueError('Invalid projection weight')
            numerator[t] += w * a
            denominator[t] += w
    return [a / max(1.0, w) for a, w in zip(numerator, denominator)]


def _fingerprint(ep):
    value = dict(alignment=ep.asr_alignment, metadata=ep.generation_metadata,
                 tokens=ep.generated_token_ids, group=ep.group_token_ids)
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode('utf-8')).hexdigest()


def _structure(ep, cfg):
    value = build_structure(ep.asr_alignment, ep.generation_metadata,
                            ambiguous=cfg['ambiguous'])
    if len(ep.generated_token_ids) != value['num_decisions']:
        raise ValueError('Character CER decisions do not match replay targets')
    if value['eos_decision_index'] is not None and any(ep.generated_token_ids[-1][:16]):
        raise ValueError('Character CER EOS row is not a sampled EOS')
    return value


def attach_group_advantages(episodes, conf):
    """Freeze profiles on the complete rollout group, before applying frame credit."""
    from .token_cer import settings
    cfg = settings(conf)
    if cfg['mode'] != MODE or not cfg['enabled']:
        raise ValueError('Character advantage attachment needs enabled char_exp mode')
    expected = conf.get('num_answers_per_question')
    if type(expected) is not int or expected < 2:
        raise ValueError('Character CER requires explicit full rollout group size >=2')
    groups = defaultdict(list)
    for index, ep in enumerate(episodes):
        groups[ep.group_token_ids].append((index, ep))
    if not groups:
        raise ValueError('Empty character CER rollout batch')
    output = list(episodes)
    signal = False
    for group in groups.values():
        if len(group) != expected:
            raise ValueError(f'Character CER expected {expected} full-group candidates, got {len(group)}')
        structures = [_structure(ep, cfg) for _, ep in group]
        first_ep = group[0][1]
        if any(ep.prompt_index != first_ep.prompt_index
               or ep.asr_alignment['target_text'] != first_ep.asr_alignment['target_text']
               or ep.asr_alignment['reference_chars'] != first_ep.asr_alignment['reference_chars']
               for _, ep in group):
            raise ValueError('Mixed reference/prompt identity in character CER group')
        entries = [character_costs_and_support(s) for s in structures]
        all_advantages, means, stds = character_group_advantages([x[0] for x in entries])
        for (index, ep), structure, (cost, support, anchors), raw in zip(
                group, structures, entries, all_advantages):
            clause_ids = [u['clause_id'] for u in structure['reference_units']]
            combined, prefix = decay_character_advantages(raw, clause_ids)
            frame = project_to_decisions(combined, support)
            direct_frame = project_to_decisions(raw, support)
            prefix_frame = project_to_decisions(prefix, support)
            profile = dict(schema_version=1, mode=MODE, full_group_size=expected,
                           fingerprint=_fingerprint(ep), ambiguous=cfg['ambiguous'],
                           reference_chars=list(ep.asr_alignment['reference_chars']),
                           clause_ids=clause_ids, edit_counts=cost,
                           character_rewards=[-x for x in cost], group_means=means,
                           group_stds=stds, raw_advantages=raw,
                           prefix_advantages=prefix, character_advantages=combined,
                           decision_advantages=frame, direct_decision_advantages=direct_frame,
                           prefix_decision_advantages=prefix_frame, insertion_anchors=anchors,
                           unsupported_characters=sum(not any(row) for row in support),
                           shared_word_characters=sum(u['granularity'] == 'shared_word_interval'
                                                      for u in structure['reference_units']),
                           surrogate_characters=sum(u['assignment'] not in ('acoustic_interval', 'unlocated')
                                                    for u in structure['reference_units']),
                           selection_score=math.fsum(raw) / len(raw))
            output[index] = replace(ep, token_cer_advantage=profile)
            signal = signal or (cfg['weight'] > 0 and any(x != 0.0 for x in frame))
    return output, signal


def frozen_profile(ep, conf):
    from .token_cer import settings
    cfg = settings(conf)
    p = ep.token_cer_advantage
    if not isinstance(p, dict) or p.get('mode') != MODE or p.get('schema_version') != 1:
        raise ValueError('Missing full-group character CER profile; normalize the full group first')
    if p.get('full_group_size') != conf.get('num_answers_per_question'):
        raise ValueError('Character CER full-group size changed after normalization')
    if p.get('ambiguous') != cfg['ambiguous'] or p.get('fingerprint') != _fingerprint(ep):
        raise ValueError('Character CER source changed after full-group normalization')
    values = p.get('decision_advantages')
    if not isinstance(values, list) or len(values) != len(ep.generated_token_ids) or not all(
            isinstance(x, (int, float)) and math.isfinite(x) for x in values):
        raise ValueError('Invalid frozen character CER decision vector')
    return p


def apply_character_cer(episodes, conf):
    from .token_cer import settings
    cfg = settings(conf)
    output, deltas, raw_values, prefix_values = [], [], [], []
    outcome_values, direct_values, propagated_values = [], [], []
    clipped = unsupported = shared = surrogates = flat = 0
    for ep in episodes:
        p = frozen_profile(ep, conf)
        values = p['decision_advantages']
        base = ([float(ep.reward)] * len(values) if isinstance(ep.reward, (int, float))
                else list(ep.reward))
        if len(base) != len(values) or any(not math.isfinite(x) for x in base):
            raise ValueError('Invalid scalar component advantage vector')
        delta = [cfg['weight'] * x for x in values]
        cap = cfg['advantage_clip']
        if cap is not None:
            clipped += sum(abs(x) > cap for x in delta)
            delta = [max(-cap, min(cap, x)) for x in delta]
        direct = [cfg['weight'] * x for x in p['direct_decision_advantages']]
        propagated = [cfg['weight'] * x for x in p['prefix_decision_advantages']]
        for i, (a, b, actual) in enumerate(zip(direct, propagated, delta)):
            total = a + b
            if abs(total - cfg['weight'] * values[i]) > 1e-8:
                raise ValueError('Direct and propagated frame terms do not sum to CER')
            if cap is not None and total != 0:
                factor = actual / total
                direct[i] *= factor
                propagated[i] *= factor
        outcome_values.extend(base)
        direct_values.extend(direct)
        propagated_values.extend(propagated)
        output.append(replace(ep, reward=[a + b for a, b in zip(base, delta)]))
        deltas.extend(delta)
        raw_values.extend(p['raw_advantages'])
        prefix_values.extend(p['prefix_advantages'])
        flat += sum(x == 0.0 for x in p['group_stds'])
        unsupported += p['unsupported_characters']
        shared += p['shared_word_characters']
        surrogates += p['surrogate_characters']
    from .token_cer import scalar_weights
    metrics = dict(enabled=1., char_exp_enabled=1.,
                   global_cer_objective_weight=scalar_weights(conf).get('cer_reward', 0.),
                   candidates=float(len(output)), full_group_size=float(conf['num_answers_per_question']),
                   clipped_decisions=float(clipped), unsupported_characters=float(unsupported),
                   shared_word_characters=float(shared), surrogate_characters=float(surrogates),
                   zero_std_characters=float(flat),
                   max_abs_delta=max(map(abs, deltas), default=0.),
                   positive_decisions=float(sum(x > 0 for x in deltas)),
                   negative_decisions=float(sum(x < 0 for x in deltas)),
                   mean_raw_advantage=math.fsum(raw_values) / max(1, len(raw_values)),
                   mean_prefix_advantage=math.fsum(prefix_values) / max(1, len(prefix_values)))
    result = {'token_cer/' + k: v for k, v in metrics.items()}
    result.update(reward_scale_metrics(outcome_values, deltas, direct_values, propagated_values))
    return output, result


def reward_scale_metrics(outcome, process, direct, prefix):
    """Telemetry only: use the same decision axis and actual weighted increments.

    This does not estimate component gradient norms. In particular, signed
    means and raw SIM scores are not measures of the two objectives' strength.
    """
    n = len(process)
    if not n or any(len(x) != n for x in (outcome, direct, prefix)):
        raise ValueError('Reward scale telemetry requires equal nonempty vectors')
    result = {}
    for name, values in [('outcome', outcome), ('process', process),
                         ('process_direct', direct), ('process_prefix', prefix),
                         ('combined', [a+b for a,b in zip(outcome, process)])]:
        if any(not math.isfinite(x) for x in values):
            raise ValueError('Non-finite reward scale telemetry input')
        mean = math.fsum(values) / n
        result.update({f'reward_scale/{name}/{key}': value for key, value in dict(
            mean=mean, abs_mean=math.fsum(map(abs, values))/n,
            rms=math.sqrt(math.fsum(x*x for x in values)/n),
            std=math.sqrt(math.fsum((x-mean)**2 for x in values)/n),
            max_abs=max(map(abs, values)), positive_fraction=sum(x>0 for x in values)/n,
            negative_fraction=sum(x<0 for x in values)/n).items()})
    outcome_rms = result['reward_scale/outcome/rms']
    result['reward_scale/outcome_rms_zero'] = float(outcome_rms <= EPS)
    if outcome_rms > EPS:
        result['reward_scale/process_to_outcome_rms'] = result['reward_scale/process/rms'] / outcome_rms
    result['reward_scale/process_larger_fraction'] = sum(abs(a)>abs(b) for a,b in zip(process,outcome))/n
    result['reward_scale/opposite_sign_fraction'] = sum(a*b<0 for a,b in zip(process,outcome))/n
    result['reward_scale/outcome_sign_flip_fraction'] = sum(b*(a+b)<0 for a,b in zip(process,outcome))/n
    result['reward_scale/decision_count'] = float(n)
    return result
