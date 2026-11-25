"""Cosy reference-ratio GRPO objective adapted to M5 C0 and C1-C15 log-probs."""
import torch
from m5_tts.grpo_utils import compute_kl_loss
import torch.distributed as dist


def episode_tensors(episode, device):
    prefix = torch.tensor(episode.prefix_token_ids, dtype=torch.long, device=device)
    generated = torch.tensor(episode.generated_token_ids, dtype=torch.long, device=device)
    tokens = torch.cat([prefix, generated], dim=0)
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    # A combined continuation prefix has no audio EOS or padding. Text ID 0 is valid.
    prefix_audio = prefix[:, :16].ne(0).any(dim=-1)
    mask[:len(prefix), :16] = prefix_audio[:, None]
    mask[:len(prefix), -1] = ~prefix_audio
    mask[len(prefix):, :16] = True
    advantages = torch.zeros(len(tokens), dtype=torch.float32, device=device)
    advantages[len(prefix):] = torch.as_tensor(episode.reward, dtype=torch.float32, device=device)
    return tokens[None], mask[None], advantages[1:][mask[1:, 0]]


def clipped_objective(log_probs, ref_log_probs, advantages, low, high):
    ratio = (log_probs.float() - ref_log_probs.float()).exp()
    policy = -torch.minimum(ratio * advantages, ratio.clamp(1 - low, 1 + high) * advantages)
    return policy, compute_kl_loss(log_probs, ref_log_probs), ratio


def batch_forward_m5(model, ref_model, episodes, conf, device):
    """Sequential candidate forwards; retain the local batch graph for one backward.

    Sum the frame counts across the four ranks sharing each virtual stream.
    Scale the denominator to match the FSDP2 gradient average.
    """
    episodes = sorted(episodes, key=lambda ep: len(ep.generated_token_ids))
    frame_count = sum(len(ep.generated_token_ids) for ep in episodes)
    if frame_count == 0:
        raise ValueError('No generated frames in this local batch')
    fraction = float(conf.get('decoder_fraction', 0.125))
    keeps = []
    for ep in episodes:
        frames = len(ep.generated_token_ids) + sum(any(row[:16]) for row in ep.prefix_token_ids[1:])
        keeps.append(torch.rand(frames, device=device) < fraction)
    sampled = sum(int(keep.sum().item()) for keep in keeps)
    local_frame_count = frame_count
    local_sampled = sampled
    frame_count, depth_denom = virtual_denominators(frame_count, sampled, conf, device)
    weights = torch.linspace(1, 0.1, 16, device=device)[1:]
    weights /= weights.sum()
    low, high = float(conf['clip_eps_low']), float(conf['clip_eps_high'])
    dtype = torch.bfloat16 if conf.get('dtype') == 'bf16' else torch.float32
    total = None
    stats = {key: 0.0 for key in ('grpo_loss', 'kl_loss', 'c0_grpo_loss', 'depth_grpo_loss',
                                   'ratio_clip_low', 'ratio_clip_high')}
    collect_entropy = bool(conf.get('collect_policy_entropy', False))
    if collect_entropy:
        stats.update(_entropy_c0_sum=0., _entropy_c0_count=0,
                     _entropy_depth_sum=0., _entropy_depth_count=0)
        stats.update({f"_entropy_c{codebook}_{field}": 0.0
                      for codebook in range(1, 16) for field in ("sum", "count")})
    for ep, keep in zip(episodes, keeps):
        tokens, mask, advantage = episode_tensors(ep, device)
        with torch.autocast(device_type='cuda', dtype=dtype, enabled=dtype != torch.float32):
            actor = model(tokens, mask, decoder_fraction=fraction, decoder_frame_keep=keep,
                          collect_policy_entropy=collect_entropy)
            with torch.no_grad():
                reference = ref_model(tokens, mask, decoder_fraction=fraction, decoder_frame_keep=keep)
            c0_pg, c0_kl, ratio = clipped_objective(
                actor['c0_log_probs'], reference['c0_log_probs'], advantage, low, high)
            depth_pg, depth_kl, _ = clipped_objective(
                actor['decoder_log_probs'], reference['decoder_log_probs'],
                advantage[keep, None], low, high)
            c0_loss = c0_pg.sum() / frame_count
            depth_loss = (depth_pg * weights).sum() / depth_denom
            pg = c0_loss + depth_loss
            kl = c0_kl.sum() / frame_count + (depth_kl * weights).sum() / depth_denom
            loss = pg + float(conf['kl_weight']) * kl
        if collect_entropy:
            generated = len(ep.generated_token_ids)
            c0_entropy = actor["c0_entropy"][-generated:]
            selected_is_generated = torch.arange(len(keep), device=device)[keep] >= len(keep) - generated
            depth_entropy = actor["decoder_entropy"][selected_is_generated]
            stats["_entropy_c0_sum"] += float(c0_entropy.sum())
            stats["_entropy_c0_count"] += c0_entropy.numel()
            stats["_entropy_depth_sum"] += float(depth_entropy.sum())
            stats["_entropy_depth_count"] += depth_entropy.numel()
            # Reuse the already computed full-vocabulary entropy; no new forward/RNG.
            for codebook, subtotal in enumerate(depth_entropy.sum(dim=0).tolist(), 1):
                stats[f"_entropy_c{codebook}_sum"] += subtotal
                stats[f"_entropy_c{codebook}_count"] += depth_entropy.shape[0]
        total = loss if total is None else total + loss
        for key, value in [('grpo_loss', pg), ('kl_loss', kl), ('c0_grpo_loss', c0_loss),
                           ('depth_grpo_loss', depth_loss),
                           ('ratio_clip_low', (ratio < 1 - low).sum() / frame_count),
                           ('ratio_clip_high', (ratio > 1 + high).sum() / frame_count)]:
            stats[key] += float(value.detach())
    stats.update(local_candidates=len(episodes), local_target_frames=local_frame_count,
                 local_decoder_sampled_frames=local_sampled)
    return {'loss': total, **stats}


def virtual_denominators(frames, sampled, conf, device):
    size = int(conf.get('_virtual_subranks', 1))
    if size == 1:
        return frames, max(sampled, 1)
    counts = torch.tensor([frames, sampled], dtype=torch.float64, device=device)
    dist.all_reduce(counts, group=conf['_virtual_dp_group'])
    return float(counts[0]) / size, max(float(counts[1]), 1.) / size
