"""Read-only policy entropy measurements."""
import torch
import torch.distributed as dist


@torch.no_grad()
def categorical_entropy(logits, chunk_rows=256):
    """Full-vocabulary Shannon entropy in nats; never sampled-token surprisal."""
    flat = logits.detach().reshape(-1, logits.shape[-1])
    entropy = torch.empty(len(flat), device=logits.device, dtype=torch.float32)
    for start in range(0, len(flat), chunk_rows):
        logp = torch.log_softmax(flat[start:start + chunk_rows].float(), dim=-1)
        entropy[start:start + chunk_rows] = -(logp.exp() * logp).sum(dim=-1)
    return entropy.reshape(logits.shape[:-1])


def entropy_metrics(sums, device):
    """Per-codebook, token-weighted entropy across ranks; C1-C15 use sampled frames."""
    keys = ['_entropy_c0_sum', '_entropy_c0_count', '_entropy_depth_sum', '_entropy_depth_count']
    keys.extend(f"_entropy_c{codebook}_{field}"
                for codebook in range(1, 16) for field in ("sum", "count"))
    values = torch.tensor([sums.pop(key, 0.) for key in keys],
                          dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    totals = values.tolist()
    c0_sum, c0_count, depth_sum, depth_count = totals[:4]
    if not c0_count:
        raise ValueError('Policy entropy has no generated C0 positions')
    metrics = {'policy_entropy_c0': c0_sum / c0_count,
               'policy_entropy_c0_tokens': c0_count,
               'policy_entropy_depth_sampled_tokens': depth_count}
    if depth_count:
        depth = depth_sum / depth_count
        metrics.update(policy_entropy_depth_sampled=depth,
                       policy_entropy_codebook_mean_estimate=(metrics['policy_entropy_c0'] + 15 * depth) / 16)
    for codebook in range(1, 16):
        subtotal, count = totals[4 + 2 * (codebook - 1):6 + 2 * (codebook - 1)]
        metrics[f"policy_entropy_c{codebook}_sampled_tokens"] = count
        # Empty sampled frames are missing observations, not zero entropy.
        if count:
            metrics[f"policy_entropy_c{codebook}"] = subtotal / count
    return metrics
