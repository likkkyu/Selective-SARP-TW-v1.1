"""Candidate common decoder fix, not yet enabled in training.

Suppress a depot self-loop only when a different action has finite log-probability.
Apply identically to all ablation arms; preserve original terminal fallback.
"""
import torch

def suppress_depot_noop(log_p, at_depot):
    active = at_depot.reshape(-1, 1) & torch.isfinite(log_p[..., 1:]).any(-1)
    result = log_p.clone()
    result[..., 0] = result[..., 0].masked_fill(active, -torch.inf)
    return result - torch.logsumexp(result, dim=-1, keepdim=True)


def install():
    from nets.attention_model import AttentionModel
    if hasattr(AttentionModel, '_before_noop_fix'):
        return
    AttentionModel._before_noop_fix = AttentionModel._get_log_p
    def controlled(self, fixed, state, *args, **kwargs):
        if kwargs.get('normalize', True) is not True:
            raise ValueError('No-op fix requires normalized decoder probabilities')
        log_p, mask, debug = self._before_noop_fix(fixed, state, *args, **kwargs)
        return suppress_depot_noop(log_p, state.prev_a.squeeze(-1) == 0), mask, debug
    AttentionModel._get_log_p = controlled
