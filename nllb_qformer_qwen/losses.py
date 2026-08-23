from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)


def symmetric_info_nce(left: torch.Tensor, right: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    if left.size(0) < 2:
        return left.new_zeros(())
    logits = F.normalize(left.float(), dim=-1) @ F.normalize(right.float(), dim=-1).T / temperature
    labels = torch.arange(left.size(0), device=left.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def sinkhorn_hidden_state_ot(
    left_hidden: torch.Tensor,
    right_hidden: torch.Tensor,
    left_mask: torch.Tensor,
    right_mask: torch.Tensor,
    epsilon: float = 0.1,
    iterations: int = 20,
) -> torch.Tensor:
    """Masked OT over a selected Q-Former layer's output hidden states.

    Each batch item is transported independently. Invalid/padded positions are
    removed before constructing its cost matrix, so they receive exactly zero
    marginal mass and cannot affect either Sinkhorn updates or the batch mean.
    """
    if left_hidden.ndim != 3 or right_hidden.ndim != 3:
        raise ValueError("hidden states must have shape [batch, sequence, hidden]")
    if left_hidden.size(0) != right_hidden.size(0):
        raise ValueError("left and right hidden-state batches must have equal size")
    if left_mask.shape != left_hidden.shape[:2] or right_mask.shape != right_hidden.shape[:2]:
        raise ValueError("each mask must have shape [batch, sequence]")

    sample_costs = []
    for left, right, valid_left, valid_right in zip(
        left_hidden, right_hidden, left_mask.bool(), right_mask.bool()
    ):
        if not valid_left.any() or not valid_right.any():
            raise ValueError("every sample must contain at least one valid hidden-state position")
        left = F.normalize(left[valid_left].float(), dim=-1)
        right = F.normalize(right[valid_right].float(), dim=-1)
        cost = 1.0 - left @ right.T
        n, m = cost.shape
        source_mass = cost.new_full((n,), 1.0 / n)
        target_mass = cost.new_full((m,), 1.0 / m)
        kernel = torch.exp(-cost / epsilon).clamp_min(1e-8)
        u, v = torch.ones_like(source_mass), torch.ones_like(target_mass)
        for _ in range(iterations):
            u = source_mass / (kernel @ v).clamp_min(1e-8)
            v = target_mass / (kernel.T @ u).clamp_min(1e-8)
        transport = u[:, None] * kernel * v[None, :]
        sample_costs.append((transport * cost).sum())
    return torch.stack(sample_costs).mean()
