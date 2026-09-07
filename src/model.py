"""Causal LM augmented with sentence contrastive and token-level OT losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class AlignmentCausalLMOutputWithPast(CausalLMOutputWithPast):
    """Standard causal-LM output plus detached alignment logging values."""

    model_total_loss: Optional[torch.FloatTensor] = None
    ntp_loss: Optional[torch.FloatTensor] = None
    contrastive_loss: Optional[torch.FloatTensor] = None
    ot_loss: Optional[torch.FloatTensor] = None
    weighted_contrastive_loss: Optional[torch.FloatTensor] = None
    weighted_ot_loss: Optional[torch.FloatTensor] = None


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)


class MultilingualAlignmentModel(nn.Module):
    """A shared causal LM trained with NTP + InfoNCE + token-level OT."""

    def __init__(
        self,
        model_name_or_path: str,
        contrastive_weight: float = 0.0,
        ot_weight: float = 0.0,
        temperature: float = 0.07,
        align_layer: int = -1,
        attention_mass_weight: float = 0.5,
        ot_solver: str = "sinkhorn",
        sinkhorn_epsilon: float = 0.1,
        sinkhorn_iterations: int = 20,
        ipot_beta: float = 0.5,
        ipot_iterations: int = 50,
        ipot_inner_iterations: int = 1,
        attn_implementation: str = "eager",
        trust_remote_code: bool = False,
    ):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )
        self.config = self.lm.config
        self.contrastive_weight = contrastive_weight
        self.ot_weight = ot_weight
        self.temperature = temperature
        self.align_layer = align_layer
        if not 0.0 <= attention_mass_weight <= 1.0:
            raise ValueError("attention_mass_weight must be between 0 and 1")
        self.attention_mass_weight = attention_mass_weight
        if ot_solver not in {"sinkhorn", "ipot"}:
            raise ValueError("ot_solver must be 'sinkhorn' or 'ipot'")
        if sinkhorn_epsilon <= 0.0 or ipot_beta <= 0.0:
            raise ValueError("sinkhorn_epsilon and ipot_beta must be positive")
        if sinkhorn_iterations < 1 or ipot_iterations < 1 or ipot_inner_iterations < 1:
            raise ValueError("all OT iteration counts must be positive")
        self.ot_solver = ot_solver
        self.initial_sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        self.ipot_beta = ipot_beta
        self.ipot_iterations = ipot_iterations
        self.ipot_inner_iterations = ipot_inner_iterations

    def gradient_checkpointing_enable(self, **kwargs):
        return self.lm.gradient_checkpointing_enable(**kwargs)

    def _contrastive(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Within-instruction contrastive loss following the reference Llama model.

        Rows/columns contain all in-batch source-target pairs; diagonal entries
        are translations and off-diagonal entries are negatives. The reference
        applies LogSoftmax over dim=0 and scales the mean diagonal NLL by 1/2.
        """
        if src.size(0) <= 1:
            return src.new_zeros(())
        similarity = F.cosine_similarity(src[:, None, :], tgt[None, :, :], dim=-1)
        log_prob = F.log_softmax(similarity / self.temperature, dim=0)
        return -torch.diagonal(log_prob).mean() / 2.0

    def _mixed_mass(
        self, attention_scores: torch.Tensor, span_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mix normalized attention salience with a uniform span distribution."""
        mask = span_mask.to(dtype=attention_scores.dtype)
        token_count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        uniform_mass = mask / token_count

        masked_scores = attention_scores * mask
        score_sum = masked_scores.sum(dim=1, keepdim=True)
        attention_mass = masked_scores / score_sum.clamp_min(1e-8)

        # A backend/layer can exceptionally return zero attention for a span.
        # In that case use uniform instead of producing a zero marginal.
        attention_mass = torch.where(
            score_sum > 1e-8,
            attention_mass,
            uniform_mass,
        )
        alpha = self.attention_mass_weight
        return alpha * attention_mass + (1.0 - alpha) * uniform_mass

    def _sinkhorn_ot(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        source_mass: torch.Tensor,
        target_mass: torch.Tensor,
    ) -> torch.Tensor:
        """Entropic OT with cosine cost and attention-derived marginals."""
        costs = []
        for x, y, mx, my, ax, by in zip(
            src, tgt, src_mask.bool(), tgt_mask.bool(), source_mass, target_mass
        ):
            # Sinkhorn is particularly sensitive to bf16/fp16 underflow.
            x = F.normalize(x[mx].float(), dim=-1)
            y = F.normalize(y[my].float(), dim=-1)
            cost = 1.0 - x @ y.T
            # Attention scores have already been normalized over their spans.
            a = ax[mx].to(dtype=cost.dtype)
            b = by[my].to(dtype=cost.dtype)
            kernel = torch.exp(-cost / self.sinkhorn_epsilon).clamp_min(1e-8)
            u, v = torch.ones_like(a), torch.ones_like(b)
            for _ in range(self.sinkhorn_iterations):
                u = a / (kernel @ v).clamp_min(1e-8)
                v = b / (kernel.T @ u).clamp_min(1e-8)
            transport = u[:, None] * kernel * v[None, :]
            costs.append((transport * cost).sum())
        return torch.stack(costs).mean()

    def _ipot_ot(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        source_mass: torch.Tensor,
        target_mass: torch.Tensor,
    ) -> torch.Tensor:
        """IPOT with cosine cost and arbitrary attention-derived marginals.

        IPOT repeatedly solves an inexact KL-proximal transport subproblem. In
        contrast to entropic Sinkhorn, beta controls the proximal update rather
        than permanently smoothing the final transport plan.
        """
        costs = []
        for x, y, mx, my, ax, by in zip(
            src, tgt, src_mask.bool(), tgt_mask.bool(), source_mass, target_mass
        ):
            # Perform transport iterations in fp32 to avoid fp16/bf16 underflow.
            x = F.normalize(x[mx].float(), dim=-1)
            y = F.normalize(y[my].float(), dim=-1)
            cost = 1.0 - x @ y.T
            a = ax[mx].to(dtype=cost.dtype)
            b = by[my].to(dtype=cost.dtype)
            a = a / a.sum().clamp_min(1e-8)
            b = b / b.sum().clamp_min(1e-8)

            proximal_kernel = torch.exp(-cost / self.ipot_beta).clamp_min(1e-8)
            # Start from a feasible independent coupling. Each outer iteration
            # multiplies the kernel by the previous transport plan.
            transport = a[:, None] * b[None, :]
            v = torch.ones_like(b)
            for _ in range(self.ipot_iterations):
                q = proximal_kernel * transport
                for _ in range(self.ipot_inner_iterations):
                    u = a / (q @ v).clamp_min(1e-8)
                    v = b / (q.T @ u).clamp_min(1e-8)
                transport = u[:, None] * q * v[None, :]

            costs.append((transport * cost).sum())
        return torch.stack(costs).mean()

    def _optimal_transport(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        source_mass: torch.Tensor,
        target_mass: torch.Tensor,
    ) -> torch.Tensor:
        solver = self._ipot_ot if self.ot_solver == "ipot" else self._sinkhorn_ot
        return solver(src, tgt, src_mask, tgt_mask, source_mass, target_mass)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        source_start_positions: Optional[torch.Tensor] = None,
        source_end_positions: Optional[torch.Tensor] = None,
        target_start_positions: Optional[torch.Tensor] = None,
        target_end_positions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        compute_contrastive = self.contrastive_weight != 0.0
        compute_ot = self.ot_weight != 0.0
        compute_alignment = compute_contrastive or compute_ot
        output = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=compute_alignment,
            output_attentions=compute_ot,
            return_dict=True,
        )
        if source_start_positions is None or target_start_positions is None:
            return output

        ntp_loss = output.loss
        contrastive = ntp_loss.new_zeros(())
        ot = ntp_loss.new_zeros(())

        if compute_alignment:
            final_hidden = output.hidden_states[-1]
            positions = torch.arange(
                final_hidden.size(1), device=final_hidden.device
            ).unsqueeze(0)
            source_mask = (positions >= source_start_positions[:, None]) & (
                positions < source_end_positions[:, None]
            )
            target_mask = (positions >= target_start_positions[:, None]) & (
                positions < target_end_positions[:, None]
            )

            if compute_contrastive:
                contrastive_hidden = output.hidden_states[self.align_layer]
                contrastive = self._contrastive(
                    masked_mean(contrastive_hidden, source_mask),
                    masked_mean(contrastive_hidden, target_mask),
                )

            if compute_ot:
                # hidden_states[0] is embeddings; attentions[0] is layer 1.
                attention_layer = (
                    self.align_layer
                    if self.align_layer < 0
                    else max(self.align_layer - 1, 0)
                )
                attention = output.attentions[attention_layer].mean(dim=1)
                received_attention = (
                    attention.float() * target_mask.unsqueeze(-1)
                ).sum(dim=1)
                source_mass = self._mixed_mass(
                    received_attention * source_mask, source_mask
                )
                target_mass = self._mixed_mass(
                    received_attention * target_mask, target_mask
                )
                ot = self._optimal_transport(
                    final_hidden, final_hidden, source_mask, target_mask,
                    source_mass, target_mass,
                )
        # Usually all three losses are already colocated. Explicit movement is
        # required for model/tensor parallel layouts where alignment hidden
        # states and the LM head may live on different devices.
        contrastive = contrastive.to(ntp_loss.device)
        ot = ot.to(ntp_loss.device)
        weighted_contrastive = self.contrastive_weight * contrastive
        weighted_ot = self.ot_weight * ot
        total_loss = ntp_loss + weighted_contrastive + weighted_ot
        return AlignmentCausalLMOutputWithPast(
            loss=total_loss,
            logits=output.logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            model_total_loss=total_loss.detach(),
            ntp_loss=ntp_loss.detach(),
            contrastive_loss=contrastive.detach(),
            ot_loss=ot.detach(),
            weighted_contrastive_loss=weighted_contrastive.detach(),
            weighted_ot_loss=weighted_ot.detach(),
        )
