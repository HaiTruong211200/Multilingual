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
        contrastive_forward_mode: str = "joint",
        ot_forward_mode: str = "joint",
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
        if contrastive_forward_mode not in {"joint", "independent"}:
            raise ValueError(
                "contrastive_forward_mode must be 'joint' or 'independent'"
            )
        if ot_forward_mode not in {"joint", "independent", "bidirectional"}:
            raise ValueError(
                "ot_forward_mode must be 'joint', 'independent', or 'bidirectional'"
            )
        self.contrastive_forward_mode = contrastive_forward_mode
        self.ot_forward_mode = ot_forward_mode
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
        with torch.autocast(device_type=src.device.type, enabled=False):
            a = source_mass.float().masked_fill(~src_mask.bool(), 0.0)
            b = target_mass.float().masked_fill(~tgt_mask.bool(), 0.0)

            a_sum = a.sum(-1, keepdim=True)
            b_sum = b.sum(-1, keepdim=True)
            valid = (a_sum[:, 0] > 1e-8) & (b_sum[:, 0] > 1e-8)

            if not valid.any():
                return (
                    src.float().reshape(-1)[:0].sum()
                    + tgt.float().reshape(-1)[:0].sum()
                    + a.reshape(-1)[:0].sum()
                    + b.reshape(-1)[:0].sum()
                )

            # Loại mẫu rỗng và xác định token có mass dương.
            a = a[valid] / a_sum[valid]
            b = b[valid] / b_sum[valid]
            sm, tm = a > 0, b > 0

            x = src[valid].float().masked_fill(~sm.unsqueeze(-1), 0.0)
            y = tgt[valid].float().masked_fill(~tm.unsqueeze(-1), 0.0)
            x = F.normalize(x, dim=-1, eps=1e-6)
            y = F.normalize(y, dim=-1, eps=1e-6)
            cost = (1.0 - torch.bmm(x, y.transpose(1, 2))).clamp(0, 2)

            log_K = -cost / self.sinkhorn_epsilon
            log_a = a.masked_fill(~sm, 1.0).log()
            log_b = b.masked_fill(~tm, 1.0).log()
            log_u, log_v = torch.zeros_like(a), torch.zeros_like(b)

            for _ in range(self.sinkhorn_iterations):
                # Chỉ mask chiều lấy tổng: không có hàng toàn -inf.
                scores = (log_K + log_v.unsqueeze(1)).masked_fill(
                    ~tm.unsqueeze(1), float("-inf")
                )
                log_u = (log_a - torch.logsumexp(scores, dim=-1)).masked_fill(
                    ~sm, 0.0
                )

                scores = (log_K + log_u.unsqueeze(-1)).masked_fill(
                    ~sm.unsqueeze(-1), float("-inf")
                )
                log_v = (log_b - torch.logsumexp(scores, dim=1)).masked_fill(
                    ~tm, 0.0
                )

            pair_mask = sm.unsqueeze(-1) & tm.unsqueeze(1)
            log_P = log_u.unsqueeze(-1) + log_K + log_v.unsqueeze(1)
            transport = log_P.masked_fill(~pair_mask, float("-inf")).exp()

            return (transport * cost).sum(dim=(1, 2)).mean()

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

        Fully batched (no python loop over B) and numerically stable: all dual
        updates are done in log-space via logsumexp, which stays stable even for
        small beta / bf16-fp16 upstream tensors (unlike exp(-C/beta) followed by
        clamping, which under/overflows easily).
        """
        B, Ls, _ = src.shape
        Lt = tgt.shape[1]
        device = src.device

        src_mask = src_mask.bool()
        tgt_mask = tgt_mask.bool()

        # --- cosine cost, computed in fp32 regardless of upstream dtype ---
        x = F.normalize(src.float(), dim=-1, eps=1e-8)
        y = F.normalize(tgt.float(), dim=-1, eps=1e-8)
        cost = 1.0 - torch.bmm(x, y.transpose(1, 2))  # [B, Ls, Lt]
        cost = torch.nan_to_num(cost, nan=0.0, posinf=1e4, neginf=-1e4)

        # --- marginals: mask out padding, guard fully-empty samples ---
        a = source_mass.float() * src_mask
        b = target_mass.float() * tgt_mask
        a_sum = a.sum(dim=-1, keepdim=True)
        b_sum = b.sum(dim=-1, keepdim=True)
        valid = (a_sum.squeeze(-1) > 1e-8) & (b_sum.squeeze(-1) > 1e-8)  # [B]
        a = a / a_sum.clamp_min(1e-8)
        b = b / b_sum.clamp_min(1e-8)

        # log(0) = -inf for padded/empty entries — the correct signal to exclude
        # them from logsumexp, so don't clamp before the log.
        log_a = torch.where(a > 0, torch.log(a.clamp_min(1e-38)), a.new_full((), float("-inf")))
        log_b = torch.where(b > 0, torch.log(b.clamp_min(1e-38)), b.new_full((), float("-inf")))

        log_proximal_kernel = -cost / self.ipot_beta
        log_proximal_kernel = log_proximal_kernel.masked_fill(~src_mask.unsqueeze(-1), float("-inf"))
        log_proximal_kernel = log_proximal_kernel.masked_fill(~tgt_mask.unsqueeze(1), float("-inf"))

        # Feasible independent coupling to start from: log(a_i * b_j) = log_a_i + log_b_j.
        # Padding is already -inf via log_a/log_b, no extra masking needed here.
        log_transport = log_a.unsqueeze(-1) + log_b.unsqueeze(1)  # [B, Ls, Lt]

        log_v = torch.zeros(B, Lt, device=device, dtype=torch.float32)
        for _ in range(self.ipot_iterations):
            log_q = log_proximal_kernel + log_transport
            for _ in range(self.ipot_inner_iterations):
                log_u = log_a - torch.logsumexp(log_q + log_v.unsqueeze(1), dim=-1)
                log_u = torch.nan_to_num(
                    log_u, nan=float("-inf"), posinf=float("-inf"), neginf=float("-inf")
                )
                log_v = log_b - torch.logsumexp(
                    log_q.transpose(-1, -2) + log_u.unsqueeze(1), dim=-1
                )
                log_v = torch.nan_to_num(
                    log_v, nan=float("-inf"), posinf=float("-inf"), neginf=float("-inf")
                )
            log_transport = log_u.unsqueeze(-1) + log_q + log_v.unsqueeze(1)
            log_transport = torch.nan_to_num(log_transport, nan=float("-inf"), posinf=float("-inf"))

        transport = torch.exp(log_transport)
        transport = torch.nan_to_num(transport, nan=0.0, posinf=0.0, neginf=0.0)
        # Fully-empty samples: force the plan to exactly zero rather than trust
        # whatever fell out of the -inf/-inf arithmetic above.
        transport = transport * valid.view(B, 1, 1).to(transport.dtype)

        per_sample_cost = (transport * cost).sum(dim=(1, 2))  # [B]
        if not valid.any():
            return src.new_zeros(())
        return per_sample_cost[valid].mean()

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
        alignment_source_input_ids: Optional[torch.Tensor] = None,
        alignment_source_attention_mask: Optional[torch.Tensor] = None,
        alignment_source_content_mask: Optional[torch.Tensor] = None,
        alignment_target_input_ids: Optional[torch.Tensor] = None,
        alignment_target_attention_mask: Optional[torch.Tensor] = None,
        alignment_target_content_mask: Optional[torch.Tensor] = None,
        reverse_input_ids: Optional[torch.Tensor] = None,
        reverse_attention_mask: Optional[torch.Tensor] = None,
        reverse_target_start_positions: Optional[torch.Tensor] = None,
        reverse_target_end_positions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # ------------------------------------------------------------------
        # Block 1: Decide which representations are actually required.
        #
        # The full prompt forward is always needed for NTP. Hidden states from
        # that forward are retained only when CL uses joint or OT uses
        # joint/bidirectional. Source-only and target-only forwards are shared
        # when both CL and OT choose independent.
        # ------------------------------------------------------------------
        compute_contrastive = self.contrastive_weight != 0.0
        compute_ot = self.ot_weight != 0.0
        need_joint = (
            compute_contrastive and self.contrastive_forward_mode == "joint"
        ) or (
            compute_ot and self.ot_forward_mode in {"joint", "bidirectional"}
        )
        need_independent = (
            compute_contrastive and self.contrastive_forward_mode == "independent"
        ) or (
            compute_ot and self.ot_forward_mode == "independent"
        )
        # ------------------------------------------------------------------
        # Block 2: Main causal-LM forward over (instruction, source, target).
        # This is the only forward that receives labels and therefore the only
        # one contributing NTP loss.
        # ------------------------------------------------------------------
        output = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=need_joint,
            output_attentions=compute_ot
            and self.ot_forward_mode in {"joint", "bidirectional"},
            return_dict=True,
        )
        if source_start_positions is None or target_start_positions is None:
            return output

        ntp_loss = output.loss
        contrastive = ntp_loss.new_zeros(())
        ot = ntp_loss.new_zeros(())

        attention_layer = (
            self.align_layer if self.align_layer < 0 else max(self.align_layer - 1, 0)
        )

        # ------------------------------------------------------------------
        # Block 3: Build source/target masks inside the full prompt.
        # Intervals are [start, end), so EOS and prompt markers are excluded.
        # These tensors represent H_x|prompt and H_y|x for joint objectives.
        # ------------------------------------------------------------------
        joint_source_mask = joint_target_mask = None
        if need_joint:
            # Use the configured alignment layer for both representation
            # objectives. hidden_states[-1] is only correct when align_layer=-1.
            joint_alignment_hidden = output.hidden_states[self.align_layer]
            positions = torch.arange(
                joint_alignment_hidden.size(1), device=joint_alignment_hidden.device
            ).unsqueeze(0)
            source_start_positions = source_start_positions.to(
                joint_alignment_hidden.device
            )
            source_end_positions = source_end_positions.to(joint_alignment_hidden.device)
            target_start_positions = target_start_positions.to(
                joint_alignment_hidden.device
            )
            target_end_positions = target_end_positions.to(joint_alignment_hidden.device)
            joint_source_mask = (positions >= source_start_positions[:, None]) & (
                positions < source_end_positions[:, None]
            )
            joint_target_mask = (positions >= target_start_positions[:, None]) & (
                positions < target_end_positions[:, None]
            )

        source_output = target_output = None
        source_mask = target_mask = None
        if need_independent:
            # --------------------------------------------------------------
            # Block 4: Obtain context-independent H_x and H_y.
            # prepare_data tokenized both complete sentences beforehand and
            # the collator padded them independently. content_mask removes
            # BOS/EOS/PAD from pooling and token-level OT.
            # --------------------------------------------------------------
            independent_tensors = (
                alignment_source_input_ids,
                alignment_source_attention_mask,
                alignment_source_content_mask,
                alignment_target_input_ids,
                alignment_target_attention_mask,
                alignment_target_content_mask,
            )
            if any(tensor is None for tensor in independent_tensors):
                raise ValueError(
                    "independent contrastive/OT requires source/target input_ids, "
                    "attention_mask, and content_mask from the collator"
                )
            source_output = self.lm(
                input_ids=alignment_source_input_ids,
                attention_mask=alignment_source_attention_mask,
                output_hidden_states=True,
                output_attentions=compute_ot and self.ot_forward_mode == "independent",
                return_dict=True,
            )
            target_output = self.lm(
                input_ids=alignment_target_input_ids,
                attention_mask=alignment_target_attention_mask,
                output_hidden_states=True,
                output_attentions=compute_ot and self.ot_forward_mode == "independent",
                return_dict=True,
            )
            source_alignment_hidden = source_output.hidden_states[self.align_layer]
            target_alignment_hidden = target_output.hidden_states[self.align_layer]
            source_mask = alignment_source_content_mask.to(
                device=source_alignment_hidden.device, dtype=torch.bool
            )
            target_mask = alignment_target_content_mask.to(
                device=target_alignment_hidden.device, dtype=torch.bool
            )

        # ------------------------------------------------------------------
        # Block 5: Contrastive objective has exactly two supported views.
        # - joint:       mean-pool source/target spans from the main prompt.
        # - independent: mean-pool two separately encoded sentences.
        # It deliberately has no bidirectional conditional mode.
        # ------------------------------------------------------------------
        if compute_contrastive:
            if self.contrastive_forward_mode == "joint":
                contrastive_hidden = output.hidden_states[self.align_layer]
                contrastive = self._contrastive(
                    masked_mean(
                        contrastive_hidden,
                        joint_source_mask.to(contrastive_hidden.device),
                    ),
                    masked_mean(
                        contrastive_hidden,
                        joint_target_mask.to(contrastive_hidden.device),
                    ),
                )
            else:
                source_contrastive = source_output.hidden_states[self.align_layer]
                target_contrastive = target_output.hidden_states[self.align_layer]
                contrastive = self._contrastive(
                    masked_mean(
                        source_contrastive,
                        source_mask.to(source_contrastive.device),
                    ),
                    masked_mean(
                        target_contrastive,
                        target_mask.to(target_contrastive.device),
                    ),
                )

        # ------------------------------------------------------------------
        # Block 6: Token-level OT routing.
        # - joint:         OT(H_x from the prompt, H_y|x).
        # - independent:   OT(H_x from source-only, H_y from target-only).
        # - bidirectional: OT(H_y|x, H_x|y), requiring one reverse prompt.
        #
        # Marginals mix attention salience with a uniform distribution. The
        # transport cost itself is cosine distance inside _optimal_transport.
        # ------------------------------------------------------------------
        if compute_ot:
            if self.ot_forward_mode == "joint":
                # Target queries indicate how much attention each source and
                # target token receives inside the forward translation prompt.
                attention = output.attentions[attention_layer].mean(dim=1)
                attention_source_mask = joint_source_mask.to(attention.device)
                attention_target_mask = joint_target_mask.to(attention.device)
                received_attention = (
                    attention.float() * attention_target_mask.unsqueeze(-1)
                ).sum(dim=1)
                source_mass = self._mixed_mass(
                    received_attention * attention_source_mask,
                    attention_source_mask,
                )
                target_mass = self._mixed_mass(
                    received_attention * attention_target_mask,
                    attention_target_mask,
                )
                ot = self._optimal_transport(
                    joint_alignment_hidden,
                    joint_alignment_hidden,
                    joint_source_mask,
                    joint_target_mask,
                    source_mass.to(joint_alignment_hidden.device),
                    target_mass.to(joint_alignment_hidden.device),
                )
            elif self.ot_forward_mode == "independent":
                # No cross-sentence attention exists here. Each marginal is
                # therefore derived from self-attention in its own sequence.
                source_attention = source_output.attentions[attention_layer].mean(dim=1)
                target_attention = target_output.attentions[attention_layer].mean(dim=1)
                source_attention_mask = source_mask.to(source_attention.device)
                target_attention_mask = target_mask.to(target_attention.device)
                source_received = (
                    source_attention.float()
                    * source_attention_mask.unsqueeze(-1)
                ).sum(dim=1)
                target_received = (
                    target_attention.float()
                    * target_attention_mask.unsqueeze(-1)
                ).sum(dim=1)
                source_mass = self._mixed_mass(
                    source_received * source_attention_mask,
                    source_attention_mask,
                )
                target_mass = self._mixed_mass(
                    target_received * target_attention_mask,
                    target_attention_mask,
                )
                ot = self._optimal_transport(
                    source_alignment_hidden,
                    target_alignment_hidden.to(source_alignment_hidden.device),
                    source_mask,
                    target_mask.to(source_alignment_hidden.device),
                    source_mass.to(source_alignment_hidden.device),
                    target_mass.to(source_alignment_hidden.device),
                )
            else:
                # Reverse prompt is (instruction target->source, target, source).
                # Its target span is x, yielding H_x|y; the forward target span
                # yields H_y|x. Reverse labels are intentionally not supplied.
                reverse_tensors = (
                    reverse_input_ids,
                    reverse_attention_mask,
                    reverse_target_start_positions,
                    reverse_target_end_positions,
                )
                if any(tensor is None for tensor in reverse_tensors):
                    raise ValueError(
                        "bidirectional OT requires the reversed prompt and its "
                        "target span from the collator"
                    )
                reverse_output = self.lm(
                    input_ids=reverse_input_ids,
                    attention_mask=reverse_attention_mask,
                    output_hidden_states=True,
                    output_attentions=True,
                    return_dict=True,
                )
                reverse_alignment_hidden = reverse_output.hidden_states[self.align_layer]
                reverse_positions = torch.arange(
                    reverse_alignment_hidden.size(1),
                    device=reverse_alignment_hidden.device,
                ).unsqueeze(0)
                reverse_target_mask = (
                    reverse_positions
                    >= reverse_target_start_positions.to(
                        reverse_alignment_hidden.device
                    )[:, None]
                ) & (
                    reverse_positions
                    < reverse_target_end_positions.to(
                        reverse_alignment_hidden.device
                    )[:, None]
                )
                # Attention mass for each side comes from target queries in its
                # own directional prompt; instruction/source/EOS remain masked
                # out of the transport marginals.
                forward_attention = output.attentions[attention_layer].mean(dim=1)
                reverse_attention = reverse_output.attentions[attention_layer].mean(dim=1)
                forward_attention_mask = joint_target_mask.to(forward_attention.device)
                reverse_attention_mask = reverse_target_mask.to(reverse_attention.device)
                forward_received = (
                    forward_attention.float()
                    * forward_attention_mask.unsqueeze(-1)
                ).sum(dim=1)
                reverse_received = (
                    reverse_attention.float()
                    * reverse_attention_mask.unsqueeze(-1)
                ).sum(dim=1)
                forward_mass = self._mixed_mass(
                    forward_received * forward_attention_mask,
                    forward_attention_mask,
                )
                reverse_mass = self._mixed_mass(
                    reverse_received * reverse_attention_mask,
                    reverse_attention_mask,
                )
                ot = self._optimal_transport(
                    joint_alignment_hidden,
                    reverse_alignment_hidden.to(joint_alignment_hidden.device),
                    joint_target_mask,
                    reverse_target_mask.to(joint_alignment_hidden.device),
                    forward_mass.to(joint_alignment_hidden.device),
                    reverse_mass.to(joint_alignment_hidden.device),
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
