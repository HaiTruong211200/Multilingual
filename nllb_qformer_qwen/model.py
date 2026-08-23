from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

from .config import BridgeConfig
from .losses import masked_mean, sinkhorn_hidden_state_ot, symmetric_info_nce
from .modules import GlobalAdapter, QFormer, TransformerBridge


@dataclass
class BridgeOutput:
    loss: torch.Tensor
    qa_loss: torch.Tensor
    global_loss: torch.Tensor
    local_loss: torch.Tensor
    logits: torch.Tensor | None = None


class NLLBQFormerQwen(nn.Module):
    def __init__(self, config: BridgeConfig, torch_dtype=None):
        super().__init__()
        self.bridge_config = config
        nllb = AutoModelForSeq2SeqLM.from_pretrained(config.nllb_name, torch_dtype=torch_dtype)
        self.nllb_encoder = nllb.get_encoder()
        self.qwen = AutoModelForCausalLM.from_pretrained(config.qwen_name, torch_dtype=torch_dtype)
        nllb_hidden = int(nllb.config.d_model)
        qwen_hidden = int(self.qwen.config.hidden_size)
        self.adapter = GlobalAdapter(nllb_hidden, config.adapter_bottleneck, config.dropout)
        if config.bridge_type == "qformer":
            self.bridge = QFormer(
                nllb_hidden, config.qformer_hidden, config.qformer_queries, config.qformer_layers,
                config.qformer_heads, config.qformer_ffn_ratio, config.dropout,
            )
        else:
            self.bridge = TransformerBridge(
                nllb_hidden, config.qformer_hidden, config.qformer_layers,
                config.qformer_heads, config.qformer_ffn_ratio, config.dropout,
            )
        self.projector = nn.Linear(config.qformer_hidden, qwen_hidden)
        self._freeze_backbones()

    def _freeze_backbones(self) -> None:
        for module in (self.nllb_encoder, self.qwen):
            module.requires_grad_(False)
            module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.nllb_encoder.eval()
        self.qwen.eval()
        return self

    def trainable_parameter_summary(self) -> tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    def forward(self, nllb_input_ids, nllb_attention_mask, prompt_input_ids, answer_input_ids, answer_attention_mask, pair_batch_size=None, **_):
        cfg = self.bridge_config
        with torch.no_grad():
            encoded = self.nllb_encoder(input_ids=nllb_input_ids, attention_mask=nllb_attention_mask, return_dict=True).last_hidden_state
        adapted = self.adapter(encoded)
        latent, bridge_states = self.bridge(adapted, nllb_attention_mask, return_hidden_states=True)
        if cfg.bridge_type == "qformer":
            # All M learned-query positions are valid.
            bridge_hidden_mask = torch.ones(latent.shape[:2], dtype=torch.bool, device=latent.device)
        else:
            # TransformerBridge preserves NLLB's padded token sequence.
            bridge_hidden_mask = nllb_attention_mask.bool()
        global_loss = adapted.new_zeros(())
        local_loss = adapted.new_zeros(())
        if cfg.contrastive_weight and pair_batch_size:
            pooled = masked_mean(adapted, nllb_attention_mask)
            global_loss = symmetric_info_nce(pooled[0::2], pooled[1::2], cfg.temperature)
        if cfg.ot_weight and pair_batch_size:
            selected = [bridge_states[index] for index in cfg.ot_layers]
            local_loss = torch.stack([
                sinkhorn_hidden_state_ot(
                    state[0::2], state[1::2],
                    bridge_hidden_mask[0::2], bridge_hidden_mask[1::2],
                    cfg.sinkhorn_epsilon, cfg.sinkhorn_iterations,
                )
                for state in selected
            ]).mean()

        latent_embeds = self.projector(latent)
        embedding = self.qwen.get_input_embeddings()
        prompt_embeds = embedding(prompt_input_ids)
        answer_embeds = embedding(answer_input_ids)
        inputs_embeds = torch.cat((latent_embeds, prompt_embeds, answer_embeds), dim=1)
        prefix_length = latent.size(1) + prompt_input_ids.size(1)
        prompt_mask = torch.ones(
            answer_attention_mask.size(0), prompt_input_ids.size(1),
            dtype=answer_attention_mask.dtype, device=answer_attention_mask.device,
        )
        prefix_mask = torch.cat((bridge_hidden_mask.to(answer_attention_mask.dtype), prompt_mask), dim=1)
        attention_mask = torch.cat((prefix_mask, answer_attention_mask), dim=1)
        ignored = torch.full(
            (answer_input_ids.size(0), prefix_length), -100, dtype=torch.long, device=answer_input_ids.device
        )
        answer_labels = answer_input_ids.masked_fill(~answer_attention_mask.bool(), -100)
        labels = torch.cat((ignored, answer_labels), dim=1)
        output = self.qwen(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, return_dict=True)
        qa_loss = output.loss
        total = qa_loss + cfg.contrastive_weight * global_loss.to(qa_loss.device) + cfg.ot_weight * local_loss.to(qa_loss.device)
        return BridgeOutput(total, qa_loss.detach(), global_loss.detach(), local_loss.detach(), output.logits)
