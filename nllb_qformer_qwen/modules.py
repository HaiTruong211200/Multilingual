from __future__ import annotations

import torch
from torch import nn


class GlobalAdapter(nn.Module):
    """Token-preserving residual bottleneck used for coarse alignment."""

    def __init__(self, hidden_size: int, bottleneck: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, hidden_size)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        delta = self.up(self.dropout(self.activation(self.down(self.norm(hidden)))))
        return hidden + self.alpha * delta


class QFormerBlock(nn.Module):
    def __init__(self, hidden: int, encoder_hidden: int, heads: int, ffn_ratio: int, dropout: float):
        super().__init__()
        self.encoder_projection = nn.Linear(encoder_hidden, hidden) if encoder_hidden != hidden else nn.Identity()
        self.self_norm = nn.LayerNorm(hidden)
        self.self_attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden)
        self.cross_attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * ffn_ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden * ffn_ratio, hidden), nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, encoder_hidden: torch.Tensor, encoder_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.self_norm(queries)
        queries = queries + self.self_attention(normalized, normalized, normalized, need_weights=False)[0]
        normalized = self.cross_norm(queries)
        memory = self.encoder_projection(encoder_hidden)
        queries = queries + self.cross_attention(
            normalized, memory, memory, key_padding_mask=~encoder_mask.bool(), need_weights=False
        )[0]
        return queries + self.ffn(self.ffn_norm(queries))


class QFormer(nn.Module):
    def __init__(self, encoder_hidden: int, hidden: int, num_queries: int, layers: int, heads: int, ffn_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(1, num_queries, hidden))
        nn.init.normal_(self.queries, std=0.02)
        self.layers = nn.ModuleList([
            QFormerBlock(hidden, encoder_hidden, heads, ffn_ratio, dropout) for _ in range(layers)
        ])
        self.output_norm = nn.LayerNorm(hidden)

    def forward(self, encoder_hidden: torch.Tensor, encoder_mask: torch.Tensor, return_hidden_states: bool = False):
        queries = self.queries.expand(encoder_hidden.size(0), -1, -1)
        states = []
        for layer in self.layers:
            queries = layer(queries, encoder_hidden, encoder_mask)
            states.append(queries)
        output = self.output_norm(queries)
        if states:
            states[-1] = output
        return (output, tuple(states)) if return_hidden_states else output


class TransformerBridge(nn.Module):
    """Token-preserving Transformer alternative to the learnable-query bridge."""

    def __init__(self, encoder_hidden: int, hidden: int, layers: int, heads: int, ffn_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(encoder_hidden, hidden) if encoder_hidden != hidden else nn.Identity()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=heads,
                dim_feedforward=hidden * ffn_ratio,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layers)
        ])
        self.output_norm = nn.LayerNorm(hidden)

    def forward(self, encoder_hidden: torch.Tensor, encoder_mask: torch.Tensor, return_hidden_states: bool = False):
        hidden = self.input_projection(encoder_hidden)
        states = []
        padding_mask = ~encoder_mask.bool()
        for layer in self.layers:
            hidden = layer(hidden, src_key_padding_mask=padding_mask)
            # Keep padded positions inert for downstream OT and projection.
            hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            states.append(hidden)
        output = self.output_norm(hidden)
        output = output.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        if states:
            states[-1] = output
        return (output, tuple(states)) if return_hidden_states else output
