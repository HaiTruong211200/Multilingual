from dataclasses import dataclass


@dataclass
class BridgeConfig:
    nllb_name: str = "facebook/nllb-200-distilled-600M"
    qwen_name: str = "Qwen/Qwen2.5-0.5B"
    adapter_bottleneck: int = 256
    bridge_type: str = "qformer"
    qformer_hidden: int = 768
    qformer_layers: int = 4
    qformer_heads: int = 8
    qformer_queries: int = 32
    qformer_ffn_ratio: int = 4
    dropout: float = 0.1
    contrastive_weight: float = 0.1
    ot_weight: float = 0.05
    temperature: float = 0.07
    sinkhorn_epsilon: float = 0.1
    sinkhorn_iterations: int = 20
    ot_layers: tuple[int, ...] = (-1,)
    prompt: str = "Answer the question based on the provided representation.\nAnswer:"

    def __post_init__(self) -> None:
        if self.bridge_type not in {"qformer", "transformer"}:
            raise ValueError("bridge_type must be 'qformer' or 'transformer'")
