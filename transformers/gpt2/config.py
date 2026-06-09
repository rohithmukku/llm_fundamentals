from dataclasses import dataclass

@dataclass
class TrainingConfig:
    steps: int = 1000
    dropout: float = 0.2

@dataclass
class ModelConfig:
    n_embed: int = 512
    n_heads: int = 12
    n_layers: int = 12
    max_seq_len: int = 1024
    dropout: float = 0.2
    bias: bool = False
    use_torch_dot_product: bool = True