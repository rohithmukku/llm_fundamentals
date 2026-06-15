from dataclasses import dataclass

@dataclass
class TrainerConfig:
    seed: int = 8
    steps: int = 1000
    batch_size: int = 32
    lr: float = 3e-4
    grad_norm_clip: float = 1.0
    log_step: int = 100

@dataclass
class ModelConfig:
    n_embed: int = 512
    n_heads: int = 12
    n_layers: int = 12
    max_seq_len: int = 1024
    dropout: float = 0.2
    bias: bool = False
    use_torch_dot_product: bool = True
    use_rope: bool = True