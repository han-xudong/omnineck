"""
Dataclass for training configuration parameters.
"""

from dataclasses import dataclass, field
from .data import DataConfig
from .model import ModelConfig


@dataclass
class TrainConfig:
    batch_size: int = 128
    """Batch size for training."""

    lr: float = 1e-5
    """Learning rate for the optimizer."""

    max_epochs: int = 2000
    """Maximum number of training epochs."""

    save_dir: str = "lightning_logs"
    """Directory to save training logs and checkpoints."""
    
    zero_loss_weight: float = 0.5
    """Weight for the zero-input loss component."""

    data: DataConfig = field(default_factory=DataConfig)

    model: ModelConfig = field(default_factory=ModelConfig)
