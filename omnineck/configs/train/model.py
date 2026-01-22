"""
Dataclass for model configuration parameters.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    
    name: str = "NeckNet"
    """Model name"""

    x_dim: Tuple[int, ...] = (6,)
    """Input dimension"""

    y_dim: Tuple[int, ...] = (6, 2862)
    """Output dimension"""

    hidden_dim: Tuple[Tuple[int, ...], ...] = (
        (512, 512),
        (1024, 1024),
    )
    """Hidden layer dimensions for each part of the model."""
