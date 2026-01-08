"""
Dataclass for model configuration parameters.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelConfig:
    name: str = "NeckNet"
    """Model name"""

    x_dim: Tuple[int, ...] = (6,)
    """Input dimension"""

    y_dim: Tuple[int, ...] = (6, 1800)
    """Output dimension"""

    h1_dim: Tuple[int, ...] = (128, 1024)
    """Hidden layer 1 dimension"""

    h2_dim: Tuple[int, ...] = (128, 1024)
    """Hidden layer 2 dimension"""
