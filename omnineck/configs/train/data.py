"""
Dataclass for train configuration parameters.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DataConfig:
    dataset_path: str = "./data/omnineck/sim"
    """Path to the dataset directory."""

    num_workers: int = 4
    """Number of workers for data loading."""

    pin_memory: bool = False
    """Whether to pin memory during data loading."""

    train_val_split: Tuple[float, float] = (0.875, 0.125)
    """Train-validation split ratios."""
