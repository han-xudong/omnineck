"""
Dataclass for camera configuration parameters.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class CameraConfig:
    id: int = 0
    """OpenCV camera ID."""

    width: int = 640
    """Frame width."""

    height: int = 480
    """Frame height."""

    fps: int = 60
    """Frames per second."""

    dist: Tuple[Tuple[float, ...], ...] = ((-0.150044, 0.051017, 0.000000, 0.000000, -0.067817),)
    """Distortion coefficients."""

    mtx: Tuple[Tuple[float, ...], ...] = (
        (-180.971415, 0.000000, 148.729196),
        (0.000000, 180.598787, 121.966872),
        (0.000000, 0.000000, 1.000000),
    )
    """Camera matrix."""

    filter_on: bool = False
    """Enable pose filtering."""

    filter_frame: int = 5
    """Number of frames for pose filtering."""

    marker_size: float = 0.012
    """Marker size in meters."""
    
    marker_num: int = 1
    """Number of markers."""

    transfer_tvec: Tuple[Tuple[Tuple[float, ...], ...], ...] = (
        (
            (0.0, 0.0, -25.0),
        ),
    )
    """Translation vector from marker to camera frame in mm."""

    transfer_rmat: Tuple[Tuple[Tuple[Tuple[float, ...], ...], ...], ...] = (
        (
            (1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, -1.0),
        ),
    )
    """Rotation matrix from marker to camera frame."""
