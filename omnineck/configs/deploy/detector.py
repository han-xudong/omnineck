"""
Dataclass for detector configuration parameters.
"""

from dataclasses import dataclass


@dataclass
class DetectorConfig:
    adaptiveThreshConstant: int = 7
    "Constant for adaptive thresholding before finding contours"

    adaptiveThreshWinSizeMin: int = 3
    """Minimum window size for adaptive thresholding before finding contours."""

    adaptiveThreshWinSizeMax: int = 23
    """Maximum window size for adaptive thresholding before finding contours."""

    adaptiveThreshWinSizeStep: int = 5
    """Increments from adaptiveThreshWinSizeMin to adaptiveThreshWinSizeMax during the thresholding."""

    minMarkerPerimeterRate: float = 0.05
    """Minimum perimeter for marker contour to be detected."""

    maxMarkerPerimeterRate: float = 3.0
    """Maximum perimeter for marker contour to be detected."""

    polygonalApproxAccuracyRate: float = 0.02
    """Minimum accuracy during the polygonal approximation process to determine which contours are squares."""

    minCornerDistanceRate: float = 0.05
    """Minimum distance between corners for detected markers relative to its perimeter."""

    minDistanceToBorder: int = 1
    """Minimum distance of any corner to the image border for detected markers (in pixels)."""

    minMarkerDistanceRate: float = 0.05
    """Minimum average distance between the corners of the two markers to be grouped."""

    markerBorderBits: int = 1
    """Number of bits of the marker border, i.e. marker border width."""

    perspectiveRemovePixelPerCell: int = 8
    """Number of bits (per dimension) for each cell of the marker when removing the perspective."""

    perspectiveRemoveIgnoredMarginPerCell: float = 0.13
    """Width of the margin of pixels on each cell not considered for the determination of the cell bit."""

    maxErroneousBitsInBorderRate: float = 0.04
    """Maximum number of accepted erroneous bits in the border (i.e. number of allowed white bits in the border)."""

    minOtsuStdDev: float = 5.0
    """Minimun standard deviation in pixels values during the decodification step to apply Otsu thresholding (otherwise, all the bits are set to 0 or 1 depending on mean higher than 128 or not)."""

    cornerRefinementMethod: int = 2
    """CORNER_REFINE_NONE=0, CORNER_REFINE_SUBPIX=1, CORNER_REFINE_CONTOUR=2, CORNER_REFINE_APRILTAG=3"""

    cornerRefinementWinSize: int = 5
    """Maximum window size for the corner refinement process (in pixels)."""

    cornerRefinementMaxIterations: int = 50
    """Maximum number of iterations for stop criteria of the corner refinement process."""

    cornerRefinementMinAccuracy: float = 0.01
    """Minimum error for the stop cristeria of the corner refinement process."""

    errorCorrectionRate: float = 0.1
    """Error correction rate respect to the maximun error correction capability for each dictionary."""

    useAruco3Detection: int = 1
    """Enable the new and faster Aruco detection strategy."""

    minSideLengthCanonicalImg: int = 16
    """Minimum side length of a marker in the canonical image. Latter is the binarized image in which contours are searched."""

    minMarkerLengthRatioOriginalImg: float = 0.05
    """Range [0,1], eq (2) from paper. The parameter tau_i has a direct influence on the processing speed."""
