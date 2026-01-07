from dataclasses import dataclass


@dataclass
class DetectorConfig:
    adaptiveThreshConstant: int = 7
    adaptiveThreshWinSizeMin: int = 3
    adaptiveThreshWinSizeMax: int = 23
    adaptiveThreshWinSizeStep: int = 5
    minMarkerPerimeterRate: float = 0.05
    maxMarkerPerimeterRate: float = 3.0
    polygonalApproxAccuracyRate: float = 0.01
    minCornerDistanceRate: float = 0.05
    minDistanceToBorder: int = 1
    minMarkerDistanceRate: float = 0.05
    markerBorderBits: int = 1
    perspectiveRemovePixelPerCell: int = 8
    perspectiveRemoveIgnoredMarginPerCell: float = 0.13
    maxErroneousBitsInBorderRate: float = 0.04
    minOtsuStdDev: float = 5.0
    cornerRefinementMethod: int = 2
    cornerRefinementWinSize: int = 5
    cornerRefinementMaxIterations: int = 50
    cornerRefinementMinAccuracy: float = 0.001
    errorCorrectionRate: float = 0.1
    useAruco3Detection: int = 1
    minSideLengthCanonicalImg: int = 16
    minMarkerLengthRatioOriginalImg: float = 0.05