"""
Core components of FaceTrust.
"""

from .client import FaceTrust
from .config import FaceTrustConfig
from .registry import ModelRegistry
from .types import (
    BoundingBox,
    FaceDetection,
    FacialKeypoints,
    LivenessResult,
    MatchResult,
    Point2D,
    VerificationResult,
)

__all__ = [
    "FaceTrust",
    "FaceTrustConfig",
    "ModelRegistry",
    "BoundingBox",
    "Point2D",
    "FacialKeypoints",
    "FaceDetection",
    "LivenessResult",
    "VerificationResult",
    "MatchResult",
]
