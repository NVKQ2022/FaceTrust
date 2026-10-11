"""
Strongly typed domain models and Data Transfer Objects (DTOs) for FaceTrust.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class Point2D(BaseModel):
    """2D spatial coordinate on an image plane."""
    x: float
    y: float


class BoundingBox(BaseModel):
    """Bounding box for a detected face."""
    origin_x: int
    origin_y: int
    width: int
    height: int

    @property
    def x1(self) -> int:
        return self.origin_x

    @property
    def y1(self) -> int:
        return self.origin_y

    @property
    def x2(self) -> int:
        return self.origin_x + self.width

    @property
    def y2(self) -> int:
        return self.origin_y + self.height

    @property
    def center(self) -> Point2D:
        return Point2D(
            x=self.origin_x + self.width / 2.0,
            y=self.origin_y + self.height / 2.0,
        )


class FacialKeypoints(BaseModel):
    """Standard 6 facial anatomical landmarks."""
    right_eye: Point2D
    left_eye: Point2D
    nose_tip: Point2D
    mouth_center: Point2D
    right_ear_tragion: Optional[Point2D] = None
    left_ear_tragion: Optional[Point2D] = None


class FaceDetection(BaseModel):
    """A detected face with localization and confidence."""
    bbox: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    keypoints: FacialKeypoints


class LivenessResult(BaseModel):
    """Anti-spoofing and liveness classification evaluation."""
    is_real: bool
    confidence: float = Field(ge=0.0, le=1.0)
    label: str  # "Real" | "Spoof"
    attack_type: Optional[str] = None  # "print" | "replay" | None
    raw_scores: Optional[List[float]] = None  # [p_print, p_real, p_replay]


class VerificationResult(BaseModel):
    """1:1 biometric face verification outcome."""
    is_match: bool
    similarity: float = Field(ge=-1.0, le=1.0)
    threshold: float
    status: str
    probe_liveness: Optional[LivenessResult] = None
    reference_liveness: Optional[LivenessResult] = None

    @property
    def is_trusted(self) -> bool:
        """True if the faces match and both passed liveness checks (if evaluated)."""
        if not self.is_match:
            return False
        if self.probe_liveness and not self.probe_liveness.is_real:
            return False
        if self.reference_liveness and not self.reference_liveness.is_real:
            return False
        return True


class MatchResult(BaseModel):
    """1:N identity match ranking."""
    identity_id: str
    similarity: float
    rank: int = 1
