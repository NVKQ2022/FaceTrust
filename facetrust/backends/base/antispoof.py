"""
Abstract Base Interface for Anti-Spoofing and Liveness Verification backends.
"""

from abc import ABC, abstractmethod
import numpy as np
from facetrust.core.types import BoundingBox, LivenessResult


class BaseAntiSpoofer(ABC):
    """Contract for Anti-Spoofing / Presentation Attack Detection models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the anti-spoofing backend."""
        pass

    @abstractmethod
    def check_liveness(
        self,
        image_bgr: np.ndarray,
        bbox: BoundingBox,
    ) -> LivenessResult:
        """
        Evaluates face liveness using the uncropped image and detected face bbox.
        Returns a LivenessResult DTO.
        """
        pass

    def close(self) -> None:
        """Release underlying inference resources."""
        pass
