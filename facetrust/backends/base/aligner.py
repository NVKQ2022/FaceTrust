"""
Abstract Base Interface for Landmark-Based Face Alignment backends.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from facetrust.core.types import FaceDetection


class BaseAligner(ABC):
    """Contract for geometry standardization and canonical alignment."""

    @property
    @abstractmethod
    def target_size(self) -> Tuple[int, int]:
        """Expected output resolution (e.g. (112, 112))."""
        pass

    @abstractmethod
    def align_and_crop(
        self,
        image_bgr: np.ndarray,
        detection: FaceDetection,
    ) -> np.ndarray:
        """
        Transforms the face into canonical coordinates and crops to target_size.
        """
        pass

    @abstractmethod
    def normalize_tensor(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """
        Converts aligned BGR image to the normalized NCHW float32 tensor
        required by the embedding network.
        """
        pass
