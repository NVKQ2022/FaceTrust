"""
Abstract Base Interface for Face Detection backends.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from facetrust.core.types import FaceDetection


class BaseDetector(ABC):
    """Contract for any face detection model (BlazeFace, RetinaFace, YOLO, MTCNN)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the detector backend."""
        pass

    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> List[FaceDetection]:
        """
        Detects all faces in a BGR image.
        Returns a list of FaceDetection instances.
        """
        pass

    def detect_best(self, image_bgr: np.ndarray) -> Optional[FaceDetection]:
        """
        Convenience method: returns the most prominent / highest-confidence face.
        """
        faces = self.detect(image_bgr)
        if not faces:
            return None
        # Rank by area * confidence
        return max(faces, key=lambda f: (f.bbox.width * f.bbox.height) * f.confidence)

    def close(self) -> None:
        """Release underlying inference resources."""
        pass
