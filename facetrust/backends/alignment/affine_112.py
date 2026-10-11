"""
Canonical 4-Point Partial Affine Face Alignment to 112x112 Grid.
"""

from typing import Tuple
import cv2
import numpy as np

from facetrust.backends.base.aligner import BaseAligner
from facetrust.core.registry import ModelRegistry
from facetrust.core.types import FaceDetection

CANONICAL_POINTS_112 = np.array(
    [
        [38.2946, 51.6963],  # Right eye
        [73.5318, 51.6963],  # Left eye
        [56.0252, 71.7366],  # Nose tip
        [56.0252, 92.3655],  # Mouth center
    ],
    dtype=np.float32,
)


@ModelRegistry.register_aligner("affine_112")
class Affine112Aligner(BaseAligner):
    """
    Standard ArcFace canonical alignment engine.
    Solves 2D partial affine similarity transformation to remove head tilt,
    yaw, pitch, and scale variances.
    """

    def __init__(self, target_size: Tuple[int, int] = (112, 112), **kwargs):
        self._target_size = target_size

    @property
    def target_size(self) -> Tuple[int, int]:
        return self._target_size

    def align_and_crop(self, image_bgr: np.ndarray, detection: FaceDetection) -> np.ndarray:
        kp = detection.keypoints
        src_points = np.array(
            [
                [kp.right_eye.x, kp.right_eye.y],
                [kp.left_eye.x, kp.left_eye.y],
                [kp.nose_tip.x, kp.nose_tip.y],
                [kp.mouth_center.x, kp.mouth_center.y],
            ],
            dtype=np.float32,
        )

        matrix, _ = cv2.estimateAffinePartial2D(src_points, CANONICAL_POINTS_112)
        if matrix is None:
            # Fallback to direct bbox crop
            bbox = detection.bbox
            h, w = image_bgr.shape[:2]
            crop = image_bgr[max(0, bbox.y1):min(h, bbox.y2), max(0, bbox.x1):min(w, bbox.x2)]
            return cv2.resize(crop, self._target_size)

        return cv2.warpAffine(
            image_bgr,
            matrix,
            self._target_size,
            borderMode=cv2.BORDER_REFLECT101,
        )

    def normalize_tensor(self, aligned_bgr: np.ndarray) -> np.ndarray:
        # BGR -> RGB
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        # uint8 [0, 255] -> float32 [-1.0, 1.0]
        normalized = (rgb.astype(np.float32) - 127.5) / 127.5
        # HWC (112, 112, 3) -> CHW (3, 112, 112)
        transposed = np.transpose(normalized, (2, 0, 1))
        # Add batch dimension -> NCHW (1, 3, 112, 112)
        return np.expand_dims(transposed, axis=0)
