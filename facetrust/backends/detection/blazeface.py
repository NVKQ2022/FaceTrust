import ctypes
import os
from pathlib import Path
from typing import List, Optional, Union
import cv2
import numpy as np


def _try_preload_gles():
    for name in ["libGLESv2.so.2", "libGLESv2.so"]:
        try:
            ctypes.CDLL(name)
            return
        except OSError:
            pass
    candidates = [
        "/usr/lib/x86_64-linux-gnu/libGLESv2.so.2",
        "/usr/lib/libGLESv2.so.2",
        "/home/quan/projects/maivenpoint/AI/FaceAuthentication/Recognition/libs/usr/lib/x86_64-linux-gnu/libGLESv2.so.2",
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                ctypes.CDLL(c)
                return
            except OSError:
                pass


_try_preload_gles()

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from facetrust.backends.base.detector import BaseDetector
from facetrust.core.registry import ModelRegistry
from facetrust.core.types import BoundingBox, FaceDetection, FacialKeypoints, Point2D
from facetrust.utils.downloader import ModelManager


@ModelRegistry.register_detector("blazeface")
class BlazeFaceDetector(BaseDetector):
    """
    Google MediaPipe BlazeFace short-range detector implementation.
    Optimized for sub-15ms edge/CPU inference.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        min_confidence: float = 0.50,
        **kwargs,
    ):
        self.min_confidence = min_confidence
        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = ModelManager.get_model_path("blazeface")

        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=self.min_confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    @property
    def name(self) -> str:
        return "BlazeFace"

    def detect(self, image_bgr: np.ndarray) -> List[FaceDetection]:
        h, w = image_bgr.shape[:2]
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        results = self._detector.detect(mp_image)
        if not results or not results.detections:
            return []

        detections: List[FaceDetection] = []
        for det in results.detections:
            score = float(det.categories[0].score) if det.categories else 0.0
            box = det.bounding_box

            bbox = BoundingBox(
                origin_x=max(0, int(box.origin_x)),
                origin_y=max(0, int(box.origin_y)),
                width=int(box.width),
                height=int(box.height),
            )

            # Extract 6 keypoints
            keypoints_dict = {}
            if det.keypoints:
                # MediaPipe BlazeFace indices:
                # 0: right eye, 1: left eye, 2: nose tip, 3: mouth center, 4: right ear, 5: left ear
                names = [
                    "right_eye",
                    "left_eye",
                    "nose_tip",
                    "mouth_center",
                    "right_ear_tragion",
                    "left_ear_tragion",
                ]
                for idx, kp in enumerate(det.keypoints[:6]):
                    keypoints_dict[names[idx]] = Point2D(
                        x=float(kp.x * w),
                        y=float(kp.y * h),
                    )

            keypoints = FacialKeypoints(
                right_eye=keypoints_dict.get("right_eye", Point2D(x=0.0, y=0.0)),
                left_eye=keypoints_dict.get("left_eye", Point2D(x=0.0, y=0.0)),
                nose_tip=keypoints_dict.get("nose_tip", Point2D(x=0.0, y=0.0)),
                mouth_center=keypoints_dict.get("mouth_center", Point2D(x=0.0, y=0.0)),
                right_ear_tragion=keypoints_dict.get("right_ear_tragion"),
                left_ear_tragion=keypoints_dict.get("left_ear_tragion"),
            )

            detections.append(
                FaceDetection(
                    bbox=bbox,
                    confidence=round(score, 4),
                    keypoints=keypoints,
                )
            )

        return detections

    def close(self) -> None:
        if hasattr(self, "_detector") and self._detector:
            self._detector.close()
            self._detector = None
