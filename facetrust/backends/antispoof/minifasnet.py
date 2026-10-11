"""
MiniFASNet Face Anti-Spoofing & Liveness Backend using ONNX Runtime.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np
import onnxruntime as ort

from facetrust.backends.base.antispoof import BaseAntiSpoofer
from facetrust.core.registry import ModelRegistry
from facetrust.core.types import BoundingBox, LivenessResult
from facetrust.utils.downloader import ModelManager


@ModelRegistry.register_antispoof("minifasnet")
class MiniFASNetAntiSpoofer(BaseAntiSpoofer):
    """
    MiniFASNetV2 Anti-Spoofing and Liveness Verification backend.
    Ingests 2.7x context-expanded patch to detect print and screen replay attacks.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        threshold: float = 0.60,
        scale: float = 2.7,
        target_size: Tuple[int, int] = (80, 80),
        device: str = "cpu",
        **kwargs,
    ):
        self.threshold = threshold
        self.scale = scale
        self.target_size = target_size
        self.device = device.lower()

        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = ModelManager.get_model_path("minifasnet")

        providers = ["CPUExecutionProvider"]
        if self.device == "cuda":
            providers = [("CUDAExecutionProvider", {"device_id": kwargs.get("device_id", 0)}), "CPUExecutionProvider"]
        elif self.device == "tensorrt":
            providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    @property
    def name(self) -> str:
        return "MiniFASNetV2"

    def _crop_expanded_patch(self, image_bgr: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        src_h, src_w = image_bgr.shape[:2]
        box_w = max(bbox.width, 1)
        box_h = max(bbox.height, 1)

        scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, self.scale))
        new_w = box_w * scale
        new_h = box_h * scale

        cx = bbox.origin_x + box_w / 2.0
        cy = bbox.origin_y + box_h / 2.0

        x1 = max(0, int(cx - new_w / 2.0))
        y1 = max(0, int(cy - new_h / 2.0))
        x2 = min(src_w - 1, int(cx + new_w / 2.0))
        y2 = min(src_h - 1, int(cy + new_h / 2.0))

        patch = image_bgr[y1:y2 + 1, x1:x2 + 1]
        if patch.size == 0 or patch.shape[0] < 2 or patch.shape[1] < 2:
            patch = image_bgr[max(0, bbox.y1):min(src_h, bbox.y2), max(0, bbox.x1):min(src_w, bbox.x2)]

        return cv2.resize(patch, self.target_size, interpolation=cv2.INTER_AREA)

    def check_liveness(self, image_bgr: np.ndarray, bbox: BoundingBox) -> LivenessResult:
        patch = self._crop_expanded_patch(image_bgr, bbox)

        # Convert to BGR float32 in range [0, 255] and NCHW shape (1, 3, 80, 80)
        blob = np.transpose(patch.astype(np.float32), (2, 0, 1))[np.newaxis, ...]

        logits = self.session.run(None, {self.input_name: blob})[0][0]

        # Numerically stable softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        real_confidence = float(probs[1])
        is_real = bool(real_confidence >= self.threshold)

        if is_real:
            label = "Real"
            attack_type = None
        else:
            label = "Spoof"
            attack_type = "print" if probs[0] > probs[2] else "replay"

        return LivenessResult(
            is_real=is_real,
            confidence=round(real_confidence, 4),
            label=label,
            attack_type=attack_type,
            raw_scores=[round(float(p), 4) for p in probs],
        )

    def close(self) -> None:
        if hasattr(self, "session") and self.session is not None:
            del self.session
            self.session = None
