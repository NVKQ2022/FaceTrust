"""
ArcFace MobileFaceNet 512-D Deep Embedding Backend using ONNX Runtime.
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import onnxruntime as ort

from facetrust.backends.base.embedder import BaseEmbedder
from facetrust.core.registry import ModelRegistry
from facetrust.utils.downloader import ModelManager


@ModelRegistry.register_embedder("arcface_mobile")
class ArcFaceMobileEmbedder(BaseEmbedder):
    """
    ArcFace MobileFaceNet (w600k_mbf) deep feature extractor.
    Extracts 512-D L2-normalized feature representation from aligned 112x112 crops.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        **kwargs,
    ):
        self.device = device.lower()
        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = ModelManager.get_model_path("arcface_mobile")

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
        return "ArcFace MobileFaceNet"

    @property
    def embedding_dim(self) -> int:
        return 512

    def extract(self, tensor_nchw: np.ndarray, normalize: bool = True) -> np.ndarray:
        if tensor_nchw.ndim == 3:
            tensor_nchw = np.expand_dims(tensor_nchw, axis=0)

        raw_outputs = self.session.run(None, {self.input_name: tensor_nchw})[0]

        if normalize:
            # L2 hypersphere normalization across axis 1
            norms = np.linalg.norm(raw_outputs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-12, norms)
            normalized = raw_outputs / norms
            if normalized.shape[0] == 1:
                return normalized[0]
            return normalized

        if raw_outputs.shape[0] == 1:
            return raw_outputs[0]
        return raw_outputs

    def close(self) -> None:
        if hasattr(self, "session") and self.session is not None:
            del self.session
            self.session = None
