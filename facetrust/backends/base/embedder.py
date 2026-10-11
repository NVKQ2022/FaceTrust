"""
Abstract Base Interface for Deep Face Feature Extraction / Embedding backends.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseEmbedder(ABC):
    """Contract for Face Feature Extractor models (ArcFace, AdaFace, InsightFace)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the embedding backend."""
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the feature vector (e.g. 512)."""
        pass

    @abstractmethod
    def extract(self, tensor_nchw: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Runs inference on an (N, C, H, W) normalized tensor and returns
        the (N, D) or (D,) feature embedding vector.
        """
        pass

    def close(self) -> None:
        """Release underlying inference resources."""
        pass
