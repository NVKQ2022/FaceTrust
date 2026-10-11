"""
FaceTrust: Production-Grade, Modular Face Authentication with Built-in Anti-Spoofing.

Quickstart:
    >>> from facetrust import FaceTrust
    >>> auth = FaceTrust()
    >>> result = auth.verify("id.jpg", "selfie.jpg")
    >>> print(result.is_trusted, result.similarity)
"""

__version__ = "0.1.0"

from facetrust.core.client import FaceTrust
from facetrust.core.config import FaceTrustConfig
from facetrust.core.registry import ModelRegistry
from facetrust.core.types import (
    BoundingBox,
    FaceDetection,
    FacialKeypoints,
    LivenessResult,
    MatchResult,
    Point2D,
    VerificationResult,
)
from facetrust.backends.storage.langchain import LangChainVectorStoreAdapter
from facetrust.integrations.langchain import FaceTrustEmbeddings, FaceVectorStore
from facetrust.utils.image import ImageCodec

__all__ = [
    "__version__",
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
    "ImageCodec",
    "LangChainVectorStoreAdapter",
    "FaceTrustEmbeddings",
    "FaceVectorStore",
]
