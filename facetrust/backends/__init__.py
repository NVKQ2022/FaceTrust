"""
Auto-register built-in backends upon package initialization.
"""

from .detection.blazeface import BlazeFaceDetector
from .antispoof.minifasnet import MiniFASNetAntiSpoofer
from .alignment.affine_112 import Affine112Aligner
from .embedding.arcface import ArcFaceMobileEmbedder
from .storage.memory import MemoryStorage

__all__ = [
    "BlazeFaceDetector",
    "MiniFASNetAntiSpoofer",
    "Affine112Aligner",
    "ArcFaceMobileEmbedder",
    "MemoryStorage",
]
