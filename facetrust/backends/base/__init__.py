"""
Abstract base contracts for all pluggable backends.
"""

from .detector import BaseDetector
from .antispoof import BaseAntiSpoofer
from .aligner import BaseAligner
from .embedder import BaseEmbedder
from .storage import BaseStorage

__all__ = [
    "BaseDetector",
    "BaseAntiSpoofer",
    "BaseAligner",
    "BaseEmbedder",
    "BaseStorage",
]
