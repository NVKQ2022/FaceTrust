"""
Configuration settings for FaceTrust client and pipeline components.
"""

from pathlib import Path
from typing import Optional, Union
from pydantic import BaseModel, Field


class FaceTrustConfig(BaseModel):
    """Configuration options for FaceTrust pipeline."""

    # Backend selections
    detector: str = "blazeface"
    antispoof: Optional[str] = "minifasnet"
    aligner: str = "affine_112"
    embedder: str = "arcface_mobile"
    storage: str = "memory"

    # Execution hardware
    device: str = "cpu"  # "cpu" | "cuda" | "tensorrt"
    device_id: int = 0

    # Decision thresholds
    similarity_threshold: float = Field(default=0.40, ge=-1.0, le=1.0)
    liveness_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    detection_confidence: float = Field(default=0.50, ge=0.0, le=1.0)
    enable_liveness: bool = True

    # Cache directory for model weights
    cache_dir: Optional[Union[str, Path]] = None

    @property
    def resolved_cache_dir(self) -> Path:
        if self.cache_dir:
            p = Path(self.cache_dir)
        else:
            p = Path.home() / ".cache" / "facetrust" / "models"
        p.mkdir(parents=True, exist_ok=True)
        return p
