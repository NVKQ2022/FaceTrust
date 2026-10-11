"""
Autonomous Model Cache & Downloader: Fetches required deep learning model
weights on demand and verifies local cache integrity.
"""

import os
from pathlib import Path
import urllib.request
import zipfile
from typing import Optional
from tqdm import tqdm

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "facetrust" / "models"

MODEL_METADATA = {
    "blazeface": {
        "filename": "blaze_face_short_range.tflite",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/face_detector/"
            "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
        ),
        "is_zip": False,
    },
    "minifasnet": {
        "filename": "minifasnet_v2.onnx",
        "url": (
            "https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx/"
            "resolve/main/minifasnet_v2.onnx"
        ),
        "is_zip": False,
    },
    "arcface_mobile": {
        "filename": "w600k_mbf.onnx",
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip",
        "is_zip": True,
        "zip_target": "w600k_mbf.onnx",
    },
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class ModelManager:
    """Manages local model weights downloading, caching, and resolution."""

    @classmethod
    def get_model_path(
        cls,
        model_key: str,
        cache_dir: Optional[Path] = None,
    ) -> Path:
        """
        Returns the resolved local path to the model file.
        Downloads on demand if not present.
        """
        key = model_key.lower()
        if key not in MODEL_METADATA:
            raise KeyError(f"Unknown model key: '{key}'. Supported: {list(MODEL_METADATA.keys())}")

        meta = MODEL_METADATA[key]
        target_dir = Path(cache_dir or os.getenv("FACETRUST_MODELS_DIR", DEFAULT_CACHE_DIR))
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / meta["filename"]

        if target_path.exists() and target_path.stat().st_size > 0:
            return target_path

        # Also check local common fallback paths (e.g. current project models/ directory)
        local_fallback = Path("models") / meta["filename"]
        if local_fallback.exists() and local_fallback.stat().st_size > 0:
            return local_fallback.resolve()

        # Download automatically
        cls._download(meta, target_dir, target_path)
        return target_path

    @classmethod
    def _download(cls, meta: dict, target_dir: Path, target_path: Path) -> None:
        url = meta["url"]
        filename = meta["filename"]

        if meta.get("is_zip"):
            temp_zip = target_dir / f"{filename}.zip"
            print(f"[FaceTrust] Downloading {filename} from {url}...")
            with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=filename) as t:
                urllib.request.urlretrieve(url, filename=str(temp_zip), reporthook=t.update_to)

            with zipfile.ZipFile(temp_zip, "r") as z:
                z.extract(meta["zip_target"], path=str(target_dir))

            if temp_zip.exists():
                temp_zip.unlink()
            print(f"[FaceTrust] Successfully extracted {filename} to {target_path}")
        else:
            print(f"[FaceTrust] Downloading {filename} to {target_path}...")
            with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=filename) as t:
                urllib.request.urlretrieve(url, filename=str(target_path), reporthook=t.update_to)
            print(f"[FaceTrust] Successfully downloaded {filename}")
