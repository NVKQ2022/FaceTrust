"""
Universal Image Codec: Ingests diverse image representations and produces
standardized BGR uint8 NumPy arrays for computer vision pipelines.
"""

import base64
from pathlib import Path
from typing import Union
import cv2
import numpy as np
from PIL import Image
import io


class ImageCodec:
    """Decodes file paths, URLs, PIL Images, bytes, and Base64 into BGR arrays."""

    @classmethod
    def to_bgr(cls, image_input: Union[str, Path, bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Converts any supported image format to a 3-channel BGR numpy array.
        """
        # 1. Already a numpy array
        if isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                # Grayscale to BGR
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif image_input.ndim == 3:
                if image_input.shape[2] == 4:
                    return cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
                return image_input
            raise ValueError(f"Invalid numpy image shape: {image_input.shape}")

        # 2. PIL Image
        if isinstance(image_input, Image.Image):
            rgb = np.array(image_input.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # 3. Path or Path-like string
        if isinstance(image_input, (Path, str)):
            str_path = str(image_input)
            # Check if it is a Base64 string
            if str_path.startswith("data:image") or len(str_path) > 500:
                return cls._decode_base64(str_path)

            file_path = Path(str_path)
            if file_path.exists() and file_path.is_file():
                img = cv2.imread(str(file_path))
                if img is None:
                    raise ValueError(f"Failed to load image from file path: '{file_path}'")
                return img

            # Fallback attempt at base64 if short
            try:
                return cls._decode_base64(str_path)
            except Exception:
                raise FileNotFoundError(f"Image file does not exist: '{file_path}'")

        # 4. Raw bytes
        if isinstance(image_input, (bytes, bytearray)):
            np_arr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image from raw bytes.")
            return img

        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    @classmethod
    def _decode_base64(cls, base64_str: str) -> np.ndarray:
        """Decodes standard or Data-URI Base64 string into BGR image."""
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
        raw_bytes = base64.b64decode(base64_str)
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from Base64 string.")
        return img

    @classmethod
    def to_base64(cls, image_bgr: np.ndarray, format: str = ".jpg") -> str:
        """Encodes a BGR array to a Base64 JPEG/PNG string."""
        success, buffer = cv2.imencode(format, image_bgr)
        if not success:
            raise RuntimeError("Failed to encode image to buffer.")
        return base64.b64encode(buffer).decode("utf-8")
