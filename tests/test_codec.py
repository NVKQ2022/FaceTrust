"""
Tests for Universal ImageCodec.
"""

import cv2
import numpy as np
from PIL import Image
from facetrust.utils.image import ImageCodec


def test_image_codec_numpy_array():
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    bgr = ImageCodec.to_bgr(arr)
    assert bgr.shape == (100, 100, 3)


def test_image_codec_pil_image():
    pil_img = Image.new("RGB", (80, 80), color=(255, 0, 0))
    bgr = ImageCodec.to_bgr(pil_img)
    assert bgr.shape == (80, 80, 3)
    # Check red channel is converted to BGR correctly (B=0, G=0, R=255)
    assert bgr[0, 0, 2] == 255
    assert bgr[0, 0, 0] == 0


def test_image_codec_base64_roundtrip():
    original = np.full((50, 50, 3), 128, dtype=np.uint8)
    b64_str = ImageCodec.to_base64(original)
    decoded = ImageCodec.to_bgr(b64_str)
    assert decoded.shape == (50, 50, 3)
