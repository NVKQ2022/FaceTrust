"""
Tests for ModelRegistry dynamic plugin registration and Dependency Injection.
"""

import pytest
import numpy as np
from facetrust.core.registry import ModelRegistry
from facetrust.backends.base import BaseDetector, BaseEmbedder
from facetrust.core.types import FaceDetection


class MockDetector(BaseDetector):
    @property
    def name(self) -> str:
        return "MockDetector"

    def detect(self, image_bgr: np.ndarray):
        return []


def test_model_registry_custom_registration():
    @ModelRegistry.register_detector("test_mock")
    class CustomDetector(MockDetector):
        pass

    assert "test_mock" in ModelRegistry.list_available()["detectors"]

    inst = ModelRegistry.create_detector("test_mock")
    assert isinstance(inst, CustomDetector)
    assert inst.name == "MockDetector"


def test_model_registry_direct_instance_injection():
    direct_inst = MockDetector()
    resolved = ModelRegistry.create_detector(direct_inst)
    assert resolved is direct_inst


def test_model_registry_unknown_model_raises():
    with pytest.raises(KeyError):
        ModelRegistry.create_detector("non_existent_detector_xyz")
