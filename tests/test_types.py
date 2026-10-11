"""
Tests for FaceTrust domain types and DTO models.
"""

from facetrust.core.types import (
    BoundingBox,
    FacialKeypoints,
    FaceDetection,
    LivenessResult,
    Point2D,
    VerificationResult,
)


def test_bounding_box_geometry():
    box = BoundingBox(origin_x=10, origin_y=20, width=100, height=150)
    assert box.x1 == 10
    assert box.y1 == 20
    assert box.x2 == 110
    assert box.y2 == 170
    assert box.center.x == 60.0
    assert box.center.y == 95.0


def test_verification_result_is_trusted():
    # Both match and real -> True
    res = VerificationResult(
        is_match=True,
        similarity=0.75,
        threshold=0.40,
        status="MATCH",
        probe_liveness=LivenessResult(is_real=True, confidence=0.92, label="Real"),
        reference_liveness=LivenessResult(is_real=True, confidence=0.88, label="Real"),
    )
    assert res.is_trusted is True

    # Match but probe spoof -> False
    res_spoof = VerificationResult(
        is_match=True,
        similarity=0.75,
        threshold=0.40,
        status="MATCH",
        probe_liveness=LivenessResult(is_real=False, confidence=0.15, label="Spoof", attack_type="replay"),
    )
    assert res_spoof.is_trusted is False

    # Mismatch -> False
    res_mismatch = VerificationResult(
        is_match=False,
        similarity=0.20,
        threshold=0.40,
        status="MISMATCH",
    )
    assert res_mismatch.is_trusted is False
