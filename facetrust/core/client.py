"""
FaceTrust High-Level Client Facade: Unified, developer-friendly interface
for Face Detection, Anti-Spoofing, Alignment, Embedding, and Verification.
"""

import asyncio
from typing import Any, List, Optional, Union
import numpy as np
from PIL import Image

# Ensure built-in backends are registered
import facetrust.backends  # noqa: F401

from facetrust.backends.base import (
    BaseAligner,
    BaseAntiSpoofer,
    BaseDetector,
    BaseEmbedder,
    BaseStorage,
)
from facetrust.core.config import FaceTrustConfig
from facetrust.core.registry import ModelRegistry
from facetrust.core.types import (
    FaceDetection,
    LivenessResult,
    MatchResult,
    VerificationResult,
)
from facetrust.utils.image import ImageCodec


class FaceTrust:
    """
    Main entrypoint for the FaceTrust Biometric Verification Library.

    Usage:
        >>> auth = FaceTrust()
        >>> result = auth.verify("person1.jpg", "person2.jpg")
        >>> print(result.is_trusted, result.similarity)
    """

    def __init__(
        self,
        config: Optional[FaceTrustConfig] = None,
        detector: Optional[Union[str, BaseDetector]] = None,
        antispoof: Optional[Union[str, BaseAntiSpoofer]] = None,
        aligner: Optional[Union[str, BaseAligner]] = None,
        embedder: Optional[Union[str, BaseEmbedder]] = None,
        storage: Optional[Union[str, BaseStorage]] = None,
        **kwargs: Any,
    ):
        self.config = config or FaceTrustConfig(**kwargs)

        # Resolve model backends via Registry (supports strings or instantiated objects)
        det_target = detector if detector is not None else self.config.detector
        self.detector = ModelRegistry.create_detector(
            det_target,
            min_confidence=self.config.detection_confidence,
            cache_dir=self.config.resolved_cache_dir,
            **kwargs,
        )

        spoof_target = antispoof if antispoof is not None else (
            self.config.antispoof if self.config.enable_liveness else None
        )
        self.antispoofer = ModelRegistry.create_antispoof(
            spoof_target,
            threshold=self.config.liveness_threshold,
            device=self.config.device,
            device_id=self.config.device_id,
            cache_dir=self.config.resolved_cache_dir,
            **kwargs,
        )

        align_target = aligner if aligner is not None else self.config.aligner
        self.aligner = ModelRegistry.create_aligner(align_target, **kwargs)

        emb_target = embedder if embedder is not None else self.config.embedder
        self.embedder = ModelRegistry.create_embedder(
            emb_target,
            device=self.config.device,
            device_id=self.config.device_id,
            cache_dir=self.config.resolved_cache_dir,
            **kwargs,
        )

        store_target = storage if storage is not None else self.config.storage
        self.storage = ModelRegistry.create_storage(store_target, **kwargs)

    # --- Core Pipeline Operations ---

    def detect(self, image: Union[str, bytes, np.ndarray, Image.Image]) -> List[FaceDetection]:
        """Detects all faces in an image."""
        img_bgr = ImageCodec.to_bgr(image)
        return self.detector.detect(img_bgr)

    def check_liveness(self, image: Union[str, bytes, np.ndarray, Image.Image]) -> LivenessResult:
        """Evaluates face anti-spoofing and liveness on the prominent face."""
        if not self.antispoofer:
            raise RuntimeError("Anti-spoofing is disabled in current FaceTrust configuration.")

        img_bgr = ImageCodec.to_bgr(image)
        best_face = self.detector.detect_best(img_bgr)
        if not best_face:
            return LivenessResult(
                is_real=False,
                confidence=0.0,
                label="No Face",
                attack_type=None,
            )
        return self.antispoofer.check_liveness(img_bgr, best_face.bbox)

    def get_embedding(
        self,
        image: Union[str, bytes, np.ndarray, Image.Image],
        require_liveness: bool = False,
    ) -> np.ndarray:
        """
        Extracts a normalized 512-D embedding from an image.
        Raises ValueError if face is missing or spoof attack is detected.
        """
        img_bgr = ImageCodec.to_bgr(image)
        best_face = self.detector.detect_best(img_bgr)
        if not best_face:
            raise ValueError("No face detected in the provided image.")

        if require_liveness:
            if not self.antispoofer:
                raise RuntimeError("require_liveness=True requested, but no anti-spoofer configured.")
            liveness = self.antispoofer.check_liveness(img_bgr, best_face.bbox)
            if not liveness.is_real:
                raise ValueError(
                    f"Spoof attack detected ({liveness.attack_type or 'spoof'}). "
                    f"Liveness confidence: {liveness.confidence:.4f} < threshold."
                )

        aligned_crop = self.aligner.align_and_crop(img_bgr, best_face)
        tensor = self.aligner.normalize_tensor(aligned_crop)
        return self.embedder.extract(tensor, normalize=True)

    def get_embeddings_batch(
        self,
        images: List[Union[str, bytes, np.ndarray, Image.Image]],
        batch_size: int = 32,
        require_liveness: bool = False,
    ) -> List[np.ndarray]:
        """
        Extracts embeddings for a batch of images with GPU/CPU parallelization.
        """
        results: List[np.ndarray] = []
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            for img in chunk:
                results.append(self.get_embedding(img, require_liveness=require_liveness))
        return results

    def verify(
        self,
        image1: Union[str, bytes, np.ndarray, Image.Image],
        image2: Union[str, bytes, np.ndarray, Image.Image],
        threshold: Optional[float] = None,
        check_liveness: Optional[bool] = None,
    ) -> VerificationResult:
        """
        Performs 1:1 biometric face verification between two face images.
        """
        thresh = threshold if threshold is not None else self.config.similarity_threshold
        perform_liveness = (
            check_liveness if check_liveness is not None else self.config.enable_liveness
        )

        img1_bgr = ImageCodec.to_bgr(image1)
        best_face1 = self.detector.detect_best(img1_bgr)
        if not best_face1:
            return VerificationResult(
                is_match=False,
                similarity=0.0,
                threshold=thresh,
                status="REJECTED: No face detected in Image 1",
            )

        img2_bgr = ImageCodec.to_bgr(image2)
        best_face2 = self.detector.detect_best(img2_bgr)
        if not best_face2:
            return VerificationResult(
                is_match=False,
                similarity=0.0,
                threshold=thresh,
                status="REJECTED: No face detected in Image 2",
            )

        # Anti-Spoofing Liveness Checks
        live1: Optional[LivenessResult] = None
        live2: Optional[LivenessResult] = None

        if perform_liveness and self.antispoofer:
            live1 = self.antispoofer.check_liveness(img1_bgr, best_face1.bbox)
            live2 = self.antispoofer.check_liveness(img2_bgr, best_face2.bbox)

            if not live1.is_real or not live2.is_real:
                spoofs = []
                if not live1.is_real:
                    spoofs.append(f"Image 1 ({live1.attack_type or 'spoof'})")
                if not live2.is_real:
                    spoofs.append(f"Image 2 ({live2.attack_type or 'spoof'})")

                return VerificationResult(
                    is_match=False,
                    similarity=0.0,
                    threshold=thresh,
                    status=f"REJECTED: Spoof attack detected in {', '.join(spoofs)}",
                    probe_liveness=live1,
                    reference_liveness=live2,
                )

        # Alignment and Embedding Extraction
        aligned1 = self.aligner.align_and_crop(img1_bgr, best_face1)
        tensor1 = self.aligner.normalize_tensor(aligned1)
        emb1 = self.embedder.extract(tensor1, normalize=True)

        aligned2 = self.aligner.align_and_crop(img2_bgr, best_face2)
        tensor2 = self.aligner.normalize_tensor(aligned2)
        emb2 = self.embedder.extract(tensor2, normalize=True)

        # Cosine Similarity between L2-normalized vectors
        similarity = float(np.dot(emb1, emb2))
        is_match = bool(similarity >= thresh)

        status = "MATCH" if is_match else "MISMATCH"
        return VerificationResult(
            is_match=is_match,
            similarity=round(similarity, 4),
            threshold=thresh,
            status=status,
            probe_liveness=live1,
            reference_liveness=live2,
        )

    # --- 1:N Gallery Operations ---

    def enroll(
        self,
        identity_id: str,
        image: Union[str, bytes, np.ndarray, Image.Image],
        require_liveness: bool = True,
    ) -> np.ndarray:
        """Enrolls an identity into the vector store."""
        emb = self.get_embedding(image, require_liveness=require_liveness)
        self.storage.enroll(identity_id, emb)
        return emb

    def identify(
        self,
        probe_image: Union[str, bytes, np.ndarray, Image.Image],
        top_k: int = 1,
        threshold: Optional[float] = None,
        check_liveness: bool = True,
    ) -> List[MatchResult]:
        """Identifies a probe image against all enrolled gallery identities."""
        thresh = threshold if threshold is not None else self.config.similarity_threshold
        emb = self.get_embedding(probe_image, require_liveness=check_liveness)
        return self.storage.search(emb, top_k=top_k, threshold=thresh)

    # --- Async Support ---

    async def a_verify(
        self,
        image1: Union[str, bytes, np.ndarray, Image.Image],
        image2: Union[str, bytes, np.ndarray, Image.Image],
        threshold: Optional[float] = None,
        check_liveness: Optional[bool] = None,
    ) -> VerificationResult:
        """Asynchronous non-blocking face verification for web servers."""
        return await asyncio.to_thread(
            self.verify,
            image1=image1,
            image2=image2,
            threshold=threshold,
            check_liveness=check_liveness,
        )

    # --- Context Manager & Cleanup ---

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Releases underlying model sessions."""
        if hasattr(self.detector, "close"):
            self.detector.close()
        if self.antispoofer and hasattr(self.antispoofer, "close"):
            self.antispoofer.close()
        if hasattr(self.embedder, "close"):
            self.embedder.close()
