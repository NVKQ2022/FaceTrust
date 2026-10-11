"""
In-Memory Vector & Identity Storage Backend using NumPy Vectorized Cosine Search.
"""

from threading import Lock
from typing import Dict, List, Optional
import numpy as np

from facetrust.backends.base.storage import BaseStorage
from facetrust.core.registry import ModelRegistry
from facetrust.core.types import MatchResult


@ModelRegistry.register_storage("memory")
class MemoryStorage(BaseStorage):
    """
    Fast in-memory gallery storage for 1:N identity identification.
    Uses vectorized dot products over L2-normalized embeddings.
    """

    def __init__(self, **kwargs):
        self._gallery: Dict[str, np.ndarray] = {}
        self._lock = Lock()

    def enroll(self, identity_id: str, embedding: np.ndarray) -> None:
        with self._lock:
            # Ensure 1D float32 normalized vector
            emb = np.asarray(embedding, dtype=np.float32).flatten()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            self._gallery[str(identity_id)] = emb

    def get(self, identity_id: str) -> Optional[np.ndarray]:
        with self._lock:
            return self._gallery.get(str(identity_id))

    def delete(self, identity_id: str) -> bool:
        with self._lock:
            return self._gallery.pop(str(identity_id), None) is not None

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 1,
        threshold: float = 0.40,
    ) -> List[MatchResult]:
        with self._lock:
            if not self._gallery:
                return []

            q_emb = np.asarray(query_embedding, dtype=np.float32).flatten()
            q_norm = np.linalg.norm(q_emb)
            if q_norm > 0:
                q_emb = q_emb / q_norm

            identities = list(self._gallery.keys())
            matrix = np.stack([self._gallery[i] for i in identities], axis=0)

            # Vectorized cosine similarities = dot products of L2-normalized vectors
            sims = np.dot(matrix, q_emb)

            # Filter and sort
            matches: List[MatchResult] = []
            sorted_indices = np.argsort(sims)[::-1]

            for rank_idx, idx in enumerate(sorted_indices[:top_k], start=1):
                score = float(sims[idx])
                if score >= threshold:
                    matches.append(
                        MatchResult(
                            identity_id=identities[idx],
                            similarity=round(score, 4),
                            rank=rank_idx,
                        )
                    )

            return matches

    def count(self) -> int:
        with self._lock:
            return len(self._gallery)
