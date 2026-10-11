"""
Abstract Base Interface for Identity Storage and Vector Retrieval backends.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from facetrust.core.types import MatchResult


class BaseStorage(ABC):
    """Contract for gallery identity storage and 1:N vector similarity search."""

    @abstractmethod
    def enroll(self, identity_id: str, embedding: np.ndarray) -> None:
        """Saves or updates an enrolled identity's normalized embedding."""
        pass

    @abstractmethod
    def get(self, identity_id: str) -> Optional[np.ndarray]:
        """Retrieves an enrolled identity's embedding."""
        pass

    @abstractmethod
    def delete(self, identity_id: str) -> bool:
        """Removes an identity from the gallery."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 1,
        threshold: float = 0.40,
    ) -> List[MatchResult]:
        """
        Computes cosine similarity between query and all enrolled identities,
        returning top_k matches above threshold.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Returns the total number of enrolled identities."""
        pass
