"""
Tests for LangChain VectorStore Adapter integration.
"""

from typing import List, Tuple
import numpy as np
from facetrust.backends.storage.langchain import LangChainVectorStoreAdapter
from facetrust.core.registry import ModelRegistry


class MockDocument:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


class MockLangChainVectorStore:
    """Simulates a LangChain VectorStore instance."""

    def __init__(self):
        self.entries = []  # List of (embedding, text, metadata)

    def add_texts(self, texts: List[str], metadatas: List[dict] = None, **kwargs):
        metadatas = metadatas or [{}] * len(texts)
        for t, m in zip(texts, metadatas):
            self.entries.append({"text": t, "metadata": m})

    def similarity_search_by_vector_with_relevance_scores(
        self,
        embedding: List[float],
        k: int = 1,
    ) -> List[Tuple[MockDocument, float]]:
        # Simulate returning 2 matches
        results = [
            (
                MockDocument("biometric_identity:alice", {"identity_id": "alice"}),
                0.85,
            ),
            (
                MockDocument("biometric_identity:bob", {"identity_id": "bob"}),
                0.35,
            ),
        ]
        return results[:k]

    def delete(self, ids: List[str]):
        return True


def test_langchain_adapter_auto_detection():
    mock_store = MockLangChainVectorStore()
    resolved = ModelRegistry.create_storage(mock_store)

    assert isinstance(resolved, LangChainVectorStoreAdapter)
    assert resolved.vectorstore is mock_store


def test_langchain_adapter_enroll_and_search():
    mock_store = MockLangChainVectorStore()
    adapter = LangChainVectorStoreAdapter(mock_store)

    # Enroll
    test_vec = np.random.randn(512).astype(np.float32)
    adapter.enroll("alice", test_vec)
    assert len(mock_store.entries) == 1
    assert mock_store.entries[0]["metadata"]["identity_id"] == "alice"

    # Search with threshold 0.40 (alice matches at 0.85, bob filtered out at 0.35)
    matches = adapter.search(test_vec, top_k=2, threshold=0.40)
    assert len(matches) == 1
    assert matches[0].identity_id == "alice"
    assert matches[0].similarity == 0.85
    assert matches[0].rank == 1

    # Search with lower threshold 0.30 (both pass)
    all_matches = adapter.search(test_vec, top_k=2, threshold=0.30)
    assert len(all_matches) == 2
    assert all_matches[1].identity_id == "bob"
