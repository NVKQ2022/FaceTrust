"""
Tests for FaceTrustEmbeddings and FaceVectorStore overriding LangChain native contracts.
"""

from typing import List, Tuple
from unittest.mock import MagicMock
import numpy as np

from facetrust.integrations.langchain import FaceTrustEmbeddings, FaceVectorStore
from facetrust.core.types import MatchResult


class MockDocument:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


class MockVendorVectorStore:
    """Simulates a vendor VectorStore (Chroma/FAISS/Qdrant) inside LangChain."""

    def __init__(self):
        self.texts = []
        self.metadatas = []
        self.embeddings = []
        self.ids = []

    def add_texts(self, texts, metadatas=None, **kwargs):
        self.texts.extend(texts)
        if metadatas:
            self.metadatas.extend(metadatas)
        if "embeddings" in kwargs:
            self.embeddings.extend(kwargs["embeddings"])
        if "ids" in kwargs:
            self.ids.extend(kwargs["ids"])
        return kwargs.get("ids", ["doc_0"])

    def add_embeddings(self, text_embeddings, metadatas=None, ids=None):
        for text, emb in text_embeddings:
            self.texts.append(text)
            self.embeddings.append(emb)
        if metadatas:
            self.metadatas.extend(metadatas)
        if ids:
            self.ids.extend(ids)
        return ids

    def similarity_search_by_vector(self, embedding: List[float], k: int = 4, **kwargs):
        return [
            MockDocument("biometric_identity:user_1", {"identity_id": "user_1"})
        ][:k]

    def similarity_search_by_vector_with_relevance_scores(
        self, embedding: List[float], k: int = 4, **kwargs
    ) -> List[Tuple[MockDocument, float]]:
        return [
            (MockDocument("biometric_identity:alice", {"identity_id": "alice"}), 0.88),
            (MockDocument("biometric_identity:eve", {"identity_id": "eve"}), 0.25),
        ][:k]


def test_facetrust_embeddings_override():
    mock_auth = MagicMock()
    dummy_vec = np.ones(512, dtype=np.float32)
    mock_auth.get_embedding.return_value = dummy_vec

    embeddings = FaceTrustEmbeddings(facetrust=mock_auth)

    # 1. embed_image
    emb = embeddings.embed_image("test_face.jpg")
    assert len(emb) == 512
    assert isinstance(emb, list)
    mock_auth.get_embedding.assert_called_with("test_face.jpg", require_liveness=False)

    # 2. embed_documents
    doc_embs = embeddings.embed_documents(["face1.jpg", "face2.jpg"])
    assert len(doc_embs) == 2
    assert len(doc_embs[0]) == 512

    # 3. embed_query
    q_emb = embeddings.embed_query("query_face.jpg")
    assert len(q_emb) == 512


def test_face_vectorstore_override():
    mock_auth = MagicMock()
    dummy_vec = np.ones(512, dtype=np.float32)
    mock_auth.get_embedding.return_value = dummy_vec

    vendor_db = MockVendorVectorStore()
    face_store = FaceVectorStore(vendor_db, facetrust=mock_auth)

    # 1. add_faces
    ids = face_store.add_faces(images=["face_alice.jpg"], ids=["alice"])
    assert ids == ["alice"]
    assert len(vendor_db.texts) == 1
    assert "biometric_identity:alice" in vendor_db.texts[0]
    assert vendor_db.metadatas[0]["identity_id"] == "alice"

    # 2. similarity_search_by_face
    matches = face_store.similarity_search_by_face("probe.jpg", k=2, threshold=0.40)
    assert len(matches) == 1
    assert isinstance(matches[0], MatchResult)
    assert matches[0].identity_id == "alice"
    assert matches[0].similarity == 0.88
    assert matches[0].rank == 1

    # 3. similarity_search_by_face with lower threshold
    all_matches = face_store.similarity_search_by_face("probe.jpg", k=2, threshold=0.20)
    assert len(all_matches) == 2
    assert all_matches[1].identity_id == "eve"
