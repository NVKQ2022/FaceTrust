"""
LangChain VectorStore Adapter: Enables FaceTrust to connect with 50+ vector database
providers (Pinecone, Qdrant, Chroma, Milvus, Weaviate, FAISS, pgvector, Redis, OpenSearch)
by leveraging existing LangChain VectorStore instances.
"""

from typing import Any, List, Optional
import numpy as np

from facetrust.backends.base.storage import BaseStorage
from facetrust.core.registry import ModelRegistry
from facetrust.core.types import MatchResult


@ModelRegistry.register_storage("langchain")
class LangChainVectorStoreAdapter(BaseStorage):
    """
    Adapter pattern mapping FaceTrust 512-D biometric face embeddings
    into any LangChain VectorStore implementation.

    Usage:
        >>> from langchain_community.vectorstores import Qdrant # or Chroma, Pinecone, FAISS
        >>> qdrant_store = Qdrant(...)
        >>> auth = FaceTrust(storage=qdrant_store)  # Auto-wrapped!
    """

    def __init__(self, vectorstore: Any, **kwargs):
        """
        Args:
            vectorstore: Any LangChain VectorStore instance.
        """
        self.vectorstore = vectorstore

    def enroll(self, identity_id: str, embedding: np.ndarray) -> None:
        """
        Enrolls an identity embedding into the underlying LangChain vector store.
        """
        emb_list = np.asarray(embedding, dtype=np.float32).flatten().tolist()
        metadata = {"identity_id": str(identity_id), "type": "facetrust_biometric"}
        text_content = f"biometric_identity:{identity_id}"
        doc_id = str(identity_id)

        # 1. Check for add_embeddings (direct pre-computed vectors)
        if hasattr(self.vectorstore, "add_embeddings"):
            try:
                self.vectorstore.add_embeddings(
                    text_embeddings=[(text_content, emb_list)],
                    metadatas=[metadata],
                    ids=[doc_id],
                )
                return
            except (TypeError, NotImplementedError):
                pass

        # 2. Check for add_texts with embeddings parameter
        if hasattr(self.vectorstore, "add_texts"):
            try:
                self.vectorstore.add_texts(
                    texts=[text_content],
                    metadatas=[metadata],
                    embeddings=[emb_list],
                    ids=[doc_id],
                )
                return
            except (TypeError, NotImplementedError):
                pass

            # 3. Fallback: add_texts standard call
            self.vectorstore.add_texts(
                texts=[text_content],
                metadatas=[metadata],
                ids=[doc_id],
            )

    def get(self, identity_id: str) -> Optional[np.ndarray]:
        """
        LangChain VectorStore implementations typically do not expose
        universal get_by_id across all 50+ backends.
        """
        return None

    def delete(self, identity_id: str) -> bool:
        """Deletes an identity from the vector store if supported."""
        if hasattr(self.vectorstore, "delete"):
            try:
                self.vectorstore.delete(ids=[str(identity_id)])
                return True
            except Exception:
                pass
        return False

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 1,
        threshold: float = 0.40,
    ) -> List[MatchResult]:
        """
        Performs vector similarity search against the LangChain store using
        the 512-D ArcFace query vector.
        """
        emb_list = np.asarray(query_embedding, dtype=np.float32).flatten().tolist()
        matches: List[MatchResult] = []

        # 1. Best: similarity_search_by_vector_with_relevance_scores
        if hasattr(self.vectorstore, "similarity_search_by_vector_with_relevance_scores"):
            results = self.vectorstore.similarity_search_by_vector_with_relevance_scores(
                emb_list, k=top_k
            )
            for rank_idx, (doc, score) in enumerate(results, start=1):
                score_f = float(score)
                if score_f >= threshold:
                    identity_id = doc.metadata.get("identity_id") or doc.page_content.replace(
                        "biometric_identity:", ""
                    )
                    matches.append(
                        MatchResult(
                            identity_id=identity_id,
                            similarity=round(score_f, 4),
                            rank=rank_idx,
                        )
                    )
            return matches

        # 2. Fallback: similarity_search_by_vector_with_score
        if hasattr(self.vectorstore, "similarity_search_by_vector_with_score"):
            results = self.vectorstore.similarity_search_by_vector_with_score(
                emb_list, k=top_k
            )
            for rank_idx, (doc, score) in enumerate(results, start=1):
                score_f = float(score)
                if score_f >= threshold:
                    identity_id = doc.metadata.get("identity_id") or doc.page_content.replace(
                        "biometric_identity:", ""
                    )
                    matches.append(
                        MatchResult(
                            identity_id=identity_id,
                            similarity=round(score_f, 4),
                            rank=rank_idx,
                        )
                    )
            return matches

        # 3. Fallback: similarity_search_by_vector (unscored)
        if hasattr(self.vectorstore, "similarity_search_by_vector"):
            docs = self.vectorstore.similarity_search_by_vector(emb_list, k=top_k)
            for rank_idx, doc in enumerate(docs, start=1):
                identity_id = doc.metadata.get("identity_id") or doc.page_content.replace(
                    "biometric_identity:", ""
                )
                matches.append(
                    MatchResult(
                        identity_id=identity_id,
                        similarity=1.0,
                        rank=rank_idx,
                    )
                )
            return matches

        return []

    def count(self) -> int:
        if hasattr(self.vectorstore, "__len__"):
            return len(self.vectorstore)
        return -1
