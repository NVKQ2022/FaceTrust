"""
LangChain Native Integration: Direct Overrides of LangChain Embeddings and VectorStore.
Allows developers to use FaceTrust straight inside LangChain pipelines and any vendor
vector database (Chroma, Qdrant, Pinecone, FAISS, Milvus, Weaviate, pgvector).
"""

from typing import Any, Callable, Iterable, List, Optional, Tuple, Union
import numpy as np

# Safe fallback imports for environments where langchain_core is not yet installed
try:
    from langchain_core.embeddings import Embeddings as _LangChainEmbeddings
    from langchain_core.vectorstores import VectorStore as _LangChainVectorStore
    from langchain_core.documents import Document
except ImportError:
    from abc import ABC, abstractmethod

    class _LangChainEmbeddings(ABC):
        @abstractmethod
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            pass

        @abstractmethod
        def embed_query(self, text: str) -> List[float]:
            pass

    class _LangChainVectorStore(ABC):
        @abstractmethod
        def add_texts(
            self, texts: Iterable[str], metadatas: Optional[List[dict]] = None, **kwargs: Any
        ) -> List[str]:
            pass

        @abstractmethod
        def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Any]:
            pass

    class Document:
        def __init__(self, page_content: str, metadata: Optional[dict] = None):
            self.page_content = page_content
            self.metadata = metadata or {}


from facetrust.core.client import FaceTrust
from facetrust.core.types import MatchResult
from facetrust.utils.image import ImageCodec


class FaceTrustEmbeddings(_LangChainEmbeddings):
    """
    Direct override of LangChain's Embeddings interface powered by FaceTrust ArcFace 512-D.
    Allows ANY LangChain VectorStore (Chroma, Qdrant, Pinecone, FAISS) to be initialized
    straight with biometric face embeddings.

    Usage:
        >>> from facetrust import FaceTrustEmbeddings
        >>> from langchain_community.vectorstores import Chroma
        >>> db = Chroma(embedding_function=FaceTrustEmbeddings())
    """

    def __init__(self, facetrust: Optional[FaceTrust] = None, **kwargs):
        self._auth = facetrust or FaceTrust(**kwargs)

    def embed_image(
        self,
        image: Union[str, bytes, np.ndarray, Any],
        require_liveness: bool = False,
    ) -> List[float]:
        """Extracts 512-D ArcFace normalized embedding for a single image."""
        emb = self._auth.get_embedding(image, require_liveness=require_liveness)
        return emb.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of image file paths or Base64 strings.
        Implements LangChain Embeddings.embed_documents contract.
        """
        embeddings: List[List[float]] = []
        for text in texts:
            embeddings.append(self.embed_image(text))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a probe image file path or Base64 string.
        Implements LangChain Embeddings.embed_query contract.
        """
        return self.embed_image(text)


class FaceVectorStore(_LangChainVectorStore):
    """
    Direct override of LangChain's VectorStore for Biometric Face Authentication.
    Wraps and extends any underlying LangChain vector database with face-specific methods.

    Usage:
        >>> from facetrust import FaceVectorStore
        >>> from langchain_community.vectorstores import FAISS
        >>> base_store = FAISS(...)
        >>> face_store = FaceVectorStore(base_store)
        >>> face_store.add_faces(images=["alice.jpg"], ids=["alice"])
        >>> matches = face_store.similarity_search_by_face("probe.jpg", k=1)
    """

    def __init__(
        self,
        vectorstore: Any,
        facetrust: Optional[FaceTrust] = None,
        **kwargs,
    ):
        self.vectorstore = vectorstore
        self._auth = facetrust or FaceTrust(**kwargs)

    # --- Overridden LangChain VectorStore Core Methods ---

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Overrides LangChain add_texts by routing to the underlying vector store."""
        return self.vectorstore.add_texts(texts=texts, metadatas=metadatas, **kwargs)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Any]:
        """Overrides LangChain similarity_search by embedding query image path."""
        emb = self._auth.get_embedding(query).tolist()
        return self.similarity_search_by_vector(emb, k=k, **kwargs)

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Any]:
        """Overrides LangChain similarity_search_by_vector."""
        if hasattr(self.vectorstore, "similarity_search_by_vector"):
            return self.vectorstore.similarity_search_by_vector(embedding, k=k, **kwargs)
        raise NotImplementedError("Underlying vectorstore does not support similarity_search_by_vector.")

    # --- Direct Biometric Extension Methods ---

    def add_faces(
        self,
        images: List[Union[str, bytes, np.ndarray, Any]],
        ids: List[str],
        metadatas: Optional[List[dict]] = None,
        require_liveness: bool = True,
    ) -> List[str]:
        """
        Directly enrolls face images with automatic anti-spoofing verification
        and 512-D ArcFace embedding extraction straight into the vector store.
        """
        metadatas = metadatas or [{} for _ in images]
        embeddings: List[List[float]] = []
        texts: List[str] = []
        doc_ids: List[str] = []

        for idx, (img, identity_id) in enumerate(zip(images, ids)):
            emb = self._auth.get_embedding(img, require_liveness=require_liveness)
            embeddings.append(emb.tolist())
            texts.append(f"biometric_identity:{identity_id}")
            doc_ids.append(str(identity_id))

            # Enrich metadata
            metadatas[idx].update({
                "identity_id": str(identity_id),
                "biometric_type": "face",
                "embedding_dim": len(emb),
            })

        # Add precomputed vectors to underlying vector store
        if hasattr(self.vectorstore, "add_embeddings"):
            text_embeddings = list(zip(texts, embeddings))
            self.vectorstore.add_embeddings(
                text_embeddings=text_embeddings,
                metadatas=metadatas,
                ids=doc_ids,
            )
        elif hasattr(self.vectorstore, "add_texts"):
            try:
                self.vectorstore.add_texts(
                    texts=texts,
                    metadatas=metadatas,
                    embeddings=embeddings,
                    ids=doc_ids,
                )
            except (TypeError, NotImplementedError):
                self.vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=doc_ids)

        return doc_ids

    def similarity_search_by_face(
        self,
        probe_image: Union[str, bytes, np.ndarray, Any],
        k: int = 1,
        threshold: float = 0.40,
        require_liveness: bool = True,
    ) -> List[MatchResult]:
        """
        Searches the vector store using a probe face image directly.
        Returns top-k MatchResult instances exceeding the threshold.
        """
        emb = self._auth.get_embedding(probe_image, require_liveness=require_liveness)
        emb_list = emb.tolist()
        matches: List[MatchResult] = []

        if hasattr(self.vectorstore, "similarity_search_by_vector_with_relevance_scores"):
            results = self.vectorstore.similarity_search_by_vector_with_relevance_scores(
                emb_list, k=k
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

        # Fallback to similarity_search_by_vector
        docs = self.similarity_search_by_vector(emb_list, k=k)
        for rank_idx, doc in enumerate(docs, start=1):
            identity_id = getattr(doc, "metadata", {}).get("identity_id", "unknown")
            matches.append(MatchResult(identity_id=identity_id, similarity=1.0, rank=rank_idx))

        return matches
