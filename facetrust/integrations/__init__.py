"""
FaceTrust Integrations Module.
Provides native bridges and overrides for major AI ecosystems (LangChain, LlamaIndex, etc.).
"""

from facetrust.integrations.langchain import FaceTrustEmbeddings, FaceVectorStore

__all__ = [
    "FaceTrustEmbeddings",
    "FaceVectorStore",
]
