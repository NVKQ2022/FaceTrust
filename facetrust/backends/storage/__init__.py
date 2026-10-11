"""
Storage backends for 1:N identity galleries and vector search.
"""

from .memory import MemoryStorage
from .langchain import LangChainVectorStoreAdapter

__all__ = ["MemoryStorage", "LangChainVectorStoreAdapter"]
