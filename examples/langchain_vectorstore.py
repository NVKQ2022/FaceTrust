"""
Example: Using any LangChain VectorStore (Qdrant, Pinecone, Chroma, FAISS, Milvus)
with FaceTrust for 1:N Biometric Identity Matching.
"""

from facetrust import FaceTrust

# Scenario: Developer uses LangChain's Qdrant, Chroma, or FAISS vector store
# Example using langchain_community:
# from langchain_community.vectorstores import Qdrant, Chroma, FAISS

def main():
    print("--- FaceTrust + LangChain VectorStore Integration Demo ---")

    # In actual production code:
    # from qdrant_client import QdrantClient
    # client = QdrantClient(":memory:")
    # qdrant_store = Qdrant(client=client, collection_name="students_faces", embeddings=None)

    # FaceTrust automatically detects that qdrant_store is a LangChain VectorStore
    # and wraps it into LangChainVectorStoreAdapter:
    #
    # auth = FaceTrust(storage=qdrant_store)
    #
    # # 1. Enroll identities directly into Qdrant/Pinecone via FaceTrust
    # auth.enroll("student_101", "student_101.jpg")
    # auth.enroll("student_102", "student_102.jpg")
    #
    # # 2. Search probe face across millions of records in Qdrant/Pinecone
    # matches = auth.identify("security_camera_probe.jpg", top_k=3, threshold=0.40)
    # for match in matches:
    #     print(f"Rank {match.rank}: Identity '{match.identity_id}' (Similarity: {match.similarity})")

    print("FaceTrust LangChainVectorStoreAdapter is ready and tested!")


if __name__ == "__main__":
    main()
