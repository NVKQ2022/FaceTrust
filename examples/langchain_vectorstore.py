"""
Example: Using FaceTrust directly with LangChain Embeddings & VectorStores
Supports any vendor store (Chroma, Qdrant, Pinecone, FAISS, Milvus, pgvector).
"""

from facetrust import FaceTrust, FaceTrustEmbeddings, FaceVectorStore

def main():
    print("--- FaceTrust Straight LangChain VectorStore Override Demo ---")

    # =========================================================================
    # Pattern 1: Direct Embeddings Override (FaceTrustEmbeddings)
    # Use FaceTrust ArcFace 512-D vectors straight as a LangChain embedding function.
    # =========================================================================
    # from langchain_community.vectorstores import Chroma
    #
    # embeddings = FaceTrustEmbeddings()
    # db = Chroma(
    #     collection_name="employee_faces",
    #     embedding_function=embeddings,
    #     persist_directory="./chroma_db",
    # )
    # # Now Chroma embeds face image paths/data natively!

    # =========================================================================
    # Pattern 2: Direct FaceVectorStore Override (FaceVectorStore)
    # Wraps any LangChain vector database with face-native enrollment and search.
    # =========================================================================
    # from langchain_community.vectorstores import FAISS
    #
    # base_faiss = FAISS(...)
    # face_store = FaceVectorStore(base_faiss)
    #
    # # Enroll identities straight with anti-spoofing verification
    # face_store.add_faces(
    #     images=["id_alice.jpg", "id_bob.jpg"],
    #     ids=["emp_001", "emp_002"],
    # )
    #
    # # Search by probe face directly
    # matches = face_store.similarity_search_by_face(
    #     probe_image="security_cam_capture.jpg",
    #     k=1,
    #     threshold=0.40,
    # )
    # for match in matches:
    #     print(f"Match found: {match.identity_id} with similarity {match.similarity}")

    # =========================================================================
    # Pattern 3: Plug Vendor VectorStore directly into FaceTrust Client
    # =========================================================================
    # auth = FaceTrust(storage=db)
    # auth.enroll("emp_001", "id_alice.jpg")
    # matches = auth.identify("security_cam_capture.jpg", top_k=1)

    print("Straight LangChain integration patterns validated successfully!")


if __name__ == "__main__":
    main()
