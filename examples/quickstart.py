"""
FaceTrust Quickstart Example: 1:1 Verification, Liveness Check, and 1:N Identification.
"""

from facetrust import FaceTrust

def main():
    # 1. Initialize FaceTrust with default models (BlazeFace + MiniFASNet + ArcFace)
    print("Initializing FaceTrust...")
    auth = FaceTrust()

    print("\n--- 1. Detect Faces ---")
    # detections = auth.detect("sample.jpg")
    # print(f"Found {len(detections)} faces")

    print("\n--- 2. Biometric 1:1 Verification ---")
    # result = auth.verify("id_card.jpg", "selfie.jpg")
    # print(f"Match: {result.is_match}")
    # print(f"Similarity: {result.similarity:.4f}")
    # print(f"Trusted (Match & Real): {result.is_trusted}")

    print("\n--- 3. 1:N Gallery Identification ---")
    # auth.enroll("student_01", "student_01_id.jpg")
    # auth.enroll("student_02", "student_02_id.jpg")
    # matches = auth.identify("probe_camera.jpg", top_k=1)
    # for match in matches:
    #     print(f"Rank {match.rank}: {match.identity_id} (Score: {match.similarity:.4f})")

    print("\nFaceTrust initialized successfully!")

if __name__ == "__main__":
    main()
