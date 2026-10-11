# 🛡️ FaceTrust

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/facetrust/)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-brightgreen.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Production-Grade, Modular Face Authentication with Built-in Anti-Spoofing & ArcFace.**

**FaceTrust** is an extensible, high-performance biometric authentication library designed for production systems. It combines sub-millisecond face detection, deep liveness anti-spoofing (detecting 2D print & screen replay attacks), canonical landmark alignment, and 512-D ArcFace hypersphere embedding extraction into a single, intuitive Python API.

---

## 🌟 Highlights

- ⚡ **Zero-Friction Quickstart**: 1:1 Face Verification in just 2 lines of code.
- 🛡️ **Built-in Presentation Attack Defense**: MiniFASNetV2 with $2.7\times$ context expansion filters out phone screens, tablets, and printed photos.
- 🔄 **Plug-and-Play Swappability**: Swap detectors, anti-spoofer models, alignment strategies, and feature extractors at runtime without modifying library code.
- 🚀 **Hardware Accelerated**: Native CPU and GPU execution providers via ONNX Runtime (`CUDAExecutionProvider`, `TensorrtExecutionProvider`).
- 🤖 **Autonomous Model Cache**: Weights are downloaded automatically on first use with integrity verification and cached to `~/.cache/facetrust/models/`.
- 🌐 **Async Native**: Built-in non-blocking asynchronous methods (`await auth.a_verify(...)`) designed for FastAPI and async backends.
- 🗄️ **1:N Scalable Gallery**: Enroll identities and perform sub-millisecond vector matching across thousands of faces.

---

## 📦 Installation

```bash
# Standard CPU installation
pip install facetrust

# GPU-accelerated installation (CUDA / TensorRT)
pip install facetrust[gpu]
```

---

## 🚀 Quickstart: 2-Line Verification

```python
from facetrust import FaceTrust

# 1. Initialize FaceTrust (downloads models automatically on first run)
auth = FaceTrust()

# 2. Perform 1:1 Face Verification with built-in Anti-Spoofing
result = auth.verify("id_card.jpg", "live_selfie.jpg")

print(f"Match: {result.is_match}")          # True / False
print(f"Similarity: {result.similarity}")    # e.g., 0.78
print(f"Trusted: {result.is_trusted}")      # True (Matched & Real Face)
```

---

## 🔄 Dynamic Model Switching

FaceTrust adheres strictly to the **Open-Closed Principle**: you can swap model backends dynamically using string identifiers or inject your own custom neural network classes.

### 1. Using Built-in Backends
```python
auth = FaceTrust(
    detector="blazeface",           # "blazeface" (sub-15ms edge detector)
    antispoof="minifasnet",         # "minifasnet" (2.7x context liveness)
    embedder="arcface_mobile",      # "arcface_mobile" (512-D w600k_mbf)
    device="cuda",                  # "cpu" | "cuda" | "tensorrt"
)
```

### 2. Registering Custom Model Plugins
You can extend FaceTrust with custom PyTorch, TensorRT, or ONNX models without modifying the library:

```python
import numpy as np
from facetrust import FaceTrust, ModelRegistry
from facetrust.backends.base import BaseEmbedder

# Register your proprietary model via decorator
@ModelRegistry.register_embedder("my_resnet50")
class MyCustomEmbedder(BaseEmbedder):
    @property
    def name(self) -> str:
        return "Custom ResNet50"

    @property
    def embedding_dim(self) -> int:
        return 512

    def extract(self, tensor_nchw: np.ndarray, normalize: bool = True) -> np.ndarray:
        # Custom model inference logic
        return my_torch_model(tensor_nchw)

# Now use it seamlessly in the pipeline
auth = FaceTrust(embedder="my_resnet50")
```

---

## 🗄️ 1:N Identity Enrollment & Matching

```python
from facetrust import FaceTrust

auth = FaceTrust()

# Enroll identities into the gallery
auth.enroll("alice", "alice_passport.jpg")
auth.enroll("bob", "bob_profile.jpg")

# Identify probe photo against all enrolled faces
matches = auth.identify("security_camera_probe.jpg", top_k=1)

for match in matches:
    print(f"Rank {match.rank}: Identity '{match.identity_id}' (Similarity: {match.similarity:.4f})")
```

---

## 🦜🔗 LangChain VectorStore Integration (50+ Database Backends)

FaceTrust seamlessly integrates with **any LangChain `VectorStore`** (Qdrant, Pinecone, Chroma, Milvus, Weaviate, FAISS, pgvector, Redis, OpenSearch). Simply pass your existing LangChain store—FaceTrust will automatically detect and adapt it:

```bash
pip install facetrust[langchain]
```

```python
from langchain_community.vectorstores import Qdrant # or Chroma, Pinecone, FAISS
from facetrust import FaceTrust

# Pass any existing LangChain VectorStore directly to FaceTrust:
qdrant_store = Qdrant(...)
auth = FaceTrust(storage=qdrant_store)

# 1. Enroll identities directly into Qdrant/Pinecone
auth.enroll("alice", "alice.jpg")

# 2. Search probe face against millions of records in your cluster
matches = auth.identify("cctv_frame.jpg", top_k=5)
```

---

## ⚡ Asynchronous FastAPI Integration

```python
from fastapi import FastAPI, UploadFile, File
from facetrust import FaceTrust

app = FastAPI()
auth = FaceTrust()

@app.post("/verify")
async def verify_faces(
    id_card: UploadFile = File(...),
    selfie: UploadFile = File(...),
):
    id_bytes = await id_card.read()
    selfie_bytes = await selfie.read()

    # Non-blocking async verification
    result = await auth.a_verify(id_bytes, selfie_bytes)
    return {
        "trusted": result.is_trusted,
        "match": result.is_match,
        "similarity": result.similarity,
        "status": result.status,
    }
```

---

## 🔬 The 5-Stage Biometric Pipeline

```mermaid
flowchart LR
    A["Whole Frame<br/>(Uncropped)"] --> B["Stage 1: Detection<br/>MediaPipe BlazeFace"]
    B --> C["Stage 2: Anti-Spoofing<br/>MiniFASNetV2 (2.7x)"]
    C --> D["Stage 3: Alignment<br/>4-Pt Partial Affine"]
    D --> E["Stage 4: Embedding<br/>ArcFace (512-D)"]
    E --> F["Stage 5: Matching<br/>Cosine Similarity"]
```

1. **Stage 1: Face Detection (MediaPipe BlazeFace)**: Sub-15ms edge CPU detector producing bounding boxes and 6 landmark anchors.
2. **Stage 2: Anti-Spoofing & Liveness (MiniFASNetV2)**: Uses a $2.7\times$ context-expanded crop to detect screen bezels, printed paper margins, and moiré grid frequencies ($P(\text{real}) \ge 0.60$).
3. **Stage 3: Landmark-Based Face Alignment**: 4-point partial affine similarity transformation (`cv2.estimateAffinePartial2D`) mapping eyes, nose, and mouth to canonical $112 \times 112$ coordinates.
4. **Stage 4: Deep Embedding Extraction (ArcFace MobileFaceNet)**: Converts aligned crops to normalized 512-dimensional feature vectors on a unit hypersphere ($\|\mathbf{e}\|_2 = 1.0$).
5. **Stage 5: Biometric Matching**: Cosine similarity inner product $\mathbf{e}_1 \cdot \mathbf{e}_2 \in [-1.0, 1.0]$ with default acceptance threshold $\tau = 0.40$.

---

## 📊 Key Metrics & Thresholds

| Metric | Value | Description |
| :--- | :---: | :--- |
| **Detection Latency** | $\approx 10 - 15\text{ ms}$ | BlazeFace CPU edge execution |
| **Liveness Resolution** | $80 \times 80$ | MiniFASNet raw BGR tensor input |
| **Liveness Context Scale** | $2.7\times$ | Multiplier for face bounding box context |
| **Liveness Threshold** | $0.60$ | Minimum confidence for genuine live human |
| **Aligned Face Canvas** | $112 \times 112$ | Standard canonical ArcFace resolution |
| **Embedding Vector** | $512\text{-D}$ | L2-normalized float32 hypersphere vector |
| **Similarity Threshold** | $0.40$ | Default positive match boundary |
| **Genuine Match Range** | $0.55 - 0.85$ | Typical score range for the same individual |
| **Imposter Match Range** | $< 0.30$ | Typical score range for different people |

---

## 🧪 Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run test suite
pytest tests/ -v
```

---

## 🗺️ Roadmap & Commercial Vision

FaceTrust follows an **Open-Core** model designed for enterprise scalability and commercial adoption. See our detailed engineering blueprint:
- 📖 **[Product & Engineering Roadmap](ROADMAP.md)**: Details on Multi-Frame Temporal Liveness, Active Blink/Head Pose challenges, Vector Database scaling (Qdrant / Milvus), rPPG cardiovascular blood pulse detection, and ISO 30107-3 compliance.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

