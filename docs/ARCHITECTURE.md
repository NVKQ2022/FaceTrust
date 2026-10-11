# 🏛️ FaceTrust System Architecture & Theoretical Foundations

This document details the architectural design patterns, mathematical formulas, and computer vision pipelines powering **FaceTrust**.

---

## 📐 1. End-to-End Biometric Pipeline

FaceTrust implements a sequential, failure-early 5-stage biometric verification pipeline:

```mermaid
flowchart TD
    A["Raw Input Frame<br/>(File Path / Bytes / RGB Array)"] --> B["ImageCodec<br/>Universal Ingestion & Resizing"]
    B --> C["Stage 1: Face Detection<br/>MediaPipe BlazeFace Anchor Regressor"]
    C -->|No Face Detected| FAIL1["BiometricFailure('No face detected')"]
    C -->|Face Box + 6 Keypoints| D["Stage 2: Anti-Spoofing Defense<br/>MiniFASNetV2 (2.7x Context Crop)"]
    D -->|Spoof Attack Detected| FAIL2["BiometricFailure('Spoof attack: 2D Screen/Paper')"]
    D -->|P(real) >= 0.60| E["Stage 3: Canonical Alignment<br/>4-Point Partial Affine (112x112)"]
    E --> F["Stage 4: Feature Extraction<br/>ArcFace MobileFaceNet (512-D Hypersphere)"]
    F --> G["Stage 5: Biometric Matching / Vector Store<br/>Cosine Metric & LangChain Adapter"]
```

---

## 🔬 2. Mathematical Foundations

### 2.1 Stage 1: Face Detection (MediaPipe BlazeFace)
BlazeFace is a lightweight anchor-based detector optimized for mobile and CPU inference:
- Generates bounding box coordinates $(x_{\min}, y_{\min}, w, h)$.
- Extracts **6 key facial anchors**:
  1. Right Eye: $(x_{re}, y_{re})$
  2. Left Eye: $(x_{le}, y_{le})$
  3. Nose Tip: $(x_n, y_n)$
  4. Mouth Center: $(x_m, y_m)$
  5. Right Ear Tragion: $(x_{rt}, y_{rt})$
  6. Left Ear Tragion: $(x_{lt}, y_{lt})$

### 2.2 Stage 2: Presentation Attack Detection (MiniFASNetV2 & Context Expansion)
Attackers commonly spoof systems using printed photos on paper or digital replay on smartphone/tablet displays.

#### Context Expansion Formula ($2.7\times$)
Cropping tightly around the face removes crucial spoof clues (display edges, bezels, fingers, paper margins). FaceTrust computes an expanded contextual bounding box:

$$\text{box}_{2.7} = \left[ c_x - 1.35 \cdot w, \; c_y - 1.35 \cdot h, \; c_x + 1.35 \cdot w, \; c_y + 1.35 \cdot h \right]$$

where $(c_x, c_y)$ is the face center, and $(w, h)$ are the detected face dimensions.

#### Multi-Scale Frequency Analysis
MiniFASNet evaluates Central Difference Convolutions (CDC) to capture high-frequency texture distortions, pixel grid structures (Moiré patterns), and chromatic aberration:

$$P(\text{real}) = \frac{e^{z_{\text{real}}}}{\sum_{j \in \{\text{print}, \text{real}, \text{replay}\}} e^{z_j}} \ge 0.60$$

### 2.3 Stage 3: Canonical Landmark Alignment (Partial Affine)
To eliminate variance from head tilt (roll), camera perspective (pitch/yaw), and subject distance, FaceTrust maps detected facial landmarks onto standard ArcFace coordinates:

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} s \cos \theta & -s \sin \theta \\ s \sin \theta & s \cos \theta \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} t_x \\ t_y \end{bmatrix}$$

Using 4 anchor points (Right Eye, Left Eye, Nose Tip, Mouth Center) and the canonical $112 \times 112$ reference landmarks:
- **Right Eye**: $(38.29, 51.70)$
- **Left Eye**: $(73.53, 51.70)$
- **Nose Tip**: $(56.03, 71.74)$
- **Mouth Center**: $(56.03, 92.37)$

The transformation matrix is estimated via least-squares similarity transformation (`cv2.estimateAffinePartial2D`), warping the face into a normalized $112 \times 112 \times 3$ RGB tensor.

### 2.4 Stage 4: ArcFace Deep Embedding Extraction
ArcFace (Additive Angular Margin Loss) penalizes the geodesic distance on a normalized hypersphere:

$$\mathcal{L}_{\text{ArcFace}} = -\log \frac{e^{s \cos(\theta_{y_i} + m)}}{e^{s \cos(\theta_{y_i} + m)} + \sum_{j \ne y_i} e^{s \cos \theta_j}}$$

where $s$ is the hypersphere radius (typically $64$), and $m$ is the angular margin penalty (typically $0.50$).

The resulting feature vector $\mathbf{v} \in \mathbb{R}^{512}$ is L2-normalized:

$$\mathbf{e} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}, \quad \text{such that } \|\mathbf{e}\|_2 = 1.0$$

### 2.5 Stage 5: Biometric Verification Metric
Because all feature vectors reside on the 511-dimensional unit hypersphere, the cosine similarity simplifies to an inner product:

$$\text{Sim}(\mathbf{e}_1, \mathbf{e}_2) = \frac{\mathbf{e}_1 \cdot \mathbf{e}_2}{\|\mathbf{e}_1\|_2 \|\mathbf{e}_2\|_2} = \mathbf{e}_1^T \mathbf{e}_2$$

The match decision function:

$$\text{Match}(\mathbf{e}_1, \mathbf{e}_2) = \begin{cases} \text{True}, & \text{Sim}(\mathbf{e}_1, \mathbf{e}_2) \ge \tau \quad (\tau = 0.40) \\ \text{False}, & \text{otherwise} \end{cases}$$

---

## 🏛️ 3. Software Design Patterns

### 3.1 Facade Pattern (`FaceTrust`)
The top-level `FaceTrust` class provides a unified, simple API (`auth.verify()`, `auth.enroll()`, `auth.identify()`), abstracting the complexity of 5 sub-models, tensor normalization, and vector databases.

### 3.2 Registry & Dependency Injection (`ModelRegistry`)
Backends implement abstract base classes (`BaseDetector`, `BaseAntispoof`, `BaseAligner`, `BaseEmbedder`, `BaseStorage`).
New implementations can be registered at runtime without modifying library code:

```python
@ModelRegistry.register_embedder("custom_arcface")
class CustomArcFace(BaseEmbedder):
    ...
```

### 3.3 Adapter Pattern & Native Overrides (`LangChain`)
FaceTrust provides two integration levels with LangChain:
1. **Adapter**: Auto-wraps existing `VectorStore` instances via `LangChainVectorStoreAdapter`.
2. **Native Overrides**:
   - `FaceTrustEmbeddings`: Overrides LangChain `Embeddings` (`embed_documents`, `embed_query`).
   - `FaceVectorStore`: Overrides LangChain `VectorStore` (`add_texts`, `similarity_search`, `similarity_search_by_vector`) with face-native operations (`add_faces`, `similarity_search_by_face`).

---

## ⚡ 4. Runtime & Performance Optimizations

1. **Persistent ONNX Sessions**: ONNX Runtime sessions (`InferenceSession`) are instantiated once during initialization and reused across inferences, avoiding repeated graph parsing overhead.
2. **Execution Provider Fallback**: Automatically tries `CUDAExecutionProvider` or `TensorrtExecutionProvider`, falling back to `CPUExecutionProvider` seamlessly.
3. **Thread Safety**: Model extractors and vector storages use immutable pre-allocated tensors and stateless inference calls, allowing safe execution inside multithreaded web servers (Gunicorn/Uvicorn).
