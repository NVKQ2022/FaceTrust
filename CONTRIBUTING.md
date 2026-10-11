# 🤝 Contributing to FaceTrust

Thank you for your interest in contributing to **FaceTrust**! We welcome community contributions, from bug fixes and documentation improvements to new neural network backends and vector database adapters.

---

## 🛠️ 1. Development Setup

### 1.1 Clone the Repository
```bash
git clone https://github.com/NVKQ2022/FaceTrust.git
cd FaceTrust
```

### 1.2 Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 1.3 Install Dependencies
Install FaceTrust in editable mode with all development dependencies:
```bash
pip install -e ".[dev,langchain]"
```

---

## 🧪 2. Running Tests

Before submitting any code changes, ensure all unit tests pass cleanly:

```bash
pytest tests/ -v
```

To run with coverage reporting:
```bash
pytest tests/ --cov=facetrust --cov-report=term-missing
```

---

## 🔌 3. Adding a New Model Backend / Plugin

FaceTrust is built upon the **Open-Closed Principle**. You can introduce new computer vision models without modifying existing pipeline code.

### 3.1 Implementing a Custom Embedder
Subclass `BaseEmbedder` from `facetrust.backends.base` and decorate with `@ModelRegistry.register_embedder`:

```python
import numpy as np
from facetrust.backends.base import BaseEmbedder
from facetrust.core.registry import ModelRegistry

@ModelRegistry.register_embedder("my_custom_embedder")
class MyCustomEmbedder(BaseEmbedder):
    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.device = device
        # Initialize model session / weights here

    @property
    def name(self) -> str:
        return "My Custom Embedder"

    @property
    def embedding_dim(self) -> int:
        return 512

    def extract(self, tensor_nchw: np.ndarray, normalize: bool = True) -> np.ndarray:
        # Run inference and return (512,) float32 vector
        embedding = ...
        if normalize:
            embedding = embedding / np.linalg.norm(embedding)
        return embedding
```

### 3.2 Implementing a Custom Storage Backend
Subclass `BaseStorage` from `facetrust.backends.base` and decorate with `@ModelRegistry.register_storage`:

```python
from typing import List, Optional
import numpy as np
from facetrust.backends.base import BaseStorage
from facetrust.core.registry import ModelRegistry
from facetrust.core.types import MatchResult

@ModelRegistry.register_storage("my_storage")
class MyStorage(BaseStorage):
    def enroll(self, identity_id: str, embedding: np.ndarray, metadata: Optional[dict] = None) -> bool:
        ...

    def search(self, probe_embedding: np.ndarray, top_k: int = 1, threshold: float = 0.40) -> List[MatchResult]:
        ...

    def delete(self, identity_id: str) -> bool:
        ...
```

---

## 📐 4. Code Style & Standards

- **Formatting**: Format all Python code with `black` (88 character line limit).
- **Type Hints**: All public functions and methods must include Python 3.9+ type annotations (`typing`).
- **Docstrings**: Provide clear Google-style or NumPy-style docstrings for public classes and methods.
- **Atomic Commits**: Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
  - `feat(...)`: New features or capabilities
  - `fix(...)`: Bug fixes
  - `docs(...)`: Documentation updates
  - `test(...)`: Unit tests
  - `refactor(...)`: Code refactoring without behavioral changes

---

## 📬 5. Submitting Pull Requests

1. Fork the repository and create your branch from `main`:
   ```bash
   git checkout -b feat/add-retinaface-detector
   ```
2. Write unit tests for your changes under the `tests/` directory.
3. Verify all tests pass: `pytest tests/`.
4. Commit your changes with clear, descriptive commit messages.
5. Push to your fork and submit a Pull Request to `NVKQ2022/FaceTrust:main`.
