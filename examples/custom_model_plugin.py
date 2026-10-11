"""
Example: Creating and Registering a Custom Model Plugin in FaceTrust
without modifying internal library code.
"""

import numpy as np
from facetrust import FaceTrust, ModelRegistry
from facetrust.backends.base import BaseEmbedder


# 1. Define custom model inheriting from BaseEmbedder
@ModelRegistry.register_embedder("dummy_embedder")
class MyProprietaryEmbedder(BaseEmbedder):
    """Custom embedding extractor plugin."""

    @property
    def name(self) -> str:
        return "My Proprietary ResNet"

    @property
    def embedding_dim(self) -> int:
        return 512

    def extract(self, tensor_nchw: np.ndarray, normalize: bool = True) -> np.ndarray:
        # Custom model inference logic
        np.random.seed(42)
        vec = np.random.randn(512).astype(np.float32)
        if normalize:
            vec /= np.linalg.norm(vec)
        return vec


def main():
    print("Available embedders:", ModelRegistry.list_available()["embedders"])

    # 2. Use custom model via string identifier
    auth = FaceTrust(embedder="dummy_embedder")
    print(f"Using embedder: {auth.embedder.name}")

    # 3. Or inject directly as an object instance
    custom_instance = MyProprietaryEmbedder()
    auth2 = FaceTrust(embedder=custom_instance)
    print(f"Injected instance: {auth2.embedder.name}")


if __name__ == "__main__":
    main()
