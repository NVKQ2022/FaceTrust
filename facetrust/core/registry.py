"""
Dynamic Plugin & Model Registry supporting runtime discovery, custom registration,
and Dependency Injection for all FaceTrust sub-backends.
"""

from typing import Any, Callable, Dict, Optional, Type, Union
from facetrust.backends.base import (
    BaseAligner,
    BaseAntiSpoofer,
    BaseDetector,
    BaseEmbedder,
    BaseStorage,
)


class ModelRegistry:
    """Central registry for dynamically swappable algorithms and backends."""

    _detectors: Dict[str, Type[BaseDetector]] = {}
    _antispoofers: Dict[str, Type[BaseAntiSpoofer]] = {}
    _aligners: Dict[str, Type[BaseAligner]] = {}
    _embedders: Dict[str, Type[BaseEmbedder]] = {}
    _storages: Dict[str, Type[BaseStorage]] = {}

    # --- Decorator Registrars ---

    @classmethod
    def register_detector(cls, name: str) -> Callable[[Type[BaseDetector]], Type[BaseDetector]]:
        def decorator(subclass: Type[BaseDetector]) -> Type[BaseDetector]:
            cls._detectors[name.lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def register_antispoof(cls, name: str) -> Callable[[Type[BaseAntiSpoofer]], Type[BaseAntiSpoofer]]:
        def decorator(subclass: Type[BaseAntiSpoofer]) -> Type[BaseAntiSpoofer]:
            cls._antispoofers[name.lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def register_aligner(cls, name: str) -> Callable[[Type[BaseAligner]], Type[BaseAligner]]:
        def decorator(subclass: Type[BaseAligner]) -> Type[BaseAligner]:
            cls._aligners[name.lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def register_embedder(cls, name: str) -> Callable[[Type[BaseEmbedder]], Type[BaseEmbedder]]:
        def decorator(subclass: Type[BaseEmbedder]) -> Type[BaseEmbedder]:
            cls._embedders[name.lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def register_storage(cls, name: str) -> Callable[[Type[BaseStorage]], Type[BaseStorage]]:
        def decorator(subclass: Type[BaseStorage]) -> Type[BaseStorage]:
            cls._storages[name.lower()] = subclass
            return subclass
        return decorator

    # --- Factory Creators (Support both Name Strings & Custom Instance DI) ---

    @classmethod
    def _instantiate(cls, target_cls: Type[Any], **kwargs: Any) -> Any:
        try:
            return target_cls(**kwargs)
        except TypeError:
            import inspect
            sig = inspect.signature(target_cls.__init__)
            valid = set(sig.parameters.keys())
            filtered = {k: v for k, v in kwargs.items() if k in valid}
            return target_cls(**filtered)

    @classmethod
    def create_detector(
        cls,
        target: Union[str, BaseDetector],
        **kwargs: Any,
    ) -> BaseDetector:
        if isinstance(target, BaseDetector):
            return target
        name = target.lower()
        if name not in cls._detectors:
            raise KeyError(f"Detector '{name}' not found. Available: {list(cls._detectors.keys())}")
        return cls._instantiate(cls._detectors[name], **kwargs)

    @classmethod
    def create_antispoof(
        cls,
        target: Optional[Union[str, BaseAntiSpoofer]],
        **kwargs: Any,
    ) -> Optional[BaseAntiSpoofer]:
        if target is None:
            return None
        if isinstance(target, BaseAntiSpoofer):
            return target
        name = target.lower()
        if name not in cls._antispoofers:
            raise KeyError(f"Anti-spoofer '{name}' not found. Available: {list(cls._antispoofers.keys())}")
        return cls._instantiate(cls._antispoofers[name], **kwargs)

    @classmethod
    def create_aligner(
        cls,
        target: Union[str, BaseAligner],
        **kwargs: Any,
    ) -> BaseAligner:
        if isinstance(target, BaseAligner):
            return target
        name = target.lower()
        if name not in cls._aligners:
            raise KeyError(f"Aligner '{name}' not found. Available: {list(cls._aligners.keys())}")
        return cls._instantiate(cls._aligners[name], **kwargs)

    @classmethod
    def create_embedder(
        cls,
        target: Union[str, BaseEmbedder],
        **kwargs: Any,
    ) -> BaseEmbedder:
        if isinstance(target, BaseEmbedder):
            return target
        name = target.lower()
        if name not in cls._embedders:
            raise KeyError(f"Embedder '{name}' not found. Available: {list(cls._embedders.keys())}")
        return cls._instantiate(cls._embedders[name], **kwargs)

    @classmethod
    def create_storage(
        cls,
        target: Union[str, BaseStorage],
        **kwargs: Any,
    ) -> BaseStorage:
        if isinstance(target, BaseStorage):
            return target
        name = target.lower()
        if name not in cls._storages:
            raise KeyError(f"Storage '{name}' not found. Available: {list(cls._storages.keys())}")
        return cls._instantiate(cls._storages[name], **kwargs)

    @classmethod
    def list_available(cls) -> Dict[str, list]:
        """Returns all registered backend identifiers."""
        return {
            "detectors": list(cls._detectors.keys()),
            "antispoofers": list(cls._antispoofers.keys()),
            "aligners": list(cls._aligners.keys()),
            "embedders": list(cls._embedders.keys()),
            "storages": list(cls._storages.keys()),
        }
