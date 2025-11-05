"""Core building blocks for the RAG query pipeline.

This package re-exports a handful of convenience helpers, but importing them
eagerly pulls in heavy dependencies like PyTorch and FAISS. To keep startup
times snappy (for example when only loading configuration), we lazily import
the underlying modules on first access.
"""

from __future__ import annotations

import importlib
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .configuration import DEFAULT_CONFIG, validate_config
    from .embeddings import (
        FloatArray,
        get_sentence_encoder,
        l2_normalize,
        memory_cleanup,
    )
    from .loader import load_rag_system
    from .llm import generate_answer_with_ollama
    from .retrieval import compute_dynamic_hybrid_alpha, query_rag

__all__ = [
    "DEFAULT_CONFIG",
    "FloatArray",
    "compute_dynamic_hybrid_alpha",
    "generate_answer_with_ollama",
    "get_sentence_encoder",
    "l2_normalize",
    "load_rag_system",
    "memory_cleanup",
    "query_rag",
    "validate_config",
]

_IMPORT_MAP = {
    "DEFAULT_CONFIG": ("rag.configuration", "DEFAULT_CONFIG"),
    "validate_config": ("rag.configuration", "validate_config"),
    "load_rag_system": ("rag.loader", "load_rag_system"),
    "query_rag": ("rag.retrieval", "query_rag"),
    "compute_dynamic_hybrid_alpha": ("rag.retrieval", "compute_dynamic_hybrid_alpha"),
    "generate_answer_with_ollama": ("rag.llm", "generate_answer_with_ollama"),
    "FloatArray": ("rag.embeddings", "FloatArray"),
    "get_sentence_encoder": ("rag.embeddings", "get_sentence_encoder"),
    "l2_normalize": ("rag.embeddings", "l2_normalize"),
    "memory_cleanup": ("rag.embeddings", "memory_cleanup"),
}


def __getattr__(name: str) -> Any:
    """Lazily resolve re-exported names to avoid eager heavy imports."""
    try:
        module_name, attr_name = _IMPORT_MAP[name]
    except KeyError as exc:  # pragma: no cover - defensive for unknown attrs
        raise AttributeError(f"module 'rag' has no attribute '{name}'") from exc

    module = importlib.import_module(module_name)
    attr = getattr(module, attr_name)
    globals()[name] = attr
    return attr


def __dir__() -> list[str]:
    return sorted(__all__)
