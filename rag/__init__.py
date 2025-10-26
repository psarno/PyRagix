"""Core building blocks for the RAG query pipeline."""

from .configuration import DEFAULT_CONFIG, validate_config
from .loader import load_rag_system
from .retrieval import query_rag
from .llm import generate_answer_with_ollama
from .embeddings import (
    FloatArray,
    get_sentence_encoder,
    l2_normalize,
    memory_cleanup,
)

__all__ = [
    "DEFAULT_CONFIG",
    "FloatArray",
    "generate_answer_with_ollama",
    "get_sentence_encoder",
    "l2_normalize",
    "load_rag_system",
    "memory_cleanup",
    "query_rag",
    "validate_config",
]
