"""Configuration helpers for the RAG system."""

from __future__ import annotations

from pathlib import Path

import config
from types_models import RAGConfig


DEFAULT_CONFIG = RAGConfig(
    embed_model=config.EMBED_MODEL,
    index_path=Path("local_faiss.index"),
    db_path=Path("documents.db"),
    ollama_base_url=config.OLLAMA_BASE_URL,
    ollama_model=config.OLLAMA_MODEL,
    default_top_k=config.DEFAULT_TOP_K,
    request_timeout=config.REQUEST_TIMEOUT,
    temperature=config.TEMPERATURE,
    top_p=config.TOP_P,
    max_tokens=config.MAX_TOKENS,
    enable_query_expansion=config.ENABLE_QUERY_EXPANSION,
    query_expansion_count=config.QUERY_EXPANSION_COUNT,
    enable_reranking=config.ENABLE_RERANKING,
    reranker_model=config.RERANKER_MODEL,
    rerank_top_k=config.RERANK_TOP_K,
    enable_hybrid_search=config.ENABLE_HYBRID_SEARCH,
    hybrid_alpha=config.HYBRID_ALPHA,
    bm25_index_path=config.BM25_INDEX_PATH,
)


def validate_config(config_obj: RAGConfig) -> None:
    """Perform lightweight runtime validation on top of Pydantic checks."""
    if config_obj.default_top_k <= 0:
        raise ValueError("default_top_k must be positive")

    if config_obj.request_timeout <= 0:
        raise ValueError("request_timeout must be positive")


__all__ = ["DEFAULT_CONFIG", "validate_config"]
