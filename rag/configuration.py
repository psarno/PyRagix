"""Configuration helpers for the RAG system."""

from __future__ import annotations
from pathlib import Path

import config
from types_models import RAGConfig


# Seed runtime defaults from the shared config module so CLI scripts can hydrate config quickly.
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


def _looks_like_local_path(value: str) -> bool:
    """Heuristic to determine whether a string refers to a local filesystem path."""
    candidate = Path(value)
    if candidate.is_absolute():
        return True

    # Windows drive-relative forms (e.g., "C:\\..." or "C:/...")
    if (
        len(value) > 2
        and value[1] == ":"
        and value[0].isalpha()
        and value[2] in ("\\", "/")
    ):
        return True

    if value.startswith(("~", "./", "../", ".\\", "..\\")):
        return True

    if "\\" in value:
        return True

    # Common model file suffixes.
    if candidate.suffix.lower() in {".bin", ".pt", ".onnx", ".gguf", ".json"}:
        return True

    return False


def validate_config(config_obj: RAGConfig) -> None:
    """Perform runtime validation on top of Pydantic checks."""
    if config_obj.default_top_k <= 0:
        raise ValueError("default_top_k must be positive")

    if config_obj.request_timeout <= 0:
        raise ValueError("request_timeout must be positive")

    if _looks_like_local_path(config_obj.embed_model):
        # When an explicit model path is provided ensure it exists before SentenceTransformer loads it.
        embed_path = Path(config_obj.embed_model).expanduser().resolve()
        if not embed_path.exists():
            raise FileNotFoundError(
                f"Embedding model file or directory not found: {embed_path}. Update EMBED_MODEL in settings.toml."
            )

    if not config_obj.index_path.exists():
        # Query CLI cannot recover if the vector index is missing, so escalate early.
        raise FileNotFoundError(
            f"FAISS index not found at {config_obj.index_path}. Run the ingestion pipeline before querying."
        )

    if not config_obj.db_path.exists():
        # The metadata database stores chunk text and must stay in lockstep with FAISS indices.
        raise FileNotFoundError(
            f"Metadata database not found at {config_obj.db_path}. Run the ingestion pipeline before querying."
        )

    if config_obj.enable_hybrid_search:
        # Hybrid mode depends on the BM25 pickle built during ingestion; missing file means stale config.
        bm25_path = Path(config_obj.bm25_index_path).expanduser()
        if not bm25_path.exists():
            raise FileNotFoundError(
                f"Hybrid search is enabled but BM25 index '{bm25_path}' is missing. Re-run ingestion with hybrid search enabled to regenerate the index."
            )


__all__ = ["DEFAULT_CONFIG", "validate_config"]
