"""
Type definitions and Pydantic models for PyRagix.

This module provides strong, validated type definitions for all data structures
used throughout the RAG system.
"""

from pathlib import Path
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


class MetadataDict(BaseModel):
    """Metadata structure for document chunks with validation."""

    source: str = Field(description="Source file path")
    chunk_index: int = Field(ge=0, description="Chunk index within document")
    text: str = Field(description="Text content of the chunk")

    model_config = ConfigDict(frozen=True, validate_assignment=True)


class RAGConfig(BaseModel):
    """Configuration for RAG system with full validation."""

    embed_model: str = Field(
        description="Embedding model name from sentence-transformers"
    )
    index_path: Path = Field(description="Path to FAISS index file")
    db_path: Path = Field(description="Path to SQLite database file")
    ollama_base_url: str = Field(description="Base URL for Ollama API")
    ollama_model: str = Field(description="Ollama model name to use")
    default_top_k: int = Field(
        ge=1, description="Default number of results to retrieve"
    )
    request_timeout: int = Field(ge=1, description="Request timeout in seconds")
    temperature: float = Field(ge=0.0, le=2.0, description="LLM temperature")
    top_p: float = Field(ge=0.0, le=1.0, description="LLM top-p sampling")
    max_tokens: int = Field(ge=1, description="Maximum tokens to generate")

    # Phase 1 (v2): Query expansion and reranking
    enable_query_expansion: bool = Field(
        default=False, description="Enable multi-query expansion"
    )
    query_expansion_count: int = Field(
        default=3, ge=1, description="Number of query variants to generate"
    )
    enable_reranking: bool = Field(
        default=False, description="Enable cross-encoder reranking"
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model name",
    )
    rerank_top_k: int = Field(
        default=20,
        ge=1,
        description="Number of candidates to retrieve before reranking",
    )

    # Phase 2 (v2): Hybrid search
    enable_hybrid_search: bool = Field(
        default=False, description="Enable hybrid FAISS + BM25 search"
    )
    hybrid_alpha: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for FAISS scores (0.7 = 70% semantic + 30% keyword)",
    )
    bm25_index_path: str = Field(
        default="bm25_index.pkl", description="Path to BM25 index file"
    )

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    @field_validator("index_path", "db_path", mode="before")
    @classmethod
    def _convert_to_path(cls, v: Any) -> Path:
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v


class SearchResult(BaseModel):
    """Single search result from vector/hybrid search."""

    source: str = Field(description="Source file path or identifier")
    chunk_idx: int = Field(ge=0, description="Chunk index within the source")
    score: float = Field(description="Relevance score (higher is better)")
    text: str = Field(description="Retrieved text content")
    metadata_idx: int = Field(
        ge=0, description="Index in metadata list (for BM25 matching)"
    )

    # Optional fields for hybrid search
    faiss_score: float | None = Field(
        default=None, description="FAISS semantic similarity score"
    )
    bm25_score: float | None = Field(
        default=None, description="BM25 keyword relevance score"
    )
    fused_score: float | None = Field(
        default=None, description="Final fused score from hybrid search"
    )

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=False,  # Allow dynamic score assignment
    )


class DocumentChunk(BaseModel):
    """Document chunk stored in database."""

    source_file: str = Field(description="Source file path")
    chunk_index: int = Field(ge=0, description="Chunk index within document")
    text_content: str = Field(description="Chunk text content")
    embedding_vector: list[float] | None = Field(
        default=None, description="Embedding vector"
    )

    model_config = ConfigDict(
        frozen=False,
        arbitrary_types_allowed=True,
    )


class QueryExpansionResult(BaseModel):
    """Result from query expansion."""

    original_query: str = Field(description="Original user query")
    expanded_queries: list[str] = Field(
        description="List of expanded/paraphrased queries"
    )
    expansion_method: str = Field(
        default="ollama", description="Method used for expansion"
    )

    model_config = ConfigDict(frozen=True)


class RerankingResult(BaseModel):
    """Result from reranking operation."""

    results: list[SearchResult] = Field(description="Reranked search results")
    original_count: int = Field(ge=0, description="Number of results before reranking")
    reranked_count: int = Field(ge=0, description="Number of results after reranking")
    reranker_model: str = Field(description="Reranker model used")

    model_config = ConfigDict(frozen=True)


# Legacy support: provide dict-compatible types for existing code
def rag_config_to_dict(config: RAGConfig) -> dict[str, Any]:
    """Convert RAGConfig to dict for backward compatibility."""
    return config.model_dump()


def search_result_to_dict(result: SearchResult) -> dict[str, Any]:
    """Convert SearchResult to dict for backward compatibility."""
    return result.model_dump(exclude_none=True)
