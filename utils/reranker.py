from typing import TYPE_CHECKING, Protocol, Any, cast
import logging
from operator import attrgetter
from collections.abc import Sequence
from sentence_transformers import CrossEncoder

"""
Cross-Encoder Reranking Module

Uses semantic cross-encoder models to rerank retrieved documents by relevance.
This cuts noise and improves precision by scoring query-document pairs directly,
unlike bi-encoders (FAISS) which compare pre-computed embeddings.

Typical workflow:
1. FAISS retrieves top-20 candidates (fast, approximate)
2. Cross-encoder scores all 20 with query (slower, precise)
3. Return top-7 by cross-encoder score for LLM generation
"""

if TYPE_CHECKING:
    from types_models import SearchResult


class _CrossEncoder(Protocol):
    """Structural type for cross-encoder reranking models (sentence-transformers CrossEncoder).

    Defines the interface for scoring query-document pairs to measure semantic relevance.
    The actual implementation is CrossEncoder from the sentence-transformers library.

    Used by Reranker to score retrieved documents against a query, allowing precise
    relevance ranking. Unlike FAISS (bi-encoder, pre-computed embeddings), cross-encoders
    evaluate the pair directly, providing more accurate relevance scores.

    Design rationale: Protocol for the sentence-transformers library's CrossEncoder class.
    Prefixed with underscore (_CrossEncoder) because it's an internal implementation detail
    of the Reranker class, not a public API. Structural typing allows us to type the
    scoring interface without modifying or inheriting from sentence-transformers.

    Note: This is lazy-loaded in Reranker._load_model() to save memory until reranking
    is actually needed.

    Workflow:
    1. FAISS retrieves top-20 candidates (fast, approximate bi-encoder)
    2. CrossEncoder scores all 20 pairs: (query, document)
    3. Results reranked by cross-encoder score → top-7 for LLM

    Usage (internal in Reranker):
        model: _CrossEncoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, doc.text) for doc in results]
        scores: Sequence[float] = model.predict(pairs)
    """
    def predict(
        self,
        sentences: Sequence[tuple[str, str]],
        **kwargs: Any,
    ) -> Sequence[float]: ...

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder based document reranker."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize cross-encoder model.

        Args:
            model_name: HuggingFace cross-encoder model name
        """
        self.model_name = model_name
        self._model: _CrossEncoder | None = None
        logger.info(f"Reranker initialized with model: {model_name}")

    def _load_model(self) -> _CrossEncoder:
        """Lazy-load cross-encoder model to save memory.

        Returns:
            Loaded CrossEncoder instance
        """
        if self._model is None:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            model_instance = CrossEncoder(self.model_name)
            self._model = cast(_CrossEncoder, model_instance)
            logger.info("Cross-encoder model loaded successfully")
        return self._model

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Rerank search results using cross-encoder.

        Args:
            query: User's search query
            results: List of SearchResult Pydantic models with 'text' field
            top_k: Number of top results to return (None = return all)

        Returns:
            Reranked list of results (sorted by cross-encoder score)
        """
        if not results:
            return []

        if not query.strip():
            logger.warning("Empty query for reranking, returning original results")
            return results

        try:
            model = self._load_model()

            # Create query-document pairs for scoring
            pairs: list[tuple[str, str]] = [(query, result.text) for result in results]

            # Score all pairs
            logger.debug(f"Scoring {len(pairs)} query-document pairs")
            scores_raw: Sequence[float] = model.predict(pairs)
            scores = list(scores_raw)

            # Attach scores to results
            reranked_results: list[SearchResult] = []
            for i, result in enumerate(results):
                # Create copy to avoid modifying original
                result_copy = result.model_copy()
                result_copy.score = float(scores[i])  # Update score to rerank score
                reranked_results.append(result_copy)

            # Sort by rerank score (descending)
            reranked_results.sort(key=attrgetter('score'), reverse=True)

            # Return top-k if specified
            if top_k is not None and top_k > 0:
                reranked_results = reranked_results[:top_k]

            logger.info(
                f"Reranked {len(results)} results → returning top {len(reranked_results)}"
            )

            return reranked_results

        except Exception as e:
            logger.error(f"Reranking failed: {e}, returning original results")
            return results

    def __del__(self) -> None:
        """Cleanup on deletion."""
        if self._model is not None:
            del self._model
            logger.debug("Cross-encoder model unloaded")
