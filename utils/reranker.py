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

from typing import List, Optional, TYPE_CHECKING
import logging
from operator import attrgetter
from sentence_transformers import CrossEncoder
import numpy as np

if TYPE_CHECKING:
    from types_models import SearchResult

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder based document reranker."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize cross-encoder model.

        Args:
            model_name: HuggingFace cross-encoder model name
        """
        self.model_name = model_name
        self._model: Optional[CrossEncoder] = None
        logger.info(f"Reranker initialized with model: {model_name}")

    def _load_model(self) -> CrossEncoder:
        """Lazy-load cross-encoder model to save memory.

        Returns:
            Loaded CrossEncoder instance
        """
        if self._model is None:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder model loaded successfully")
        return self._model

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
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
            pairs = [(query, result.text) for result in results]

            # Score all pairs
            logger.debug(f"Scoring {len(pairs)} query-document pairs")
            scores = model.predict(pairs)

            # Convert scores to numpy array for easier manipulation
            scores_array = np.asarray(scores)

            # Attach scores to results
            reranked_results = []
            for i, result in enumerate(results):
                # Create copy to avoid modifying original
                result_copy = result.model_copy()
                result_copy.score = float(scores_array[i])  # Update score to rerank score
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

    def __del__(self):
        """Cleanup on deletion."""
        if self._model is not None:
            del self._model
            logger.debug("Cross-encoder model unloaded")
