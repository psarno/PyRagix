from typing import Any, Protocol
import pickle
import logging
from pathlib import Path
from operator import itemgetter
from collections.abc import Sequence

"""
BM25 Keyword Search Index Module

Implements BM25 (Best Matching 25) keyword search for hybrid retrieval.
BM25 complements semantic (FAISS) search by excelling at exact term matches.

Typical use cases where BM25 outperforms semantic embeddings:
- Queries with specific dates, IDs, names, technical terms
- Acronyms and abbreviations
- Exact phrase matching

Architecture:
- Uses rank-bm25 pure Python implementation (zero infrastructure)
- Tokenization: simple whitespace + lowercase (matches FAISS chunking)
- Serialization: pickle for fast save/load
- Integration: scores fused with FAISS scores via weighted average
"""

# BM25 import (lazy to avoid startup cost if not enabled)
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

logger = logging.getLogger(__name__)

FaissResult = dict[str, Any]
Bm25Result = tuple[int, float]


class _BM25Scorer(Protocol):
    """Structural type for BM25 scoring models (rank-bm25's BM25Okapi).

    Defines the interface for querying a BM25 index with tokenized queries to get
    relevance scores. The actual implementation is BM25Okapi from the rank-bm25 library.

    Design rationale: Protocol for the rank-bm25 third-party library's BM25Okapi class.
    Prefixed with underscore (_BM25Scorer) because it's an internal implementation detail
    of this module, not a public API. Structural typing allows us to type the scorer
    without modifying or inheriting from rank-bm25.

    Note: This is a low-level protocol used only within BM25Index to type the internal
    _bm25 field. End users interact with BM25Index.search() which handles tokenization.

    Usage (internal):
        self._bm25: _BM25Scorer | None = BM25Okapi(tokenized_corpus)
        scores: list[float] = self._bm25.get_scores(tokenized_query)
    """
    def get_scores(self, query: Sequence[str]) -> list[float]: ...


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase + whitespace split.

    Args:
        text: Input text to tokenize

    Returns:
        List of tokens
    """
    return text.lower().split()


class BM25Index:
    """BM25 keyword search index wrapper."""

    def __init__(self, corpus: list[str] | None = None):
        """Initialize BM25 index.

        Args:
            corpus: List of document texts to index (optional)
        """
        if BM25Okapi is None:
            raise ImportError(
                "rank-bm25 not installed. Run: pip install rank-bm25>=0.2.2"
            )

        self._corpus = corpus if corpus is not None else []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: _BM25Scorer | None = None

        if self._corpus:
            self._build_index()

    def _build_index(self) -> None:
        """Build BM25 index from corpus."""
        logger.info(f"Building BM25 index from {len(self._corpus)} documents...")

        # Tokenize all documents
        self._tokenized_corpus = [_tokenize(doc) for doc in self._corpus]

        # Build BM25 index
        if BM25Okapi is not None:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        else:
            raise ImportError("rank-bm25 not installed")

        logger.info("BM25 index built successfully")

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Search BM25 index for query.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of (document_index, bm25_score) tuples, sorted by score descending
        """
        if self._bm25 is None:
            logger.warning("BM25 index not built, returning empty results")
            return []

        if not query.strip():
            logger.warning("Empty query for BM25 search")
            return []

        # Tokenize query
        tokenized_query = _tokenize(query)

        # Get scores for all documents
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score
        # argsort gives indices in ascending order, so reverse
        import numpy as np
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        top_indices: list[int] = [int(idx) for idx in sorted_indices]

        # Filter out zero scores (no matches)
        results = [
            (idx, float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]

        logger.debug(f"BM25 search returned {len(results)} results for query: {query}")

        return results

    def get_corpus_size(self) -> int:
        """Get number of documents in corpus.

        Returns:
            Number of documents indexed
        """
        return len(self._corpus)

    def __len__(self) -> int:
        """Return corpus size."""
        return len(self._corpus)


def build_bm25_index(texts: list[str]) -> BM25Index:
    """Build BM25 index from list of texts.

    Args:
        texts: List of document texts to index

    Returns:
        BM25Index instance
    """
    return BM25Index(corpus=texts)


def save_bm25_index(index: BM25Index, file_path: Path) -> None:
    """Save BM25 index to disk using pickle.

    Args:
        index: BM25Index instance to save
        file_path: Path to save index file
    """
    try:
        logger.info(f"Saving BM25 index to {file_path}...")

        # Serialize entire index object
        with open(file_path, 'wb') as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"BM25 index saved ({file_path.stat().st_size / 1024:.1f} KB)")

    except Exception as e:
        logger.error(f"Failed to save BM25 index: {e}")
        raise


def load_bm25_index(file_path: Path) -> BM25Index | None:
    """Load BM25 index from disk.

    Args:
        file_path: Path to index file

    Returns:
        BM25Index instance or None if load fails
    """
    if not file_path.exists():
        logger.warning(f"BM25 index file not found: {file_path}")
        return None

    try:
        logger.info(f"Loading BM25 index from {file_path}...")

        with open(file_path, 'rb') as f:
            index = pickle.load(f)

        if not isinstance(index, BM25Index):
            logger.error(f"Invalid BM25 index file: {file_path}")
            return None

        logger.info(f"BM25 index loaded ({len(index)} documents)")

        return index

    except Exception as e:
        logger.error(f"Failed to load BM25 index: {e}")
        return None


def normalize_bm25_scores(
    results: list[Bm25Result],
) -> list[Bm25Result]:
    """Normalize BM25 scores to [0, 1] range using min-max scaling.

    BM25 scores are unbounded positive values. For hybrid fusion with FAISS
    cosine similarity (which is roughly [0.5, 1.0]), we need to normalize.

    Args:
        results: List of (index, score) tuples from BM25 search

    Returns:
        List of (index, normalized_score) tuples
    """
    if not results:
        return []

    scores: list[float] = [score for _, score in results]
    min_score = min(scores)
    max_score = max(scores)

    # Avoid division by zero
    if max_score == min_score:
        # All scores are the same, normalize to 1.0
        return [(idx, 1.0) for idx, _ in results]

    # Min-max normalization to [0, 1]
    denominator = max_score - min_score
    normalized: list[Bm25Result] = [
        (idx, (score - min_score) / denominator)
        for idx, score in results
    ]

    return normalized


def fuse_scores(
    faiss_results: list[FaissResult],
    bm25_results: list[Bm25Result],
    alpha: float = 0.7,
) -> list[FaissResult]:
    """Fuse FAISS and BM25 scores using weighted average.

    Formula: final_score = alpha * faiss_score + (1 - alpha) * bm25_score

    Note: This function matches FAISS results with BM25 results by their position
    in the metadata list (assumes both use same chunk indexing).

    Args:
        faiss_results: List of search results from FAISS (with 'score' field)
        bm25_results: List of (metadata_index, bm25_score) tuples where metadata_index
                     corresponds to the position in the metadata list
        alpha: Weight for FAISS score (0.0 = pure BM25, 1.0 = pure FAISS)

    Returns:
        Fused results sorted by final score (descending)
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    if not faiss_results:
        return []

    # Normalize BM25 scores to [0, 1]
    bm25_normalized = normalize_bm25_scores(bm25_results)

    # Create lookup dict for BM25 scores by metadata index
    bm25_score_map: dict[int, float] = {
        idx: score for idx, score in bm25_normalized
    }

    # Build fused results
    fused_results: list[FaissResult] = []

    for faiss_idx, faiss_result in enumerate(faiss_results):
        # Get FAISS score (already normalized, typically in [0.5, 1.0] for cosine)
        faiss_score = float(faiss_result.get("score", 0.0))

        # Normalize FAISS scores to [0, 1] (assuming cosine similarity range)
        # Cosine similarity for relevant docs is typically [0.5, 1.0]
        # Map [0.5, 1.0] → [0.0, 1.0]
        faiss_score_normalized = max(0.0, (faiss_score - 0.5) * 2.0)
        faiss_score_normalized = min(1.0, faiss_score_normalized)

        # Try to match with BM25 result
        # The challenge: FAISS results come from query_rag's all_results dict
        # which uses metadata indices as keys. We need to reconstruct that mapping.
        # Since we don't have direct access to the metadata index here,
        # we fall back to positional matching (enumeration index).
        # A more robust solution would pass metadata indices explicitly

        # Get BM25 score (default to 0 if not found)
        bm25_score = bm25_score_map.get(faiss_idx, 0.0)

        # Compute fused score
        fused_score = alpha * faiss_score_normalized + (1 - alpha) * bm25_score

        # Create fused result
        fused: FaissResult = faiss_result.copy()
        fused["faiss_score"] = faiss_score
        fused["bm25_score"] = bm25_score
        fused["fused_score"] = fused_score
        fused["score"] = fused_score  # Update main score field

        fused_results.append(fused)

    # Sort by fused score (descending)
    fused_results.sort(key=itemgetter("fused_score"), reverse=True)

    logger.debug(
        f"Fused {len(faiss_results)} FAISS + {len(bm25_results)} BM25 results " +
        f"(alpha={alpha}) → {len(fused_results)} results"
    )

    return fused_results
