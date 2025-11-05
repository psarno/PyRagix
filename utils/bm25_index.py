"""BM25 keyword-lookup helpers shared with the .NET hybrid retrieval flow.

Both runtimes depend on identical tokenisation (lowercase + whitespace) and on
`rank-bm25`'s scoring semantics so that FAISS/BM25 fusion yields comparable
weights regardless of language implementation.  Any changes to the persistence
format or tokenisation rules must be mirrored in `pyragix-net`.
"""

from typing import Protocol
import pickle
import logging
from pathlib import Path
from collections.abc import Sequence

# BM25 import (lazy to avoid startup cost if not enabled)
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = (
        None  # Deferred failure: only raised when callers actually build the index.
    )

logger = logging.getLogger(__name__)

# Type aliases for BM25 results
Bm25Result = tuple[int, float]  # (metadata_idx, bm25_score)


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

        super().__init__()

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
        results = [(idx, float(scores[idx])) for idx in top_indices if scores[idx] > 0]

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
        with open(file_path, "wb") as f:
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

        with open(file_path, "rb") as f:
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
        (idx, (score - min_score) / denominator) for idx, score in results
    ]

    return normalized
