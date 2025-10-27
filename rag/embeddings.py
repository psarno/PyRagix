"""Embedding utilities shared across the RAG pipeline."""

from contextlib import contextmanager
from typing import Protocol, Sequence, cast, Generator

import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer

FloatArray = npt.NDArray[np.float32]


class SentenceEncoder(Protocol):
    """Structural type for callable sentence encoding functions.

    Types a callable that converts text sentences into float32 embedding vectors.
    This Protocol describes the signature of SentenceTransformer.encode() as a
    callable interface (not the class itself).

    Design rationale: Protocols are the idiomatic way to type callable objects in
    Python. This allows us to:
    - Extract and type the encode method independently of the SentenceTransformer class
    - Use get_sentence_encoder() helper to safely cast the method
    - Enable type checkers to validate embedding operations
    - Document the exact expected signature (keyword-only args, return type)

    Usage:
        from ingestion.environment import initialize_embedder
        from rag.embeddings import get_sentence_encoder, SentenceEncoder

        embedder = initialize_embedder()
        encode: SentenceEncoder = get_sentence_encoder(embedder)
        embeddings = encode(["hello", "world"], convert_to_numpy=True, normalize_embeddings=True)
    """
    def __call__(
        self,
        sentences: Sequence[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> FloatArray: ...


@contextmanager
def memory_cleanup() -> Generator[None, None, None]:
    """Ensure large intermediate tensors are promptly released."""
    try:
        yield
    finally:
        import gc

        _ = gc.collect()


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Normalize embeddings row-wise using the L2 norm."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def get_sentence_encoder(embedder: SentenceTransformer) -> SentenceEncoder:
    """Typed accessor for SentenceTransformer.encode to help static checkers."""
    encode_callable = getattr(embedder, "encode")
    return cast(SentenceEncoder, encode_callable)


__all__ = [
    "FloatArray",
    "SentenceEncoder",
    "get_sentence_encoder",
    "l2_normalize",
    "memory_cleanup",
]
