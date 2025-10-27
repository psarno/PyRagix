"""Embedding utilities shared across the RAG pipeline."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Protocol, Sequence, cast, Generator

import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer

FloatArray = npt.NDArray[np.float32]


class SentenceEncoder(Protocol):
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

        gc.collect()


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
