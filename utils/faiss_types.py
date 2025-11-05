"""Runtime Protocol helpers for working with FAISS indices.

These protocols let us express optional FAISS attributes in a typed way while
still performing runtime checks (via isinstance) to guard access. Using them
keeps the code type-safe without falling back to bare hasattr calls.
"""

from __future__ import annotations

from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
FloatMatrix = npt.NDArray[np.float32]
IntMatrix = npt.NDArray[np.int64]


@runtime_checkable
class FaissIndex(Protocol):
    """Structural protocol describing the FAISS index surface we rely on."""

    d: int
    ntotal: int
    is_trained: bool

    def add(self, x: FloatMatrix) -> None: ...

    def train(self, x: FloatMatrix) -> None: ...

    def search(self, x: FloatMatrix, k: int) -> tuple[FloatMatrix, IntMatrix]: ...

    def reconstruct(self, key: int, recons: FloatArray) -> None: ...

    def reconstruct_n(
        self,
        n0: int,
        ni: int,
        recons: FloatMatrix,
    ) -> None: ...


@runtime_checkable
class SupportsNProbe(Protocol):
    """FAISS IVF-style index exposing the nprobe tuning attribute."""

    nprobe: int


@runtime_checkable
class SupportsDevice(Protocol):
    """Indices created on GPU expose a device attribute."""

    device: int


@runtime_checkable
class SupportsNList(Protocol):
    """IVF indices provide the nlist clustering attribute."""

    nlist: int


@runtime_checkable
class SupportsReconstruct(Protocol):
    """Indices that can reconstruct stored vectors."""

    def reconstruct(self, key: int, recons: FloatArray) -> None: ...

    def reconstruct_n(self, start: int, count: int, recons: FloatMatrix) -> None: ...


@runtime_checkable
class SupportsXB(Protocol):
    """Flat indices expose their raw vectors via the xb attribute."""

    xb: FloatMatrix | None


@runtime_checkable
class SupportsQuantizer(Protocol):
    """IVF indices reference the base quantizer."""

    quantizer: Any


def ensure_nprobe(index: FaissIndex, *, context: str) -> SupportsNProbe:
    """Ensure the provided index exposes nprobe, raising if it does not.

    Args:
        index: FAISS index instance.
        context: Description used in the error message for easier debugging.
    """
    try:
        _ = getattr(index, "nprobe")  # noqa: B018 - intentionally validate attribute existence
    except AttributeError as exc:  # pragma: no cover - defensive guard
        raise TypeError(f"{context} expected an IVF index with nprobe") from exc
    return cast(SupportsNProbe, index)


def has_nprobe(index: FaissIndex) -> bool:
    """Check whether the index exposes an nprobe attribute."""
    try:
        _ = getattr(index, "nprobe")  # noqa: B018 - attribute existence check
    except AttributeError:
        return False
    return True


def ensure_reconstruct(index: FaissIndex, *, context: str) -> SupportsReconstruct:
    """Ensure the provided index can reconstruct stored vectors."""
    reconstruct = getattr(index, "reconstruct", None)
    reconstruct_n = getattr(index, "reconstruct_n", None)
    if callable(reconstruct) and callable(reconstruct_n):
        return cast(SupportsReconstruct, index)
    raise TypeError(f"{context} expected reconstruct support")


def ensure_xb(index: FaissIndex, *, context: str) -> SupportsXB:
    """Ensure the provided index exposes xb vectors."""
    xb = getattr(index, "xb", None)
    if xb is None:
        raise TypeError(f"{context} expected xb attribute")
    return cast(SupportsXB, index)


__all__ = [
    "FaissIndex",
    "FloatArray",
    "FloatMatrix",
    "IntMatrix",
    "SupportsNProbe",
    "SupportsDevice",
    "SupportsNList",
    "SupportsReconstruct",
    "SupportsXB",
    "SupportsQuantizer",
    "ensure_nprobe",
    "ensure_reconstruct",
    "ensure_xb",
    "has_nprobe",
]
