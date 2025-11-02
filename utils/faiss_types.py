"""Runtime Protocol helpers for working with FAISS indices.

These protocols let us express optional FAISS attributes in a typed way while
still performing runtime checks (via isinstance) to guard access. Using them
keeps the code type-safe without falling back to bare hasattr calls.
"""

from __future__ import annotations

from typing import Any, Protocol, cast, runtime_checkable

import faiss

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float32]


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

    def reconstruct(self, key: int, recons: FloatArray) -> None:
        ...

    def reconstruct_n(self, start: int, count: int, recons: FloatArray) -> None:
        ...


@runtime_checkable
class SupportsXB(Protocol):
    """Flat indices expose their raw vectors via the xb attribute."""

    xb: FloatArray | None


@runtime_checkable
class SupportsQuantizer(Protocol):
    """IVF indices reference the base quantizer."""

    quantizer: Any


def ensure_nprobe(index: faiss.Index, *, context: str) -> SupportsNProbe:
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


def has_nprobe(index: faiss.Index) -> bool:
    """Check whether the index exposes an nprobe attribute."""
    try:
        _ = getattr(index, "nprobe")  # noqa: B018 - attribute existence check
    except AttributeError:
        return False
    return True


def ensure_reconstruct(
    index: faiss.Index, *, context: str
) -> SupportsReconstruct:
    """Ensure the provided index can reconstruct stored vectors."""
    reconstruct = getattr(index, "reconstruct", None)
    reconstruct_n = getattr(index, "reconstruct_n", None)
    if callable(reconstruct) and callable(reconstruct_n):
        return cast(SupportsReconstruct, index)
    raise TypeError(f"{context} expected reconstruct support")


def ensure_xb(index: faiss.Index, *, context: str) -> SupportsXB:
    """Ensure the provided index exposes xb vectors."""
    xb = getattr(index, "xb", None)
    if xb is None:
        raise TypeError(f"{context} expected xb attribute")
    return cast(SupportsXB, index)
