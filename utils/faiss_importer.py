"""Centralised FAISS import that suppresses SWIG DeprecationWarnings on CPython 3.13+."""

from __future__ import annotations

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"builtin type SwigPy.* has no __module__ attribute",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"builtin type swigvarlink has no __module__ attribute",
        category=DeprecationWarning,
    )
    import faiss as _faiss

faiss = _faiss

__all__ = ["faiss"]
