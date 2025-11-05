from __future__ import annotations

from typing import Any, Callable, cast

import pytest

from rag.retrieval import compute_dynamic_hybrid_alpha

approx: Callable[..., Any] = cast(Callable[..., Any], getattr(pytest, "approx"))


def test_compute_dynamic_hybrid_alpha_short_query() -> None:
    alpha = compute_dynamic_hybrid_alpha("tax incentives", 0.7)
    assert 0.1 <= alpha < 0.7


def test_compute_dynamic_hybrid_alpha_long_query() -> None:
    long_query = "how do federal research tax credits interact with state incentives for startups"
    alpha = compute_dynamic_hybrid_alpha(long_query, 0.7)
    assert 0.7 <= alpha <= 1.0


def test_compute_dynamic_hybrid_alpha_handles_empty() -> None:
    assert compute_dynamic_hybrid_alpha("   ", 0.65) == approx(0.65, rel=1e-6)
