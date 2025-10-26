from __future__ import annotations

from dataclasses import dataclass

from classes.ProcessingConfig import ProcessingConfig
from ingestion.text_processing import _safe_dpi_for_page


@dataclass
class _Rect:
    width: float
    height: float


@dataclass
class _Page:
    rect: _Rect


def _dimensions(page: _Page, dpi: int) -> tuple[float, float]:
    scale = dpi / 72.0
    return page.rect.width * scale, page.rect.height * scale


def test_safe_dpi_respects_max_pixels() -> None:
    cfg = ProcessingConfig()
    page = _Page(rect=_Rect(width=612, height=792))  # Letter size in points

    dpi = _safe_dpi_for_page(
        page,
        cfg,
        max_pixels=1_000_000,
        max_side=10_000,
        base_dpi=300,
    )

    width_px, height_px = _dimensions(page, dpi)

    assert dpi >= 72
    assert dpi < 300
    assert width_px * height_px <= 1_000_000 * 1.01


def test_safe_dpi_caps_long_side() -> None:
    cfg = ProcessingConfig()
    page = _Page(rect=_Rect(width=720, height=720))  # 10"x10" square in points

    dpi = _safe_dpi_for_page(
        page,
        cfg,
        max_pixels=50_000_000,
        max_side=800,
        base_dpi=300,
    )

    width_px, height_px = _dimensions(page, dpi)

    assert 79 <= dpi <= 81  # ~80 DPI after scaling
    assert width_px <= 800.5
    assert height_px <= 800.5


def test_safe_dpi_never_below_72() -> None:
    cfg = ProcessingConfig()
    page = _Page(rect=_Rect(width=2880, height=2880))  # 40"x40" square in points

    dpi = _safe_dpi_for_page(
        page,
        cfg,
        max_pixels=10_000_000,
        max_side=400,
        base_dpi=200,
    )

    assert dpi == 72
