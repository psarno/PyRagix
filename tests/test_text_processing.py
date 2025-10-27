from typing import Any

from classes.ProcessingConfig import ProcessingConfig
from ingestion.text_processing import safe_dpi_for_page


class _Rect:
    """Mock PDF rectangle for testing."""

    def __init__(self, width: float, height: float) -> None:
        super().__init__()
        self._width = width
        self._height = height

    @property
    def x0(self) -> float:
        return 0.0

    @property
    def y0(self) -> float:
        return 0.0

    @property
    def x1(self) -> float:
        return self._width

    @property
    def y1(self) -> float:
        return self._height

    @property
    def width(self) -> float:
        return self._width

    @property
    def height(self) -> float:
        return self._height


class _Page:
    """Mock PDF page for testing."""

    def __init__(self, rect: _Rect) -> None:
        super().__init__()
        self._rect = rect

    @property
    def rect(self) -> _Rect:
        return self._rect

    def get_text(self, option: str = "text") -> str:
        return ""

    def get_pixmap(
        self,
        *,
        matrix: Any = None,
        dpi: int | None = None,
        colorspace: Any = None,
        clip: Any = None,
        alpha: bool = False,
        annots: bool = True,
    ) -> Any:
        return None

    def get_images(self, full: bool = False) -> list[tuple[int, ...]]:
        return []


def _dimensions(page: _Page, dpi: int) -> tuple[float, float]:
    scale = dpi / 72.0
    return page.rect.width * scale, page.rect.height * scale


def test_safe_dpi_respects_max_pixels() -> None:
    cfg = ProcessingConfig()
    page = _Page(rect=_Rect(width=612, height=792))  # Letter size in points

    dpi = safe_dpi_for_page(
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

    dpi = safe_dpi_for_page(
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

    dpi = safe_dpi_for_page(
        page,
        cfg,
        max_pixels=10_000_000,
        max_side=400,
        base_dpi=200,
    )

    assert dpi == 72
