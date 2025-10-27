"""Type stubs for PyMuPDF (fitz)."""

from typing import Any, Iterator

class Matrix:
    def __init__(
        self,
        a: float = 1.0,
        b: float = 0.0,
        c: float = 0.0,
        d: float = 1.0,
        e: float = 0.0,
        f: float = 0.0,
    ) -> None: ...

class Rect:
    x0: float
    y0: float
    x1: float
    y1: float
    @property
    def width(self) -> float: ...
    @property
    def height(self) -> float: ...
    def __init__(self, x0: float, y0: float, x1: float, y1: float) -> None: ...

class Colorspace:
    pass

csGRAY: Colorspace
csRGB: Colorspace

class Pixmap:
    samples: bytes
    def tobytes(self, format: str = "png") -> bytes: ...

class Page:
    rect: Rect
    def get_pixmap(
        self,
        *,
        matrix: Matrix = ...,
        dpi: int | None = None,
        colorspace: Colorspace | None = None,
        clip: Rect | None = None,
        alpha: bool = False,
        annots: bool = True,
    ) -> Pixmap: ...
    def get_text(self, option: str = "text") -> str: ...
    def get_images(self, full: bool = False) -> list[tuple[int, ...]]: ...

class Document:
    page_count: int
    def __enter__(self) -> Document: ...
    def __exit__(self, *args: Any) -> None: ...
    def __iter__(self) -> Iterator[Page]: ...
    def __getitem__(self, index: int) -> Page: ...
    def __len__(self) -> int: ...
    def load_page(self, page_num: int) -> Page: ...
    def close(self) -> None: ...
    def extract_image(self, xref: int) -> dict[str, Any] | None: ...

def open(filename: str | bytes, filetype: str | None = None) -> Document: ...

Identity: Matrix
