from __future__ import annotations

from typing import Any, Iterator, Protocol, TypedDict, TYPE_CHECKING, cast

from pydantic import BaseModel, ConfigDict, Field

from classes.ProcessingConfig import ProcessingConfig
from types_models import MetadataDict

if TYPE_CHECKING:
    import faiss
    from classes.OCRProcessor import OCRProcessor  # pragma: no cover
    from ingestion.faiss_manager import FaissManager
else:  # pragma: no cover
    OCRProcessor = Any


class PDFRect(Protocol):
    @property
    def x0(self) -> float: ...
    @property
    def y0(self) -> float: ...
    @property
    def x1(self) -> float: ...
    @property
    def y1(self) -> float: ...
    @property
    def width(self) -> float: ...
    @property
    def height(self) -> float: ...


class PDFPixmap(Protocol):
    def getPNGdata(self) -> bytes: ...

    def tobytes(self, output: str = ...) -> bytes: ...


class PILImage(Protocol):
    """Protocol for PIL Image objects.

    Note: Intentionally loose to match both PIL Image types (Image, ImageFile, etc.)
    """

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    def convert(self, mode: str | None = ..., matrix: Any = ..., dither: Any = ..., palette: Any = ..., colors: int = ...) -> Any: ...

    def resize(self, size: tuple[int, int], resample: Any = ...) -> Any: ...

    def close(self) -> None: ...

    def thumbnail(self, size: tuple[int, int], resample: Any = ...) -> None: ...


class PDFPage(Protocol):
    """Protocol for PyMuPDF (fitz) Page objects.

    Matches the actual fitz.Page API without extra flexibility.
    """

    @property
    def rect(self) -> PDFRect: ...

    def get_text(self, option: str = "text") -> str: ...

    def get_pixmap(
        self,
        *,
        matrix: Any = ...,
        dpi: int | None = None,
        colorspace: Any = ...,
        clip: Any = ...,
        alpha: bool = False,
        annots: bool = True,
    ) -> PDFPixmap: ...

    def get_images(self, full: bool = False) -> list[tuple[int, ...]]: ...


class PDFDocument(Protocol):
    page_count: int

    def __iter__(self) -> Iterator[PDFPage]: ...

    def extract_image(self, xref: int) -> dict[str, Any] | None: ...


class OCRProcessorProtocol(Protocol):
    """Protocol for OCR processors to enable testing with mocks."""

    def extract_from_image(self, path: str) -> str: ...

    def ocr_embedded_images(self, doc: PDFDocument, page: PDFPage) -> str: ...

    def ocr_pil_image(self, pil_img: PILImage) -> str: ...

    def ocr_page_tiled(
        self,
        page: PDFPage,
        dpi: int,
        tile_px: int | None = None,
        overlap: int | None = None,
    ) -> str: ...


class EmbeddingModel(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> Any: ...


class ProcessingStats(TypedDict):
    index: faiss.Index | None
    file_count: int
    chunk_total: int
    skipped_already_processed: int
    skipped_problems: int
    skip_reasons: dict[str, int]


class ProcessingResult(TypedDict):
    index: faiss.Index | None
    chunk_count: int


_UNSET: Any = object()


class IngestionContext(BaseModel):
    """Validated container for shared ingestion state."""

    config: ProcessingConfig
    ocr: OCRProcessor
    embedder: EmbeddingModel
    faiss_manager: "FaissManager"
    index: faiss.Index | None = None
    metadata: list[MetadataDict] = Field(default_factory=lambda: cast(list[MetadataDict], []))
    processed_hashes: set[str] = Field(default_factory=set)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def with_index(
        self,
        *,
        index: faiss.Index | None | object = _UNSET,
        metadata: list[MetadataDict] | None | object = _UNSET,
        processed_hashes: set[str] | None | object = _UNSET,
    ) -> "IngestionContext":
        """Return a copy with the supplied state updates."""
        update: dict[str, Any] = {}
        if index is not _UNSET:
            update["index"] = index
        if metadata is not _UNSET:
            update["metadata"] = metadata
        if processed_hashes is not _UNSET:
            update["processed_hashes"] = processed_hashes
        return cast("IngestionContext", self.model_copy(update=update))


__all__ = [
    "PDFRect",
    "PDFPixmap",
    "PILImage",
    "PDFPage",
    "PDFDocument",
    "OCRProcessorProtocol",
    "EmbeddingModel",
    "ProcessingStats",
    "ProcessingResult",
    "IngestionContext",
]
