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
    x0: float
    y0: float
    x1: float
    y1: float
    width: float
    height: float


class PDFPixmap(Protocol):
    def getPNGdata(self) -> bytes: ...


class PDFPage(Protocol):
    rect: PDFRect

    def get_text(self, output: str = ...) -> str: ...

    def get_pixmap(
        self,
        matrix: Any = ...,
        colorspace: Any = ...,
        alpha: bool = ...,
        clip: Any = ...,
    ) -> PDFPixmap: ...

    def get_images(self, full: bool = ...) -> list[tuple[int, ...]]: ...


class PDFDocument(Protocol):
    page_count: int

    def __iter__(self) -> Iterator[PDFPage]: ...

    def extract_image(self, xref: int) -> dict[str, Any] | None: ...

    def widgets(self) -> list[Any]: ...


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
    metadata: list[MetadataDict] = Field(default_factory=list)
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
    "PDFPage",
    "PDFDocument",
    "EmbeddingModel",
    "ProcessingStats",
    "ProcessingResult",
    "IngestionContext",
]
