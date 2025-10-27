from __future__ import annotations

from typing import (
    Any,
    Iterator,
    Protocol,
    TypedDict,
    TYPE_CHECKING,
    cast,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, Field

from classes.ProcessingConfig import ProcessingConfig
from types_models import MetadataDict

if TYPE_CHECKING:
    import faiss
    from ingestion.faiss_manager import FaissManager
else:
    OCRProcessor = Any
    FaissManager = Any

    class _FaissStub:
        Index = Any

    faiss = _FaissStub()


class PDFRect(Protocol):
    """Structural type for rectangular regions in PDF pages (PyMuPDF fitz.Rect).

    Used via structural typing to type PyMuPDF's C++ fitz.Rect objects without requiring
    inheritance. This Protocol is used by OCRProcessor to handle page geometry calculations
    and region-based operations.

    Design rationale: Protocol chosen over ABC because fitz.Rect is a compiled C++ object
    that cannot inherit from Python classes. Structural typing lets us type it without
    modification.
    """

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
    """Structural type for rasterized PDF page content (PyMuPDF fitz.Pixmap).

    Represents a rendered pixel buffer from a PDF page. Used by OCRProcessor to convert
    PDF pages to images for optical character recognition.

    Design rationale: Protocol for PyMuPDF's C++ fitz.Pixmap binding. Structural typing
    allows us to work with rendered content without inheritance or modification of the
    underlying C++ library.
    """

    def getPNGdata(self) -> bytes: ...

    def tobytes(self, output: str = ...) -> bytes: ...


class PILImage(Protocol):
    """Structural type for PIL (Pillow) Image objects.

    Represents image data from PIL/Pillow library. Used by OCRProcessor to convert
    PDFs to PIL images for processing and perform image transformations.

    Note: Intentionally structural to match both PIL Image types (Image.Image,
    ImageFile.ImageFile, etc.) without requiring inheritance. This Protocol is
    loose on optional parameters (using Any) because PIL's actual signatures are
    flexible with many optional transformation parameters.

    Design rationale: Protocol chosen because PIL types are from an external library
    we don't control. Structural typing lets us accept any PIL-like image object
    without importing PIL at the type level.
    """

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    def convert(
        self,
        mode: str | None = ...,
        matrix: Any = ...,
        dither: Any = ...,
        palette: Any = ...,
        colors: int = ...,
    ) -> Any: ...

    def resize(self, size: tuple[int, int], resample: Any = ...) -> Any: ...

    def close(self) -> None: ...

    def thumbnail(self, size: tuple[int, int], resample: Any = ...) -> None: ...


class PDFPage(Protocol):
    """Structural type for PDF page objects (PyMuPDF fitz.Page).

    Represents a single page in a PDF document with methods to extract text,
    render pixels, and retrieve embedded images. Widely used throughout the codebase:
    - OCRProcessor: Text extraction and image-based OCR
    - text_processing: Text and image extraction pipeline
    - file_scanner: Semantic chunking

    Design rationale: Protocol for PyMuPDF's C++ fitz.Page binding. Structural typing
    allows type-safe operations on PDF pages without modifying the underlying library
    or requiring inheritance. This is the idiomatic Python 3.10+ way to type external
    C++ bindings.

    Example usage in OCRProcessor:
        def ocr_page_tiled(self, page: PDFPage, dpi: int) -> str:
            # Type checker knows page has rect, get_pixmap(), get_images() etc.
            rect = page.rect  # PDFRect
            pixmap = page.get_pixmap(dpi=dpi)  # PDFPixmap
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
    """Structural type for PDF document objects (PyMuPDF fitz.Document).

    Represents an entire PDF file with iteration over pages and image extraction.
    Used by OCRProcessor to process all pages and extract embedded images.

    Design rationale: Protocol for PyMuPDF's C++ fitz.Document binding. Structural
    typing enables safe handling of PDF documents without requiring class inheritance.

    Example usage in OCRProcessor:
        def ocr_embedded_images(self, doc: PDFDocument, page: PDFPage) -> str:
            # Type checker knows doc has page_count and extract_image()
            for xref in page.get_images():
                img_data = doc.extract_image(xref)  # dict or None
    """

    page_count: int

    def __iter__(self) -> Iterator[PDFPage]: ...

    def extract_image(self, xref: int) -> dict[str, Any] | None: ...


@runtime_checkable
class OCRProcessorProtocol(Protocol):
    """Structural type for OCR (Optical Character Recognition) processors.

    Defines the interface for extracting text from images and PDF pages. Both the
    production OCRProcessor (uses PaddleOCR) and test mocks implement this protocol.

    Design rationale: Protocol chosen over ABC to enable flexible testing. Test mocks
    (MockOCRProcessor) implement this protocol without requiring inheritance, allowing
    tests to use simple stub implementations while production uses the full PaddleOCR.
    This is duck typing with type safety - "if it can do OCR, it's an OCRProcessor."

    Implementation examples:
    - classes/OCRProcessor.py: Production implementation using PaddleOCR
    - tests/test_file_scanner.py: MockOCRProcessor for testing (returns placeholder text)

    Used throughout the codebase via dependency injection:
    - ingestion/file_scanner.py: DocumentExtractor, Chunker receive OCR processor
    - ingestion/text_processing.py: _extract_from_pdf() uses OCR for scanned PDFs
    """

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


@runtime_checkable
class EmbeddingModel(Protocol):
    """Structural type for sentence embedding models.

    Defines the interface for converting text sentences into numerical vector embeddings.
    The main implementation is SentenceTransformer from the sentence-transformers library.

    Design rationale: Protocol captures the exact method signature of
    SentenceTransformer.encode() without requiring the full class. Structural typing
    allows us to accept any embedding model with this signature, enabling:
    - Easy mocking in tests (MockEmbedder)
    - Type-safe dependency injection
    - Future compatibility with other embedding backends

    Return type (Any) is intentional: models can return numpy arrays, tensors, lists,
    or other numeric types depending on parameters and the underlying implementation.

    Used in:
    - ingestion/file_scanner.py: Chunker receives embedder for semantic chunking
    - ingestion/text_processing.py: chunk_text() requires embeddings
    - ingestion/environment.py: EnvironmentManager initializes embedder
    - Tests (MockEmbedder in test_file_scanner.py)
    """

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
    ocr: OCRProcessorProtocol
    embedder: EmbeddingModel
    faiss_manager: "FaissManager"
    index: faiss.Index | None = None
    metadata: list[MetadataDict] = Field(
        default_factory=lambda: cast(list[MetadataDict], [])
    )
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
