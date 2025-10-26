from __future__ import annotations

# ======================================
# Ingestion script
# ======================================

# Core dependencies
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup

# Heavy imports with targeted logging suppression
import torch
from sentence_transformers import SentenceTransformer

from paddleocr import PaddleOCR
import fitz  # PyMuPDF
from classes.ProcessingConfig import ProcessingConfig
from classes.OCRProcessor import OCRProcessor

# ===============================
# Standard Library
# ===============================
import argparse
import logging
import os
import sys
import sqlite_utils
import traceback
import math
import gc
import hashlib
from io import BytesIO
from pathlib import Path
from typing import (
    Any,
    Iterator,
    Protocol,
    Union,
    cast,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    import faiss
from typing_extensions import TypedDict
from contextlib import contextmanager
import config


# Protocol definitions for PyMuPDF types
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


# Type definitions
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


# Validate configuration on startup
config.validate_config()

# Set up logging early
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.INGESTION_LOG_FILE, encoding="utf-8"),
    ],
)


# Global instances - will be initialized after configuration is loaded
_config: ProcessingConfig | None = None
_ocr_processor: OCRProcessor | None = None
_gpu_resources: Any = None  # faiss.StandardGpuResources if GPU available
_gpu_functions_available: bool = False  # Track if GPU functions exist in faiss module


def _apply_user_configuration() -> None:
    """Apply configuration from config module to environment variables and torch settings."""
    # Environment variables that need to be set before library imports
    env_vars = [
        "TORCH_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
        "CUDA_VISIBLE_DEVICES",
        "PYTORCH_CUDA_ALLOC_CONF",
        "FAISS_DISABLE_CPU",
        "CUDA_LAUNCH_BLOCKING",
    ]

    for var in env_vars:
        if hasattr(config, var):
            os.environ[var] = str(getattr(config, var))

    # Suppress verbose paddle logging
    os.environ["GLOG_minloglevel"] = "2"

    # Import paddle after setting environment
    import paddle

    # Suppress verbose loggers
    for logger_name in [
        "faiss",
        "sentence_transformers",
        "torch",
        "paddle",
        "paddleocr",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Configure torch with settings from config
    torch.set_num_threads(config.TORCH_NUM_THREADS)

    logger.info(
        f"Torch loaded: {torch.__version__}, CUDA available: {torch.cuda.is_available()}"
    )
    logger.info(f"Paddle compiled with CUDA: {paddle.device.is_compiled_with_cuda()}")


def _initialize_global_instances() -> None:
    """Initialize global _config, _ocr_processor, and _gpu_resources instances."""
    global _config, _ocr_processor, _gpu_resources
    _config = ProcessingConfig()
    _ocr_processor = OCRProcessor(_config)

    # Initialize GPU resources if enabled
    if config.GPU_ENABLED:
        gpu_available, gpu_status = _detect_gpu_faiss()
        logger.info(f"🎮 GPU detection: {gpu_status}")

        if gpu_available:
            _gpu_resources = _create_gpu_resources()
            if _gpu_resources is None:
                logger.warning("⚠️  GPU requested but initialization failed, using CPU")
        else:
            logger.info("💻 GPU FAISS not available, using CPU")
    else:
        logger.info("💻 GPU disabled, using CPU FAISS")


@contextmanager
def _memory_cleanup() -> Iterator[None]:
    """Context manager for automatic memory cleanup after processing."""
    try:
        yield
    finally:
        # Force garbage collection after operations to prevent memory fragmentation
        gc.collect()
        # Keep VRAM stable
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _cleanup_memory() -> None:
    """Force garbage collection and CUDA memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _create_faiss_index(
    dim: int, index_type: str, nlist: int, gpu_res: Any = None
) -> tuple[faiss.Index, str]:
    """Create FAISS index based on configuration.

    Args:
        dim: Embedding dimension
        index_type: 'flat' or 'ivf'
        nlist: Number of clusters for IVF (ignored for flat)
        gpu_res: GPU resources for GPU acceleration (optional)

    Returns:
        tuple: (faiss.Index, actual_type) where actual_type is 'flat' or 'ivf'
    """
    import faiss
    
    # Create CPU index first
    cpu_index = None
    actual_type = "flat"

    if index_type.lower() == "ivf":
        try:
            # Create IVF index with inner product similarity
            quantizer = faiss.IndexFlatIP(dim)
            cpu_index = faiss.IndexIVFFlat(
                quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
            )
            actual_type = "ivf"
            logger.info(f"🔧 Created IVF index: dim={dim}, nlist={nlist}")
        except Exception as e:
            logger.error(f"⚠️  Failed to create IVF index: {e}")
            logger.info("Falling back to flat index...")

    if cpu_index is None:
        # Default to flat index (fallback or requested)
        cpu_index = faiss.IndexFlatIP(dim)
        actual_type = "flat"
        logger.info(f"🔧 Created flat index: dim={dim}")

    # Move to GPU if requested and possible
    if gpu_res is not None and config.GPU_ENABLED:
        gpu_index = _move_index_to_gpu(cpu_index, gpu_res)
        if gpu_index is not None:
            return gpu_index, actual_type
        else:
            logger.info("💻 Continuing with CPU index")

    return cpu_index, actual_type


def _train_ivf_index(index: faiss.Index, training_data: np.ndarray) -> bool:
    """Train IVF index with provided embeddings.

    Args:
        index: IVF index to train
        training_data: Embeddings for training centroids

    Returns:
        bool: True if training succeeded, False if failed
    """
    if not hasattr(index, "is_trained") or index.is_trained:
        logger.debug("Index is already trained or doesn't require training")
        return True

    num_vectors = len(training_data)
    nlist = getattr(index, "nlist", config.NLIST)

    # Check if we have enough vectors for clustering
    min_vectors_needed = nlist * 2  # At least 2 vectors per cluster
    if num_vectors < min_vectors_needed:
        logger.info(
            f"ℹ️  Not enough vectors for IVF training: {num_vectors} < {min_vectors_needed}"
        )
        logger.info("Falling back to flat index for now...")
        return False

    try:
        logger.info(
            f"🎯 Training IVF index with {num_vectors} vectors, {nlist} clusters..."
        )
        with _memory_cleanup():
            index.train(training_data)  # type: ignore
        logger.info("✅ IVF index training completed")
        return True
    except (RuntimeError, ValueError) as e:
        logger.error(f"⚠️  IVF training failed: {e}")
        logger.info("Will retry with accumulated vectors later...")
        return False


def _detect_gpu_faiss() -> tuple[bool, str]:
    """Detect if GPU FAISS functions are available and working.

    Returns:
        tuple: (is_available, status_message)
    """
    import faiss

    global _gpu_functions_available

    # First check if GPU functions exist in the faiss module
    required_attrs = ["StandardGpuResources", "index_cpu_to_gpu", "index_gpu_to_cpu"]
    missing_attrs = [attr for attr in required_attrs if not hasattr(faiss, attr)]

    if missing_attrs:
        _gpu_functions_available = False
        return (
            False,
            f"GPU functions not available in faiss module (missing: {missing_attrs})",
        )

    # Functions exist, now test if GPU actually works
    try:
        gpu_res = getattr(faiss, "StandardGpuResources")()
        test_index = faiss.IndexFlatIP(384)  # Common embedding dimension
        gpu_index = getattr(faiss, "index_cpu_to_gpu")(
            gpu_res, config.GPU_DEVICE, test_index
        )
        del gpu_index, test_index, gpu_res
        _gpu_functions_available = True
        return True, f"GPU {config.GPU_DEVICE} available and working"
    except Exception as e:
        _gpu_functions_available = True  # Functions exist but GPU failed
        return False, f"GPU functions available but GPU failed: {str(e)[:100]}"


def _create_gpu_resources() -> Any:
    """Create GPU resources for FAISS with memory management.

    Returns:
        GPU resources object or None if failed
    """
    if not config.GPU_ENABLED or not _gpu_functions_available:
        return None

    try:
        gpu_res = getattr(faiss, "StandardGpuResources")()

        # Set memory fraction if specified
        if hasattr(gpu_res, "setTempMemoryFraction"):
            gpu_res.setTempMemoryFraction(config.GPU_MEMORY_FRACTION)

        logger.info(
            f"🎮 GPU resources initialized (device {config.GPU_DEVICE}, memory fraction: {config.GPU_MEMORY_FRACTION})"
        )
        return gpu_res
    except Exception as e:
        logger.error(f"⚠️  Failed to create GPU resources: {e}")
        return None


def _move_index_to_gpu(index: faiss.Index, gpu_res: Any) -> faiss.Index | None:
    """Move CPU index to GPU if possible.

    Args:
        index: CPU FAISS index
        gpu_res: GPU resources

    Returns:
        GPU index or None if failed
    """
    if not _gpu_functions_available:
        return None

    try:
        gpu_index = getattr(faiss, "index_cpu_to_gpu")(
            gpu_res, config.GPU_DEVICE, index
        )
        logger.info("🎮 Index moved to GPU")
        return gpu_index
    except Exception as e:
        logger.error(f"⚠️  Failed to move index to GPU: {e}")
        return None


def _needs_retraining(index: faiss.Index, new_vector_count: int) -> bool:
    """Determine if IVF index needs retraining based on new data volume.

    Args:
        index: FAISS index to check
        new_vector_count: Number of new vectors being added

    Returns:
        bool: True if retraining is recommended
    """
    if not hasattr(index, "is_trained") or not hasattr(index, "ntotal"):
        return False

    # Retrain if adding >20% new vectors to existing index
    if index.ntotal > 0:
        growth_ratio = new_vector_count / index.ntotal
        return growth_ratio > 0.2

    return False


def _init_embedder() -> SentenceTransformer:
    assert _config is not None, "_config must be initialized before use"
    return SentenceTransformer(_config.embed_model)


def _clean_text(s: str) -> str:
    # Collapse whitespace; keep newlines sparsely
    return " ".join(s.split())


def _chunk_text(
    text: str,
    size: int | None = None,
    overlap: int | None = None,
    embedder: Any | None = None,
) -> list[str]:
    """Chunk text using either semantic or fixed-size chunking.

    Args:
        text: Text to chunk
        size: Max characters per chunk (uses _config.chunk_size if None)
        overlap: Character overlap between chunks (uses _config.chunk_overlap if None)
        embedder: SentenceTransformer model for semantic chunking (required if semantic chunking enabled)

    Returns:
        List of text chunks
    """
    # Resolve config values if not provided
    if size is None:
        assert _config is not None, "_config must be initialized before use"
        size = _config.chunk_size
    if overlap is None:
        assert _config is not None, "_config must be initialized before use"
        overlap = _config.chunk_overlap

    text = text.strip()
    if not text:
        return []

    # Phase 3: Semantic Chunking (v2)
    if config.ENABLE_SEMANTIC_CHUNKING:
        try:
            # Lazy import to avoid dependency issues if feature is disabled
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            if embedder is None:
                logging.warning(
                    "⚠️ Semantic chunking enabled but no embedder provided, falling back to fixed-size chunking"
                )
            else:
                # Use RecursiveCharacterTextSplitter for sentence-boundary-aware splitting
                # This respects sentence boundaries better than naive character splitting
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config.SEMANTIC_CHUNK_MAX_SIZE,
                    chunk_overlap=config.SEMANTIC_CHUNK_OVERLAP,
                    length_function=len,
                    separators=[
                        "\n\n",  # Paragraph breaks
                        "\n",  # Line breaks
                        ". ",  # Sentence ends
                        "? ",
                        "! ",
                        "; ",
                        ", ",  # Clause breaks
                        " ",  # Word boundaries
                        "",  # Character-level fallback
                    ],
                )
                chunks = text_splitter.split_text(text)
                logging.debug(
                    f"📝 Semantic chunking: {len(chunks)} chunks (max_size={config.SEMANTIC_CHUNK_MAX_SIZE}, overlap={config.SEMANTIC_CHUNK_OVERLAP})"
                )
                return chunks
        except ImportError:
            logging.warning(
                "⚠️ langchain-text-splitters not installed. Falling back to fixed-size chunking. Install with: uv add langchain-text-splitters"
            )
        except Exception as e:
            logging.warning(
                f"⚠️ Semantic chunking failed: {e}. Falling back to fixed-size chunking."
            )

    # Fixed-size chunking (original behavior)
    chunks = []
    i = 0
    step = max(1, size - overlap)
    while i < len(text):
        chunk = text[i : i + size]
        chunks.append(chunk)
        i += step
    return chunks


def _html_to_text(path: str) -> str:
    # Prefer lxml if available; fall back gracefully
    parser = "lxml"
    try:
        import lxml as _lxml  # Check availability but don't use directly

        _ = _lxml  # Suppress unused import warning
    except ImportError:
        parser = "html.parser"
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, parser)
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    return soup.get_text(separator="\n")


def _safe_dpi_for_page(
    page: PDFPage,
    max_pixels: int | None = None,
    max_side: int | None = None,
    base_dpi: int = config.BASE_DPI,
) -> int:
    """Calculate safe DPI for page rendering to avoid memory issues.

    Args:
        page: PyMuPDF page object
        max_pixels: Maximum total pixels allowed (width * height)
        max_side: Maximum pixels for either width or height
        base_dpi: Starting DPI to scale down from

    Returns:
        int: Safe DPI value (minimum 72)
    """
    # Resolve config values if not provided
    if max_pixels is None:
        assert _config is not None, "_config must be initialized before use"
        max_pixels = _config.max_pixels
    if max_side is None:
        assert _config is not None, "_config must be initialized before use"
        max_side = _config.max_side

    # page.rect is in points; 72 points = 1 inch
    rect = page.rect
    if rect.width == 0 or rect.height == 0:
        return 96

    # scale from DPI: pixels = points/72 * dpi
    def px_for(dpi: int) -> tuple[float, float]:
        s = dpi / 72.0
        return rect.width * s, rect.height * s

    # Start from base_dpi and adjust down if needed
    w0, h0 = px_for(base_dpi)
    scale = 1.0
    if max_pixels is not None and w0 * h0 > max_pixels:
        scale *= math.sqrt(max_pixels / (w0 * h0))
    if w0 * scale > max_side:
        scale *= max_side / (w0 * scale)
    if h0 * scale > max_side:
        scale *= max_side / (h0 * scale)
    dpi = max(
        72, int(base_dpi * scale)
    )  # don't go below 72 unless you want more aggressive downscale
    return dpi


def _ocr_pil_image(ocr: PaddleOCR, pil_img: Image.Image) -> str:
    assert _config is not None, "_config must be initialized before use"
    try:
        arr = np.array(
            pil_img.convert("RGB"), dtype=np.uint8
        )  # Paddle expects RGB ndarray
        result = ocr.predict(arr)
        if not result or not result[0]:
            return ""
        return "\n".join([line[1][0] for line in result[0]])
    except (RuntimeError, KeyboardInterrupt, OSError) as e:
        logger.error(f"⚠️  OCR failed on PIL image: {e}")
        return ""


def _ocr_page_tiled(
    ocr: PaddleOCR,
    page: fitz.Page,
    dpi: int,
    tile_px: int | None = None,
    overlap: int | None = None,
) -> str:
    """OCR a page by splitting it into tiles to manage memory usage.

    Args:
        ocr: PaddleOCR instance
        page: PyMuPDF page object
        dpi: DPI for rendering
        tile_px: Size of each tile in pixels
        overlap: Overlap between tiles in pixels

    Returns:
        str: Concatenated OCR text from all tiles
    """
    # Resolve config values if not provided
    if tile_px is None:
        assert _config is not None, "_config must be initialized before use"
        tile_px = _config.tile_size
    if overlap is None:
        assert _config is not None, "_config must be initialized before use"
        overlap = _config.tile_overlap

    rect = page.rect
    s = dpi / 72.0
    full_w = int(rect.width * s)
    full_h = int(rect.height * s)

    texts = []
    # number of tiles in each dimension
    # tile_px is resolved from config above, but add fallback for safety
    if tile_px is None:
        tile_px = 600  # fallback if _config.tile_size is also None
    nx = max(1, math.ceil(full_w / tile_px))
    ny = max(1, math.ceil(full_h / tile_px))

    # tile size in page coordinates (points)
    tile_w_pts = tile_px / s
    tile_h_pts = tile_px / s
    ov_pts = overlap / s

    for iy in range(ny):
        for ix in range(nx):
            x0 = rect.x0 + ix * tile_w_pts - (ov_pts if ix > 0 else 0)
            y0 = rect.y0 + iy * tile_h_pts - (ov_pts if iy > 0 else 0)
            x1 = min(
                rect.x0 + (ix + 1) * tile_w_pts + (ov_pts if ix + 1 < nx else 0),
                rect.x1,
            )
            y1 = min(
                rect.y0 + (iy + 1) * tile_h_pts + (ov_pts if iy + 1 < ny else 0),
                rect.y1,
            )
            clip = fitz.Rect(x0, y0, x1, y1)

            try:

                # Explicitly cast to the fitz.Page type to satisfy Pylance
                typed_page = cast(fitz.Page, page)

                # GRAY, no alpha massively reduces memory (n=1 channel)
                pix = typed_page.get_pixmap(  # type: ignore
                    matrix=fitz.Matrix(s, s),
                    colorspace=fitz.csGRAY,
                    alpha=False,
                    clip=clip,
                )

                # Avoid pix.samples  → use compressed PNG bytes
                png_bytes = pix.tobytes("png")
                im = Image.open(BytesIO(png_bytes))
                txt = _ocr_pil_image(ocr, im)
                if txt.strip():
                    texts.append(txt)
            except MemoryError:
                # If a tile still fails (rare), try halving tile size once
                if tile_px is not None and tile_px > 800:
                    return _ocr_page_tiled(
                        ocr, page, dpi, tile_px=tile_px // 2, overlap=overlap
                    )
                else:
                    continue
            except (OSError, RuntimeError, ValueError):
                # OCR/image processing errors
                continue

    return "\n".join(texts)


def _pdf_page_text_or_ocr(
    page: Any, ocr: OCRProcessor, doc: Any = None
) -> str:
    """Extract text from PDF page using native text first, then OCR fallback.

    Args:
        page: PyMuPDF page object
        ocr: PaddleOCR instance
        doc: PyMuPDF document object (optional, for embedded image extraction)

    Returns:
        str: Extracted text from the page
    """
    assert _config is not None, "_config must be initialized before use"
    # 1) Native text first
    text = page.get_text("text") or ""
    if len(text.strip()) > 20:
        return text

    # 2) Try embedded images (cheaper than full render)
    if doc is not None:
        emb_txt = ocr.ocr_embedded_images(doc, page)
        if emb_txt.strip():
            return emb_txt

    # 3) OCR path with safe DPI, grayscale, tiling
    dpi = _safe_dpi_for_page(
        page,
        max_pixels=_config.max_pixels,
        max_side=_config.max_side,
        base_dpi=config.BASE_DPI,
    )

    rect = page.rect
    s = dpi / 72.0
    w_px = int(rect.width * s)
    h_px = int(rect.height * s)

    if (
        _config.tile_size is not None
        and w_px <= _config.tile_size
        and h_px <= _config.tile_size
    ):
        try:
            pix = page.get_pixmap(
                matrix=fitz.Matrix(s, s), colorspace=fitz.csGRAY, alpha=False
            )
            png_bytes = pix.getPNGdata()
            im = Image.open(BytesIO(png_bytes))
            return ocr.ocr_pil_image(im)
        except MemoryError:
            return ocr.ocr_page_tiled(page, dpi)
        except (OSError, RuntimeError, ValueError):
            # Image processing/OCR errors
            return ""

    # Larger pages → tiled OCR
    return ocr.ocr_page_tiled(page, dpi)


def _extract_from_pdf(path: str, ocr: OCRProcessor) -> str:
    out = []
    with fitz.open(path) as doc:  # type: ignore[attr-defined]
        for p in doc:
            try:
                out.append(_pdf_page_text_or_ocr(p, ocr, doc=doc))
            except (RuntimeError, MemoryError, OSError) as e:
                logger.error(f"⚠️ Error processing PDF page: {type(e).__name__}: {e}")
                logger.debug("Full traceback:", exc_info=True)
                continue
    return "\n".join(out)


def _extract_text(path: str, ocr: OCRProcessor) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _extract_from_pdf(path, ocr)
    elif ext in {".html", ".htm"}:
        return _html_to_text(path)
    else:
        return ocr.extract_from_image(path)


def _calculate_file_hash(path: str) -> str:
    """Calculate SHA256 hash of file contents for duplicate detection.

    Uses optimized chunk size based on file size for better performance.

    Args:
        path: Full file path

    Returns:
        str: Hex digest of file hash, or empty string if file can't be read
    """
    try:
        file_size = os.path.getsize(path)

        # Optimize chunk size based on file size
        if file_size < 1024 * 1024:  # < 1MB
            chunk_size = 4096  # 4KB
        elif file_size < 10 * 1024 * 1024:  # < 10MB
            chunk_size = 64 * 1024  # 64KB
        elif file_size < 100 * 1024 * 1024:  # < 100MB
            chunk_size = 256 * 1024  # 256KB
        else:  # >= 100MB
            chunk_size = 1024 * 1024  # 1MB

        hash_sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            # Read in optimized chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except (OSError, IOError, MemoryError) as e:
        logger.error(f"⚠️ Could not hash {os.path.basename(path)}: {e}")
        return ""


def _should_skip_file(path: str, ext: str, processed: set[str]) -> tuple[bool, str]:
    """Determine if a file should be skipped during processing.

    Args:
        path: Full file path
        ext: File extension (lowercase)
        processed: Set of already processed file hashes

    Returns:
        tuple[bool, str]: (should_skip, reason)
    """
    assert _config is not None, "_config must be initialized before use"
    # Check if filename is in skip list
    filename = os.path.basename(path)
    if filename in _config.skip_files:
        return True, "file in hard-coded skip list."

    # Check against effective extensions (filtered or all supported)
    effective_extensions = _config.get_effective_extensions()
    if ext not in effective_extensions:
        if _config.allowed_extensions is not None:
            # File type filtering is active
            return True, f"file type not in filter: {ext}"
        else:
            # No filtering, so this is truly unsupported
            return True, f"unsupported file type: {ext}"

    # Check if file hash is already processed
    file_hash = _calculate_file_hash(path)
    if file_hash and file_hash in processed:
        return True, "already processed"

    # File size
    file_size_mb = os.path.getsize(path) / 1024 / 1024
    if file_size_mb > _config.max_file_mb:
        return True, f"large file ({file_size_mb:.1f} MB)"

    # PDF-specific checks
    if ext == ".pdf":
        try:
            with fitz.open(path) as doc:  # type: ignore[attr-defined]
                if doc.page_count > _config.max_pdf_pages:
                    return True, f"PDF with {doc.page_count} pages"
                if _config.skip_form_pdfs:
                    try:
                        if getattr(doc, 'widgets', lambda: cast(list[Any], []))():
                            return True, "form-heavy PDF (has interactive fields)"
                    except AttributeError:
                        # Older PyMuPDF versions don't have widgets() method
                        pass
        except (OSError, RuntimeError, ValueError) as e:
            return True, f"cannot open PDF: {e}"

    return False, ""  # do not skip


def _load_existing_index() -> tuple[faiss.Index | None,list[dict[str, Any]]]:
    """Load existing FAISS index and metadata if they exist."""
    import faiss
    
    assert _config is not None, "_config must be initialized before use"
    if (
        _config.index_path
        and _config.index_path.exists()
        and _config.db_path
        and _config.db_path.exists()
    ):
        print("📂 Loading existing index and metadata...")
        index = faiss.read_index(str(_config.index_path))

        # Load metadata from SQLite database
        db: Any = sqlite_utils.Database(str(_config.db_path))
        metadata = []
        if "chunks" in db.table_names():
            chunks_table: Any = db["chunks"]
            for row in chunks_table.rows:  # type: ignore
                metadata.append(
                    {
                        "source": row["source"],
                        "chunk_index": row["chunk_index"],
                        "text": row["text"],
                    }
                )

        # Move to GPU if enabled and available
        if _gpu_resources is not None and config.GPU_ENABLED:
            gpu_index = _move_index_to_gpu(index, _gpu_resources)
            if gpu_index is not None:
                index = gpu_index

        # Set nprobe for IVF indices
        if hasattr(index, "nprobe"):
            index.nprobe = config.NPROBE  # type: ignore
            logger.info(f"🎯 Set IVF nprobe to {config.NPROBE}")

        print(
            f"   Loaded {index.ntotal} existing chunks from {len(set(m['source'] for m in metadata))} files"
        )
        return index, metadata
    else:
        return None, []


def _load_processed_files() -> set[str]:
    """Load the set of already processed file hashes from the log.

    Expected format: "hash|filename"

    Returns:
        set[str]: Set of file hashes that have been processed
    """
    assert _config is not None, "_config must be initialized before use"
    processed_hashes = set()

    if _config.processed_log and _config.processed_log.exists():
        try:
            with open(_config.processed_log, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and "|" in line:
                        try:
                            file_hash, _ = line.split("|", 1)
                            processed_hashes.add(file_hash)
                        except ValueError:
                            # Malformed line, skip
                            continue
        except UnicodeDecodeError:
            print("⚠️  Converting processed_files.txt to UTF-8...")
            # Read with system default encoding and rewrite as UTF-8
            with open(
                _config.processed_log, "r", encoding="cp1252", errors="ignore"
            ) as f:
                lines = [line.strip() for line in f if line.strip()]
            with open(_config.processed_log, "w", encoding="utf-8") as f:
                for line in lines:
                    f.write(f"{line}\n")
            # Retry reading
            return _load_processed_files()

    return processed_hashes


def _detect_stale_documents(
    processed_files: set[str], current_files:list[Path]
) ->list[str]:
    """Detect processed files that no longer exist in the current directory.

    Args:
        processed_files: Set of file hashes from processed_files.txt
        current_files: List of current file paths in directory

    Returns:
        List of file paths that were processed but no longer exist
    """
    if not processed_files:
        return []

    # Build current file hash mapping
    current_hashes = set()
    current_paths = {}

    for file_path in current_files:
        if file_path.is_file():
            try:
                file_hash = _calculate_file_hash(str(file_path))
                current_hashes.add(file_hash)
                current_paths[file_hash] = str(file_path)
            except (OSError, PermissionError):
                continue

    # Find stale entries - hashes in processed but not in current
    stale_hashes = processed_files - current_hashes

    # Try to map back to file paths from processed_files.txt
    stale_paths = []
    if stale_hashes and _config and _config.processed_log:
        try:
            with open(_config.processed_log, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and "|" in line:
                        parts = line.split("|", 1)
                        if len(parts) == 2:
                            file_hash, file_path = parts[0].strip(), parts[1].strip()
                            if file_hash in stale_hashes:
                                stale_paths.append(file_path)
        except (OSError, UnicodeDecodeError):
            pass

    return stale_paths


def _prompt_user_stale_action(stale_files:list[str]) -> str:
    """Prompt user for action when stale documents are detected.

    Returns:
        User choice: 'fresh', 'clean', 'append', or 'quit'
    """
    print("\n⚠️  Stale document references detected!")
    print(f"\nFound {len(stale_files)} processed files that no longer exist:")

    # Show up to 5 files, then "... and X more"
    display_files = stale_files[:5]
    for file_path in display_files:
        print(f"• {file_path}")

    if len(stale_files) > 5:
        print(f"• (+ {len(stale_files) - 5} more...)")

    print("\nOptions:")
    print("[F]resh start - Clear all data and reprocess from scratch")
    print("[C]lean - Remove stale entries and process new/changed files")
    print("[A]ppend - Keep existing data and add new files only")
    print("[Q]uit - Exit without changes")

    while True:
        try:
            choice = input("\nChoose [F/C/A/Q]: ").strip().upper()
            if choice in ["F", "FRESH"]:
                return "fresh"
            elif choice in ["C", "CLEAN"]:
                return "clean"
            elif choice in ["A", "APPEND"]:
                return "append"
            elif choice in ["Q", "QUIT"]:
                return "quit"
            else:
                print("Please enter F, C, A, or Q")
        except (EOFError, KeyboardInterrupt):
            print("\nOperation cancelled.")
            return "quit"


def _clean_stale_entries(stale_files:list[str]) -> None:
    """Remove stale entries from processed files, database, and rebuild FAISS index.

    Args:
        stale_files: List of file paths that no longer exist
    """
    if not stale_files or not _config:
        return

    print(f"🧹 Cleaning {len(stale_files)} stale entries...")

    # Build hash set of stale files
    stale_hashes = set()
    hash_to_path = {}

    if _config.processed_log and _config.processed_log.exists():
        try:
            with open(_config.processed_log, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and "|" in line:
                        parts = line.split("|", 1)
                        if len(parts) == 2:
                            file_hash, file_path = parts[0].strip(), parts[1].strip()
                            hash_to_path[file_hash] = file_path
                            if file_path in stale_files:
                                stale_hashes.add(file_hash)
        except (OSError, UnicodeDecodeError):
            print("   Warning: Could not read processed files log")

    # Clean processed_files.txt
    if stale_hashes and _config.processed_log:
        try:
            # Read all entries
            valid_entries = []
            with open(_config.processed_log, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and "|" in line:
                        parts = line.split("|", 1)
                        if len(parts) == 2:
                            file_hash, file_path = parts[0].strip(), parts[1].strip()
                            if file_hash not in stale_hashes:
                                valid_entries.append(line)

            # Rewrite file with valid entries only
            with open(_config.processed_log, "w", encoding="utf-8") as f:
                for entry in valid_entries:
                    f.write(f"{entry}\n")

            print(
                f"   Cleaned processed files log ({len(valid_entries)} entries remain)"
            )

        except (OSError, UnicodeDecodeError) as e:
            print(f"   Warning: Could not clean processed files log: {e}")

    # Clean database
    if _config.db_path and _config.db_path.exists():
        try:
            import sqlite_utils

            db = sqlite_utils.Database(_config.db_path)

            # Get stale document sources in multiple formats to match database
            stale_sources = []
            for path in stale_files:
                # Try different formats: filename only, full path, relative path
                stale_sources.extend(
                    [
                        Path(path).name,  # just filename
                        path,  # full path as stored
                        (
                            str(Path(path).relative_to(Path.cwd()))
                            if Path(path).is_absolute()
                            else path
                        ),  # relative path
                    ]
                )

            if stale_sources:
                # Delete chunks for stale documents
                placeholders = ",".join(["?" for _ in stale_sources])
                query = f"DELETE FROM chunks WHERE source IN ({placeholders})"
                db.execute(query, stale_sources)

                deleted_count = db.execute("SELECT changes()").fetchone()[0]
                print(f"   Cleaned database ({deleted_count} chunks removed)")

        except Exception as e:
            print(f"   Warning: Could not clean database: {e}")

    # Force FAISS index rebuild
    if _config.index_path and _config.index_path.exists():
        _config.index_path.unlink()
        print("   Removed FAISS index (will be rebuilt)")

    print("✅ Stale entries cleaned successfully")


def _scan_and_process_files(
    root_path: Union[str, Path],
    ocr: OCRProcessor,
    embedder: SentenceTransformer,
    index: faiss.Index | None,
    metadata:list[dict[str, Any]],
    processed: set[str],
    recurse_subdirs: bool = True,
) -> ProcessingStats:
    """Scan directory and process all supported files."""
    assert _config is not None, "_config must be initialized before use"
    file_count = 0
    chunk_total = len(metadata) if metadata else 0
    skipped_already_processed = 0
    skipped_problems = 0
    skip_reasons = {}

    if recurse_subdirs:
        # Recursively walk through all subdirectories
        walker = os.walk(root_path)
    else:
        # Only process files in the root directory
        try:
            root_files = [
                f
                for f in os.listdir(root_path)
                if os.path.isfile(os.path.join(root_path, f))
            ]
            walker = [(str(root_path), [], root_files)]
        except OSError:
            print(f"❌ Cannot access directory: {root_path}")
            return {
                "index": index,
                "file_count": 0,
                "chunk_total": chunk_total,
                "skipped_already_processed": 0,
                "skipped_problems": 0,
                "skip_reasons": {},
            }

    for dirpath, _, filenames in walker:
        for fname in filenames:
            path = os.path.join(dirpath, fname)
            ext = os.path.splitext(fname)[1].lower()

            skip, reason = _should_skip_file(path, ext, processed)
            if skip:
                if reason == "already processed":
                    skipped_already_processed += 1
                    if (
                        file_count % _config.top_print_every == 0
                    ):  # Only log occasionally to reduce noise
                        print(f"✓ Already processed: {fname}")
                else:
                    skipped_problems += 1
                    print(f"💨 Skipping {fname}: {reason}")
                    # Track skip reasons for summary
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                continue

            try:
                print(f"Processing: {path}")

                file_count += 1
                with _memory_cleanup():
                    result = _process_file(path, ocr, embedder, index, metadata)
                    index = result["index"]
                    chunk_count = result["chunk_count"]
                    chunk_total += chunk_count

                # Log processed file with hash|filename format
                if _config.processed_log:
                    file_hash = _calculate_file_hash(path)
                    filename = os.path.basename(path)
                    if file_hash:
                        with open(_config.processed_log, "a", encoding="utf-8") as f:
                            f.write(f"{file_hash}|{filename}\n")

                if file_count % _config.top_print_every == 0:
                    print(
                        f"⚙️ Processed {file_count} files | total chunks: {chunk_total} | already done: {skipped_already_processed} | problems: {skipped_problems}"
                    )

            except Exception as e:
                skipped_problems += 1
                error_type = type(e).__name__
                error_msg = str(e)
                print(
                    f"⚠️ Failed: {os.path.basename(path)} - {error_type}: {error_msg[:100]}"
                )

                # Log detailed crash info to separate file
                with open(config.CRASH_LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"CRASHED FILE: {path}\n")
                    f.write(f"ERROR TYPE: {error_type}\n")
                    f.write(f"ERROR: {error_msg}\n")
                    f.write("TRACEBACK:\n")
                    f.write(traceback.format_exc())
                    f.write(f"\n{'='*60}\n")

                # Track by error type for summary
                skip_reasons[f"{error_type}"] = skip_reasons.get(f"{error_type}", 0) + 1

                # Force cleanup after crash
                with _memory_cleanup():
                    pass

    return {
        "index": index,
        "file_count": file_count,
        "chunk_total": chunk_total,
        "skipped_already_processed": skipped_already_processed,
        "skipped_problems": skipped_problems,
        "skip_reasons": skip_reasons,
    }


def _print_summary(
    file_count: int,
    chunk_total: int,
    skipped_already_processed: int,
    skipped_problems: int,
    skip_reasons: dict[str, int],
) -> None:
    """Print processing summary statistics."""
    assert _config is not None, "_config must be initialized before use"
    print("-------------------------------------------------")
    print(f"✅ Done. Files processed: {file_count} | Chunks: {chunk_total}")
    print(f"📋 Already processed: {skipped_already_processed}")
    print(f"⚠️  Problem files: {skipped_problems}")

    if skip_reasons:
        print("📊 Skip breakdown:")
        for reason, count in sorted(skip_reasons.items()):
            print(f"   • {reason}: {count}")

    print(f"📝  Index: {_config.index_path}")
    print(f"📝 Database: {_config.db_path}")
    if config.INDEX_TYPE.lower() == "ivf":
        print("🎯 Index type: IVF (nlist={config.NLIST}, nprobe={config.NPROBE})")
    else:
        print("🎯 Index type: Flat")
    print("   (Re-run this script after adding new documents.)")


# -----------------------
# Main build
# -----------------------
def build_index(
    root_folder: str, fresh_start: bool = False, recurse_subdirs: bool = True, filetypes: str | None = None
) -> None:
    """Main function to build FAISS index from documents in a folder.

    Args:
        root_folder: Root directory containing documents to process
        fresh_start: If True, clears existing index, database, and processed files log
        recurse_subdirs: If True, processes subdirectories recursively; if False, only root folder
        filetypes: Comma-separated string of file extensions to process (e.g., 'pdf,png,jpg')
    """
    assert (
        _config is not None and _ocr_processor is not None
    ), "Global instances must be initialized before use"
    root_path = Path(root_folder)

    if not root_path.is_dir():
        print(f"❌ Folder not found: {root_path}")
        sys.exit(1)

    # Clear crash log from previous runs
    crash_log_path = Path(config.CRASH_LOG_FILE)
    if crash_log_path.exists():
        crash_log_path.unlink()
        print("🗑️  Cleared previous crash log")

    print(
        f"📁 FAISS exists: {_config.index_path.exists() if _config.index_path else False}"
    )
    print(
        f"📝 Processed log exists: {_config.processed_log.exists() if _config.processed_log else False}"
    )
    print(f"⚙️ Index type: {config.INDEX_TYPE.upper()}")
    if config.INDEX_TYPE.lower() == "ivf":
        print(f"🎯 IVF settings: nlist={config.NLIST}, nprobe={config.NPROBE}")
    # FAISS GPU status (only show if user explicitly enabled it)
    if config.GPU_ENABLED:
        gpu_status = "Active" if _gpu_resources is not None else "Failed"
        print(f"🎮 FAISS GPU acceleration: {gpu_status} (device {config.GPU_DEVICE})")
        if gpu_status == "Failed":
            print("   💡 FAISS GPU unavailable - using CPU (embeddings still use GPU)")
    # When disabled, don't show confusing messages - system is optimized

    # Clean up obsolete pickle files (from pre-SQLite versions)
    pickle_file = root_path / "documents.pkl"
    if pickle_file.exists():
        pickle_file.unlink()
        print("🗑️  Removed obsolete documents.pkl file")

    # Handle fresh start or resume logic
    if fresh_start:
        print("🆕 Fresh start requested - clearing all existing files")
        # Clear index
        if _config.index_path and _config.index_path.exists():
            _config.index_path.unlink()
            print("   Removed existing FAISS index")
        # Clear database
        if _config.db_path and _config.db_path.exists():
            _config.db_path.unlink()
            print("   Removed existing metadata database")
        # Clear processed files log
        if _config.processed_log and _config.processed_log.exists():
            _config.processed_log.unlink()
            print("   Removed processed files log")
    elif (
        _config.processed_log
        and _config.processed_log.exists()
        and _config.index_path
        and _config.index_path.exists()
    ):
        print("ℹ️ Resuming - keeping existing index")
    elif _config.index_path and _config.index_path.exists():
        _config.index_path.unlink()
        if _config.db_path:
            _config.db_path.unlink()
        print("ℹ🧹 Cleared existing index for fresh run")

    # Set up file type filtering if specified
    if filetypes:
        try:
            _config.set_allowed_extensions(filetypes)
        except ValueError as e:
            print(f"❌ Invalid file types: {e}")
            sys.exit(1)

    # Display which file types will be processed
    effective_extensions = _config.get_effective_extensions()
    if _config.allowed_extensions is not None:
        print(f"📄 Processing file types: {', '.join(sorted([ext.lstrip('.') for ext in effective_extensions]))}")
    else:
        print(f"📄 Processing all supported file types: {', '.join(sorted([ext.lstrip('.') for ext in effective_extensions]))}")

    print(f"🔎 Scanning: {root_path}")
    ocr = _ocr_processor
    embedder = _init_embedder()

    # Load existing index and metadata
    index, metadata = _load_existing_index()

    # Load processed files
    processed = _load_processed_files()

    # Check for stale documents (only if not doing fresh start and we have processed files)
    if not fresh_start and processed:
        print("🔍 Checking for stale document references...")

        # Get list of current files in directory (using effective extensions)
        current_files = []
        effective_extensions = _config.get_effective_extensions()
        for ext in effective_extensions:
            # Convert .ext to *.ext for glob pattern
            glob_pattern = "*" + ext
            if recurse_subdirs:
                current_files.extend(root_path.rglob(glob_pattern))
            else:
                current_files.extend(root_path.glob(glob_pattern))

        # Filter out files that should be skipped
        filtered_files = []
        for file_path in current_files:
            if not any(file_path.match(pattern) for pattern in config.SKIP_FILES):
                filtered_files.append(file_path)

        # Detect stale documents
        stale_files = _detect_stale_documents(processed, filtered_files)

        if stale_files:
            user_choice = _prompt_user_stale_action(stale_files)

            if user_choice == "quit":
                print("Operation cancelled by user.")
                sys.exit(0)
            elif user_choice == "fresh":
                print("🆕 User chose fresh start - clearing all existing files")
                fresh_start = True
                # Clear index
                if _config.index_path and _config.index_path.exists():
                    _config.index_path.unlink()
                    print("   Removed existing FAISS index")
                # Clear database
                if _config.db_path and _config.db_path.exists():
                    _config.db_path.unlink()
                    print("   Removed existing metadata database")
                # Clear processed files log
                if _config.processed_log and _config.processed_log.exists():
                    _config.processed_log.unlink()
                    print("   Removed processed files log")
                # Reload empty processed set
                processed = set()
                # Reload empty index and metadata
                index, metadata = None, []
            elif user_choice == "clean":
                print("🧹 User chose clean - removing stale entries")
                _clean_stale_entries(stale_files)
                # Reload processed files and metadata after cleaning
                processed = _load_processed_files()
                index, metadata = _load_existing_index()
            # For 'append', we just continue with existing data

        else:
            print("✅ No stale documents detected")

    # Scan and process files
    if recurse_subdirs:
        print("🔄 Starting file scan and processing (including subdirectories)...")
    else:
        print("🔄 Starting file scan and processing (root folder only)...")
    try:
        stats = _scan_and_process_files(
            root_path, ocr, embedder, index, metadata, processed, recurse_subdirs
        )
        index = stats["index"]
        file_count = stats["file_count"]
        chunk_total = stats["chunk_total"]
        skipped_already_processed = stats["skipped_already_processed"]
        skipped_problems = stats["skipped_problems"]
        skip_reasons = stats["skip_reasons"]
        print(
            f"✅ File processing completed. Got {file_count} files, {chunk_total} chunks"
        )
    except Exception as e:
        print(f"❌ Fatal error during file processing: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

    if index is None or chunk_total == 0:
        print("❌ No text extracted. Nothing indexed.")
        sys.exit(2)

    # Phase 2: Build BM25 index (if enabled)
    if config.ENABLE_HYBRID_SEARCH:
        print("🔨 Building BM25 keyword index...")
        try:
            from utils.bm25_index import build_bm25_index, save_bm25_index

            # Extract texts from metadata for BM25
            texts = [meta["text"] for meta in metadata]

            # Build BM25 index
            bm25_index = build_bm25_index(texts)

            # Save BM25 index
            bm25_path = Path(config.BM25_INDEX_PATH)
            save_bm25_index(bm25_index, bm25_path)

            print(f"✅ BM25 index saved: {bm25_path} ({len(bm25_index)} documents)")

        except ImportError:
            print(
                "⚠️ BM25 indexing failed: rank-bm25 not installed. "
                "Run: uv add rank-bm25"
            )
        except Exception as e:
            print(f"⚠️ BM25 indexing failed: {e}")
            logger.error(f"BM25 index build error: {e}", exc_info=True)
    else:
        logger.info("BM25 indexing disabled (ENABLE_HYBRID_SEARCH=False)")

    # Print summary
    print("📊 Generating summary...")
    _print_summary(
        file_count,
        chunk_total,
        skipped_already_processed,
        skipped_problems,
        skip_reasons,
    )
    print("🎉 Script completed successfully!")


def _process_file(
    path: str,
    ocr: OCRProcessor,
    embedder: SentenceTransformer,
    index: faiss.Index | None,
    metadata:list[dict[str, Any]],
) -> ProcessingResult:
    """
    Process a single file: extract text, chunk, embed, and add to FAISS + metadata.
    Returns (index, chunk_count) - index may be newly created if it was None.
    """
    assert _config is not None, "_config must be initialized before use"
    raw_text = _extract_text(path, ocr)
    text = _clean_text(raw_text)
    chunks = _chunk_text(text, embedder=embedder)

    if not chunks:
        return {"index": index, "chunk_count": 0}  # zero chunks processed

    # Embeddings
    embs: np.ndarray | None = None
    try:
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.float16, enabled=torch.cuda.is_available()
        ):
            embs_raw = embedder.encode(
                chunks,
                batch_size=config.BATCH_SIZE,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            embs = np.asarray(embs_raw).astype(np.float32)

            # Clear CUDA cache after encoding
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except torch.OutOfMemoryError as e:
        logger.error(f"⚠️  CUDA out of memory during embedding for {path}: {e}")
        # Clear CUDA cache and retry with smaller batch size
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Retry with smaller batch size
        try:
            smaller_batch = max(1, config.BATCH_SIZE // config.BATCH_SIZE_RETRY_DIVISOR)
            logger.info(
                f"🔄 Retrying embedding with smaller batch size: {smaller_batch}"
            )
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.float16, enabled=torch.cuda.is_available()
            ):
                embs_raw = embedder.encode(
                    chunks,
                    batch_size=smaller_batch,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                embs = np.asarray(embs_raw).astype(np.float32)

                # Clear CUDA cache after retry
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as retry_e:
            logger.error(f"⚠️  Failed to embed even with smaller batch: {retry_e}")
            return {"index": index, "chunk_count": 0}

    # Initialize FAISS if needed (only happens on very first file)
    if index is None and embs is not None:
        dim = embs.shape[1]
        index, actual_type = _create_faiss_index(
            dim, config.INDEX_TYPE, config.NLIST, _gpu_resources
        )

        # Train IVF index if needed
        if actual_type == "ivf":
            training_success = _train_ivf_index(index, embs)
            if not training_success:
                # Fall back to flat index if training failed
                logger.info("🔄 Creating flat index as fallback...")
                index, _ = _create_faiss_index(dim, "flat", config.NLIST, _gpu_resources)
            else:
                # Set nprobe for search on successful IVF index
                if hasattr(index, "nprobe"):
                    index.nprobe = config.NPROBE  # type: ignore

    # Add to index (using the previously computed embeddings)
    if index is not None and embs is not None:
        # For IVF indices, check if we need retraining
        if (
            config.INDEX_TYPE.lower() == "ivf"
            and hasattr(index, "is_trained")
            and index.is_trained
            and _needs_retraining(index, len(embs))
        ):
            logger.info("🔄 Significant data growth detected, retraining IVF index...")
            # Get all existing vectors + new ones for retraining
            all_vectors = np.vstack([index.reconstruct_n(0, index.ntotal), embs])  # type: ignore

            # Recreate and retrain index
            dim = embs.shape[1]
            new_index, actual_type = _create_faiss_index(
                dim, config.INDEX_TYPE, config.NLIST, _gpu_resources
            )

            if actual_type == "ivf":
                training_success = _train_ivf_index(new_index, all_vectors)
                if training_success:
                    if hasattr(new_index, "nprobe"):
                        new_index.nprobe = config.NPROBE  # type: ignore
                    # Add all vectors to new index
                    new_index.add(all_vectors)  # type: ignore
                    index = new_index
                    logger.info("✅ IVF index retraining completed")
                else:
                    # Retraining failed, fall back to normal addition
                    logger.info("IVF retraining failed, continuing with existing index")
                    index.add(embs)  # type: ignore
            else:
                # Fallback index created, add all vectors
                new_index.add(all_vectors)  # type: ignore
                index = new_index
                logger.info("✅ Switched to flat index with all vectors")
        else:
            # Normal vector addition
            index.add(embs)  # type: ignore

    # Add metadata to SQLite database and list
    db: Any = sqlite_utils.Database(str(_config.db_path))

    # Ensure chunks table exists with proper schema
    chunks_table: Any = db["chunks"]
    if "chunks" not in db.table_names():
        chunks_table.create(
            {
                "id": int,
                "source": str,
                "chunk_index": int,
                "text": str,
                "file_hash": str,  # type: ignore
                "created_at": str,  # type: ignore
            },
            pk="id",
        )  # type: ignore
        # Create index on source for faster queries
        chunks_table.create_index(["source"])  # type: ignore
        chunks_table.create_index(["file_hash"])  # type: ignore

    # Calculate file hash for better deduplication
    file_hash = _calculate_file_hash(path)
    from datetime import datetime

    created_at = datetime.now().isoformat()

    # Insert chunks into database and add to metadata list
    chunk_records = []
    for i, ch in enumerate(chunks):
        chunk_data = {
            "source": path,
            "chunk_index": i,
            "text": ch,
            "file_hash": file_hash,
            "created_at": created_at,
        }
        chunk_records.append(chunk_data)
        metadata.append({"source": path, "chunk_index": i, "text": ch})

    # Batch insert for better performance
    chunks_table.insert_all(chunk_records)  # type: ignore

    # Incremental save (move GPU index to CPU for saving if needed)
    import faiss
    
    save_index = index
    # Check if this looks like a GPU index (has device attribute with numeric value)
    if _gpu_functions_available and index is not None and hasattr(index, "device"):
        try:
            device_val = getattr(index, "device", -1)
            is_gpu_index = hasattr(device_val, "__ge__") and device_val >= 0
        except (AttributeError, TypeError):
            is_gpu_index = False

        if is_gpu_index:  # GPU index
            try:
                save_index = getattr(faiss, "index_gpu_to_cpu")(index)
                logger.debug("Moved GPU index to CPU for saving")
            except Exception as e:
                logger.warning(f"Failed to move GPU index to CPU for saving: {e}")

    if save_index is not None:
        faiss.write_index(save_index, str(_config.index_path))

    # Clean up memory after processing each file
    _cleanup_memory()

    return {"index": index, "chunk_count": len(chunks)}


# -----------------------
# CLI
# -----------------------
def main() -> None:
    """Main entry point for PyRagix document ingestion."""
    # Ensure UTF-8 output for cross-platform emoji support
    try:
        # Try to reconfigure stdout to UTF-8 if supported
        if hasattr(sys.stdout, "reconfigure") and callable(
            getattr(sys.stdout, "reconfigure", None)
        ):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
    except (AttributeError, OSError, Exception):
        # If reconfigure fails, emojis might not display but won't crash
        pass

    ap = argparse.ArgumentParser(
        description="Ingest folder -> FAISS (PDF/HTML/Images with OCR fallback)"
    )
    ap.add_argument(
        "folder",
        nargs="?",
        default=".",
        help="Root folder of documents to process (default: current directory)",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Start from scratch, clearing existing index and processed files log",
    )
    ap.add_argument(
        "--no-recurse",
        action="store_true",
        help="Only process files in the root folder, skip subdirectories",
    )
    ap.add_argument(
        "--filetypes",
        type=str,
        help="Comma-separated list of file extensions to process (e.g., 'pdf,png,jpg'). If not specified, all supported types are processed.",
    )
    args = ap.parse_args()

    # Clear screen for cleaner experience
    os.system("cls" if os.name == "nt" else "clear")

    # Configuration is loaded automatically by config module
    print(f"Using settings from {config.SETTINGS_FILE}")

    # Apply configuration to environment and initialize libraries
    _apply_user_configuration()

    # Announce which folder is being processed
    if args.folder == ".":
        print("No folder specified, using current directory")
    else:
        print(f"Processing folder: {args.folder}")

    _initialize_global_instances()
    build_index(
        args.folder, fresh_start=args.fresh, recurse_subdirs=not args.no_recurse, filetypes=args.filetypes
    )


if __name__ == "__main__":
    main()
