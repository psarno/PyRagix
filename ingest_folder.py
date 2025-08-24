# ======================================
# Ingestion script (laptop-optimized)
# - Designed for 16 GB RAM / 6 GB VRAM
# - Torch threads capped at 4
# - CUDA alloc split tuned for GTX 1660 Ti
# - PaddleOCR runs with device fallback
#
# If porting to a bigger host:
#   - Increase thread caps
#   - Increase batch_size
#   - Consider FAISS GPU or HNSW index
# ======================================

# ===============================
# Standard Library
# ===============================
import argparse
import json
import logging
import os
import sys
import pickle
import traceback
import math
import gc
import hashlib
from io import BytesIO
from pathlib import Path
from typing import (
    Optional,
    Union,
    Tuple,
    List,
    Dict,
    Set,
    Any,
    Iterator,
    Protocol,
    cast,
)
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
    def get_images(self, full: bool = ...) -> List[Tuple[int, ...]]: ...


class PDFDocument(Protocol):
    page_count: int

    def __iter__(self) -> Iterator[PDFPage]: ...
    def extract_image(self, xref: int) -> Optional[Dict[str, Any]]: ...
    def widgets(self) -> List[Any]: ...


# Type definitions
class ProcessingStats(TypedDict):
    index: Optional["faiss.Index"]
    file_count: int
    chunk_total: int
    skipped_already_processed: int
    skipped_problems: int
    skip_reasons: Dict[str, int]


class ProcessingResult(TypedDict):
    index: Optional["faiss.Index"]
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


# ===============================
# Import with suppressed output
# ===============================
# Core dependencies
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup

# Heavy imports with targeted logging suppression
import torch
from sentence_transformers import SentenceTransformer
import faiss

from paddleocr import PaddleOCR
import fitz  # PyMuPDF
from classes.ProcessingConfig import ProcessingConfig
from classes.OCRProcessor import OCRProcessor

# Global instances - will be initialized after configuration is loaded
CONFIG: Optional[ProcessingConfig] = None
OCR_PROCESSOR: Optional[OCRProcessor] = None


def _apply_user_configuration(user_config: Optional[Dict[str, Any]] = None) -> None:
    """Apply user configuration from JSON file to environment variables and torch settings."""
    if user_config is None:
        # Try to load from user_config.json
        config_file = "user_config.json"
        if os.path.exists(config_file):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    user_config = json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning(f"Could not load {config_file}, using defaults")
                user_config = {}
        else:
            user_config = {}

    # Environment variables that need to be set before library imports
    env_vars = [
        "TORCH_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
        "CUDA_VISIBLE_DEVICES",
        "PYTORCH_CUDA_ALLOC_CONF",
    ]

    for var in env_vars:
        if user_config and var in user_config:
            os.environ[var] = str(user_config[var])
        elif hasattr(config, var):
            # Fallback to config.py defaults
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

    # Configure torch with user settings
    torch_threads = (user_config or {}).get(
        "TORCH_NUM_THREADS", getattr(config, "TORCH_NUM_THREADS", 1)
    )
    torch.set_num_threads(torch_threads)

    logger.info(
        f"Torch loaded: {torch.__version__}, CUDA available: {torch.cuda.is_available()}"
    )
    logger.info(f"FAISS version: {faiss.__version__}")
    logger.info(f"Paddle compiled with CUDA: {paddle.device.is_compiled_with_cuda()}")


def _initialize_global_instances() -> None:
    """Initialize global CONFIG and OCR_PROCESSOR instances."""
    global CONFIG, OCR_PROCESSOR
    CONFIG = ProcessingConfig()
    OCR_PROCESSOR = OCRProcessor(CONFIG)


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


def _init_embedder() -> SentenceTransformer:
    assert CONFIG is not None, "CONFIG must be initialized before use"
    return SentenceTransformer(CONFIG.embed_model)


def _clean_text(s: str) -> str:
    # Collapse whitespace; keep newlines sparsely
    return " ".join(s.split())


def _chunk_text(
    text: str, size: Optional[int] = None, overlap: Optional[int] = None
) -> List[str]:
    # Resolve config values if not provided
    if size is None:
        assert CONFIG is not None, "CONFIG must be initialized before use"
        size = CONFIG.chunk_size
    if overlap is None:
        assert CONFIG is not None, "CONFIG must be initialized before use"
        overlap = CONFIG.chunk_overlap

    text = text.strip()
    if not text:
        return []
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
    max_pixels: Optional[int] = None,
    max_side: Optional[int] = None,
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
        assert CONFIG is not None, "CONFIG must be initialized before use"
        max_pixels = CONFIG.max_pixels
    if max_side is None:
        assert CONFIG is not None, "CONFIG must be initialized before use"
        max_side = CONFIG.max_side

    # page.rect is in points; 72 points = 1 inch
    rect = page.rect
    if rect.width == 0 or rect.height == 0:
        return 96

    # scale from DPI: pixels = points/72 * dpi
    def px_for(dpi):
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
    assert CONFIG is not None, "CONFIG must be initialized before use"
    try:
        arr = np.array(
            pil_img.convert("RGB"), dtype=np.uint8
        )  # Paddle expects RGB ndarray
        result = ocr.ocr(arr, cls=CONFIG.use_ocr_cls)
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
    tile_px: Optional[int] = None,
    overlap: Optional[int] = None,
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
        assert CONFIG is not None, "CONFIG must be initialized before use"
        tile_px = CONFIG.tile_size
    if overlap is None:
        assert CONFIG is not None, "CONFIG must be initialized before use"
        overlap = CONFIG.tile_overlap

    rect = page.rect
    s = dpi / 72.0
    full_w = int(rect.width * s)
    full_h = int(rect.height * s)

    texts = []
    # number of tiles in each dimension
    # tile_px is resolved from config above, but add fallback for safety
    if tile_px is None:
        tile_px = 600  # fallback if CONFIG.tile_size is also None
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
                png_bytes = pix.getPNGdata()
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
    page: Any, ocr: OCRProcessor, doc: Optional[Any] = None
) -> str:
    """Extract text from PDF page using native text first, then OCR fallback.

    Args:
        page: PyMuPDF page object
        ocr: PaddleOCR instance
        doc: PyMuPDF document object (optional, for embedded image extraction)

    Returns:
        str: Extracted text from the page
    """
    assert CONFIG is not None, "CONFIG must be initialized before use"
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
        max_pixels=CONFIG.max_pixels,
        max_side=CONFIG.max_side,
        base_dpi=config.BASE_DPI,
    )

    rect = page.rect
    s = dpi / 72.0
    w_px = int(rect.width * s)
    h_px = int(rect.height * s)

    if (
        CONFIG.tile_size is not None
        and w_px <= CONFIG.tile_size
        and h_px <= CONFIG.tile_size
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


def _should_skip_file(path: str, ext: str, processed: Set[str]) -> Tuple[bool, str]:
    """Determine if a file should be skipped during processing.

    Args:
        path: Full file path
        ext: File extension (lowercase)
        processed: Set of already processed file hashes

    Returns:
        tuple[bool, str]: (should_skip, reason)
    """
    assert CONFIG is not None, "CONFIG must be initialized before use"
    # Check if filename is in skip list
    filename = os.path.basename(path)
    if filename in CONFIG.skip_files:
        return True, f"file in hard-coded skip list."

    # Unsupported extension
    if ext not in CONFIG.doc_extensions:
        return True, f"unsupported file type: {ext}"

    # Check if file hash is already processed
    file_hash = _calculate_file_hash(path)
    if file_hash and file_hash in processed:
        return True, "already processed"

    # File size
    file_size_mb = os.path.getsize(path) / 1024 / 1024
    if file_size_mb > CONFIG.max_file_mb:
        return True, f"large file ({file_size_mb:.1f} MB)"

    # PDF-specific checks
    if ext == ".pdf":
        try:
            with fitz.open(path) as doc:  # type: ignore[attr-defined]
                if doc.page_count > CONFIG.max_pdf_pages:
                    return True, f"PDF with {doc.page_count} pages"
                if CONFIG.skip_form_pdfs:
                    try:
                        if doc.widgets():
                            return True, "form-heavy PDF (has interactive fields)"
                    except AttributeError:
                        # Older PyMuPDF versions don't have widgets() method
                        pass
        except (OSError, RuntimeError, ValueError) as e:
            return True, f"cannot open PDF: {e}"

    return False, ""  # do not skip


def _load_existing_index() -> Tuple[Optional["faiss.Index"], List[Dict[str, Any]]]:
    """Load existing FAISS index and metadata if they exist."""
    assert CONFIG is not None, "CONFIG must be initialized before use"
    if (
        CONFIG.index_path
        and CONFIG.index_path.exists()
        and CONFIG.meta_path
        and CONFIG.meta_path.exists()
    ):
        print("📂 Loading existing index and metadata...")
        index = faiss.read_index(str(CONFIG.index_path))
        with open(CONFIG.meta_path, "rb") as f:
            metadata = pickle.load(f)
        print(
            f"   Loaded {index.ntotal} existing chunks from {len(set(m['source'] for m in metadata))} files"
        )
        return index, metadata
    else:
        return None, []


def _load_processed_files() -> Set[str]:
    """Load the set of already processed file hashes from the log.

    Expected format: "hash|filename"

    Returns:
        set[str]: Set of file hashes that have been processed
    """
    assert CONFIG is not None, "CONFIG must be initialized before use"
    processed_hashes = set()

    if CONFIG.processed_log and CONFIG.processed_log.exists():
        try:
            with open(CONFIG.processed_log, "r", encoding="utf-8") as f:
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
                CONFIG.processed_log, "r", encoding="cp1252", errors="ignore"
            ) as f:
                lines = [line.strip() for line in f if line.strip()]
            with open(CONFIG.processed_log, "w", encoding="utf-8") as f:
                for line in lines:
                    f.write(f"{line}\n")
            # Retry reading
            return _load_processed_files()

    return processed_hashes


def _scan_and_process_files(
    root_path: Union[str, Path],
    ocr: OCRProcessor,
    embedder: SentenceTransformer,
    index: Optional["faiss.Index"],
    metadata: List[Dict[str, Any]],
    processed: Set[str],
) -> ProcessingStats:
    """Scan directory and process all supported files."""
    assert CONFIG is not None, "CONFIG must be initialized before use"
    file_count = 0
    chunk_total = len(metadata) if metadata else 0
    skipped_already_processed = 0
    skipped_problems = 0
    skip_reasons = {}

    for dirpath, _, filenames in os.walk(root_path):
        for fname in filenames:
            path = os.path.join(dirpath, fname)
            ext = os.path.splitext(fname)[1].lower()

            skip, reason = _should_skip_file(path, ext, processed)
            if skip:
                if reason == "already processed":
                    skipped_already_processed += 1
                    if (
                        file_count % CONFIG.top_print_every == 0
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
                if CONFIG.processed_log:
                    file_hash = _calculate_file_hash(path)
                    filename = os.path.basename(path)
                    if file_hash:
                        with open(CONFIG.processed_log, "a", encoding="utf-8") as f:
                            f.write(f"{file_hash}|{filename}\n")

                if file_count % CONFIG.top_print_every == 0:
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
                    f.write(f"TRACEBACK:\n")
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
    skip_reasons: Dict[str, int],
) -> None:
    """Print processing summary statistics."""
    assert CONFIG is not None, "CONFIG must be initialized before use"
    print("-------------------------------------------------")
    print(f"✅ Done. Files processed: {file_count} | Chunks: {chunk_total}")
    print(f"📋 Already processed: {skipped_already_processed}")
    print(f"⚠️  Problem files: {skipped_problems}")

    if skip_reasons:
        print(f"📊 Skip breakdown:")
        for reason, count in sorted(skip_reasons.items()):
            print(f"   • {reason}: {count}")

    print(f"📝  Index: {CONFIG.index_path}")
    print(f"📝 Metadata: {CONFIG.meta_path}")
    print("   (Re-run this script after adding new documents.)")


# -----------------------
# Main build
# -----------------------
def build_index(root_folder: str) -> None:
    """Main function to build FAISS index from documents in a folder."""
    assert (
        CONFIG is not None and OCR_PROCESSOR is not None
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
        f"📁 FAISS exists: {CONFIG.index_path.exists() if CONFIG.index_path else False}"
    )
    print(
        f"📝 Processed log exists: {CONFIG.processed_log.exists() if CONFIG.processed_log else False}"
    )

    # Clear existing index if resuming
    if (
        CONFIG.processed_log
        and CONFIG.processed_log.exists()
        and CONFIG.index_path
        and CONFIG.index_path.exists()
    ):
        print("ℹ️ Resuming - keeping existing index")
    elif CONFIG.index_path and CONFIG.index_path.exists():
        CONFIG.index_path.unlink()
        if CONFIG.meta_path:
            CONFIG.meta_path.unlink()
        print("ℹ🧹 Cleared existing index for fresh run")

    print(f"🔎 Scanning: {root_path}")
    ocr = OCR_PROCESSOR
    embedder = _init_embedder()

    # Load existing index and metadata
    index, metadata = _load_existing_index()

    # Load processed files
    processed = _load_processed_files()

    # Scan and process files
    print("🔄 Starting file scan and processing...")
    try:
        stats = _scan_and_process_files(
            root_path, ocr, embedder, index, metadata, processed
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
    index: Optional["faiss.Index"],
    metadata: List[Dict[str, Any]],
) -> ProcessingResult:
    """
    Process a single file: extract text, chunk, embed, and add to FAISS + metadata.
    Returns (index, chunk_count) - index may be newly created if it was None.
    """
    assert CONFIG is not None, "CONFIG must be initialized before use"
    raw_text = _extract_text(path, ocr)
    text = _clean_text(raw_text)
    chunks = _chunk_text(text)

    if not chunks:
        return {"index": index, "chunk_count": 0}  # zero chunks processed

    # Embeddings
    embs: Optional[np.ndarray] = None
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
        index = faiss.IndexFlatIP(dim)
        print(f"🔧 Created new FAISS index with dimension {dim}")

    # Add to index (using the previously computed embeddings)
    if index is not None and embs is not None:
        # Type ignore for FAISS binding issues
        index.add(embs)  # type: ignore

    # Add metadata
    for i, ch in enumerate(chunks):
        metadata.append({"source": path, "chunk_index": i, "text": ch})

    # Incremental save
    faiss.write_index(index, str(CONFIG.index_path))
    with open(CONFIG.meta_path, "wb") as f:
        pickle.dump(metadata, f)

    # Clean up memory after processing each file
    _cleanup_memory()

    return {"index": index, "chunk_count": len(chunks)}


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Ingest folder -> FAISS (PDF/HTML/Images with OCR fallback)"
    )
    ap.add_argument(
        "folder",
        help="Root folder of documents to process",
    )
    ap.add_argument(
        "--config-menu",
        action="store_true",
        help="Launch configuration menu to modify settings",
    )
    args = ap.parse_args()

    if not args.folder:
        print("📁 Error: Please specify a folder path to process!")
        print("Usage: python ingest_folder.py <folder_path>")
        print("       python ingest_folder.py <folder_path> --config-menu")
        sys.exit(1)

    # Check if this is first run or user requested config menu
    config_file = "user_config.json"
    first_run = not Path(config_file).exists()

    user_config = None

    if first_run or args.config_menu:
        try:
            from menu import launch_config_menu

            # Clear screen for cleaner menu experience
            os.system("cls" if os.name == "nt" else "clear")

            if first_run:
                print("🔧 First run detected! Launching configuration setup...")
                print("   This will help optimize settings for your system.")
            else:
                print("🔧 Launching configuration menu...")

            user_config = launch_config_menu(config_file)
            if user_config:
                print("✅ Configuration saved! Using custom settings for ingestion.")
            else:
                if first_run:
                    print(
                        "⚠️ Configuration cancelled. Using default settings for first run."
                    )
                else:
                    print("⚠️ Configuration cancelled. Using existing settings.")
        except ImportError:
            print("⚠️ Configuration menu not available. Install dependencies:")
            print("   pip install prompt_toolkit>=3.0.51")
            if first_run:
                print("   Continuing with default settings for first run...")
        except Exception as e:
            print(f"⚠️ Error with configuration menu: {e}")
            print("Continuing with default settings...")
    elif Path(config_file).exists():
        # Load existing user config if available
        try:
            from menu import ConfigMenu

            menu = ConfigMenu(config_file)
            user_config = menu.load_params()
            print(f"📋 Loading existing configuration from {config_file}")
            print("✅ Custom configuration loaded successfully!")
        except Exception as e:
            print(f"⚠️ Error loading user config: {e}")
            print("Using default configuration...")

    # Apply user configuration to environment and initialize libraries
    _apply_user_configuration(user_config)

    # Apply configuration to config module for backwards compatibility
    if user_config:
        for key, value in user_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    config.validate_config()

    # Initialize global instances after configuration is applied
    _initialize_global_instances()

    build_index(args.folder)
