# ======================================
# Ingestion script (laptop-optimized)
# - Designed for 16 GB RAM / 6 GB VRAM
# - Torch threads capped at 2
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
import logging
import os
import sys
import pickle
import traceback
import math
import gc
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

# ===============================
# Runtime environment settings
# ===============================
import config

# Cap library threading
os.environ["OPENBLAS_NUM_THREADS"] = str(config.OPENBLAS_NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(config.MKL_NUM_THREADS)
os.environ["OMP_NUM_THREADS"] = str(config.OMP_NUM_THREADS)
os.environ["NUMEXPR_MAX_THREADS"] = str(config.NUMEXPR_MAX_THREADS)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# CUDA settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = config.PYTORCH_CUDA_ALLOC_CONF
os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES

# ===============================
# Core Numerical / Utility
# ===============================
from numpy.typing import NDArray
import numpy as np
import psutil

# ===============================
# GPU Frameworks (order critical)
# ===============================
import torch

torch.set_num_threads(config.TORCH_NUM_THREADS)
print("Torch loaded:", torch.__version__, "CUDA available:", torch.cuda.is_available())

from sentence_transformers import SentenceTransformer

# ===============================
# Vector Databases / Indexing
# ===============================
import faiss

print("FAISS version:", faiss.__version__)

# ===============================
# Other AI Frameworks
# ===============================
import paddle
from paddleocr import PaddleOCR

print("Paddle compiled with CUDA:", paddle.device.is_compiled_with_cuda())

# ===============================
# Imaging / Document Parsers
# ===============================
from PIL import Image
import fitz  # PyMuPDF
from bs4 import BeautifulSoup


@dataclass
class ProcessingConfig:
    """Configuration for document processing and ingestion."""

    # Supported file extensions
    doc_extensions: set[str] = set()

    # Text processing
    chunk_size: int = 1600  # characters
    chunk_overlap: int = 200  # characters
    embed_model: str = "all-MiniLM-L6-v2"

    # File paths
    index_path: Path = Path("local_faiss.index")
    meta_path: Path = Path("documents.pkl")
    processed_log: Path = Path("processed_files.txt")
    # Processing behavior
    top_print_every: int = 5  # print every N files

    # Memory-based settings (set dynamically)
    max_pixels: Optional[int] = None
    tile_size: Optional[int] = None
    max_side: int = 2000  # hard cap on either side
    tile_overlap: int = 40  # small overlap so words at tile edges aren't cut
    use_ocr_cls: bool = False  # angle classifier off to save memory

    # Skip criteria
    max_file_mb: int = 200  # skip PDFs/images bigger than this
    max_pdf_pages: int = 200  # skip PDFs with more pages
    skip_form_pdfs: bool = True  # skip PDFs containing form fields
    skip_files: set[str] = set()  # Hard-coded list of files to skip

    # Default folder to process (can be overridden by command line)
    default_folder: str = "C:\\Users\\psarn\\OneDrive\\Documents\\Covid-19"

    def __post_init__(self):
        if not self.doc_extensions:
            self.doc_extensions = {
                ".pdf",
                ".html",
                ".htm",
                ".png",
                ".jpg",
                ".jpeg",
                ".tif",
                ".tiff",
                ".bmp",
                ".webp",
            }

        if not self.skip_files:
            self.skip_files = config.SKIP_FILES

        # Set memory-based parameters
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        if total_ram_gb >= 32:
            # High-end systems
            self.max_pixels = 1_800_000  # ~1.8 MP per render
            self.tile_size = 1200
        elif total_ram_gb >= 16:
            # Mid-range systems- very conservative due to memory fragmentation
            self.max_pixels = 400_000  # ~0.4 MP per render (reduced further)
            self.tile_size = 600
        else:
            # Low-memory systems
            self.max_pixels = 400_000  # ~0.4 MP per render
            self.tile_size = 600

        print(
            f"🖥️  Detected {total_ram_gb:.1f}GB RAM - using MAX_PIXELS={self.max_pixels:,}, TILE_SIZE={self.tile_size}"
        )


# Global config instance
CONFIG = ProcessingConfig()


@contextmanager
def _memory_cleanup():
    """Context manager for automatic memory cleanup after processing."""
    try:
        yield
    finally:
        # Force garbage collection after operations to prevent memory fragmentation
        gc.collect()
        # Keep VRAM stable
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _init_ocr() -> PaddleOCR:

    # Suppress PaddleOCR warnings
    logging.getLogger("paddleocr").setLevel(logging.ERROR)

    # Force CPU; angle classifier off (we handle orientation OK in most docs)
    ocr = PaddleOCR(lang="en", use_angle_cls=False, use_gpu=False)

    try:
        dev = getattr(getattr(paddle, "device", None), "get_device", lambda: "cpu")()
        print(f"ℹ️ PaddlePaddle: {paddle.__version__} | Device: {dev}")
    except (AttributeError, TypeError):
        print("⚠️ Could not print Paddle version/device.")
    return ocr


def _init_embedder() -> SentenceTransformer:
    return SentenceTransformer(CONFIG.embed_model)


def _clean_text(s: str) -> str:
    # Collapse whitespace; keep newlines sparsely
    return " ".join(s.split())


def _chunk_text(
    text: str, size=CONFIG.chunk_size, overlap=CONFIG.chunk_overlap
) -> list[str]:
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
        import lxml
    except ImportError:
        parser = "html.parser"
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, parser)
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    return soup.get_text(separator="\n")


def _safe_dpi_for_page(
    page, max_pixels=CONFIG.max_pixels, max_side=CONFIG.max_side, base_dpi=150
):
    """Calculate safe DPI for page rendering to avoid memory issues.

    Args:
        page: PyMuPDF page object
        max_pixels: Maximum total pixels allowed (width * height)
        max_side: Maximum pixels for either width or height
        base_dpi: Starting DPI to scale down from

    Returns:
        int: Safe DPI value (minimum 72)
    """
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
    if w0 * h0 > max_pixels:
        scale *= math.sqrt(max_pixels / (w0 * h0))
    if w0 * scale > max_side:
        scale *= max_side / (w0 * scale)
    if h0 * scale > max_side:
        scale *= max_side / (h0 * scale)
    dpi = max(
        72, int(base_dpi * scale)
    )  # don’t go below 72 unless you want more aggressive downscale
    return dpi


def _ocr_pil_image(ocr, pil_img) -> str:
    arr = np.array(pil_img.convert("RGB"))  # Paddle expects RGB ndarray
    result = ocr.ocr(arr, cls=CONFIG.use_ocr_cls)
    if not result or not result[0]:
        return ""
    return "\n".join([line[1][0] for line in result[0]])


def _ocr_page_tiled(
    ocr, page, dpi, tile_px=CONFIG.tile_size, overlap=CONFIG.tile_overlap
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
    rect = page.rect
    s = dpi / 72.0
    full_w = int(rect.width * s)
    full_h = int(rect.height * s)

    texts = []
    # number of tiles in each dimension
    if tile_px is None:
        tile_px = CONFIG.tile_size or 600  # fallback if CONFIG.tile_size is also None
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
                # GRAY, no alpha massively reduces memory (n=1 channel)
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(s, s),
                    colorspace=fitz.csGRAY,
                    alpha=False,
                    clip=clip,
                )
                # Avoid pix.samples → use compressed PNG bytes
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


def _ocr_embedded_images(doc, page, ocr) -> str:
    out = []
    try:
        imgs = page.get_images(full=True) or []
        for xref, *_ in imgs:
            try:
                img = doc.extract_image(xref)
                im = Image.open(BytesIO(img["image"]))
                out.append(_ocr_pil_image(ocr, im))
            except (KeyError, OSError, ValueError):
                # Image extraction/processing errors
                continue
    except (AttributeError, RuntimeError):
        # PDF processing errors
        pass
    return "\n".join([t for t in out if t.strip()])


def _pdf_page_text_or_ocr(page, ocr, doc=None) -> str:
    """Extract text from PDF page using native text first, then OCR fallback.

    Args:
        page: PyMuPDF page object
        ocr: PaddleOCR instance
        doc: PyMuPDF document object (optional, for embedded image extraction)

    Returns:
        str: Extracted text from the page
    """
    # 1) Native text first
    text = page.get_text("text") or ""
    if len(text.strip()) > 20:
        return text

    # 2) Try embedded images (cheaper than full render)
    if doc is not None:
        emb_txt = _ocr_embedded_images(doc, page, ocr)
        if emb_txt.strip():
            return emb_txt

    # 3) OCR path with safe DPI, grayscale, tiling
    dpi = _safe_dpi_for_page(
        page, max_pixels=CONFIG.max_pixels, max_side=CONFIG.max_side, base_dpi=150
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
            return _ocr_pil_image(ocr, im)
        except MemoryError:
            return _ocr_page_tiled(ocr, page, dpi)
        except (OSError, RuntimeError, ValueError):
            # Image processing/OCR errors
            return ""

    # Larger pages → tiled OCR
    return _ocr_page_tiled(ocr, page, dpi)


def _extract_from_pdf(path: str, ocr) -> str:
    out = []
    with fitz.open(path) as doc:  # type: ignore[attr-defined]
        for p in doc:
            try:
                out.append(_pdf_page_text_or_ocr(p, ocr, doc=doc))
            except (RuntimeError, MemoryError, OSError) as e:
                print(f"⚠️ Error processing PDF page: {type(e).__name__}: {e}")
                traceback.print_exc()
                continue
    return "\n".join(out)


def _extract_from_image(path: str, ocr) -> str:
    # Open -> RGB -> ndarray -> OCR with memory error handling
    try:
        with Image.open(path) as im:
            # Much more aggressive sizing for 16GB RAM systems
            max_pixels = 512 * 512  # 0.25MP max
            if im.width * im.height > max_pixels:
                scale = (max_pixels / (im.width * im.height)) ** 0.5
                new_w = max(128, int(im.width * scale))  # Don't go too small
                new_h = max(128, int(im.height * scale))
                im = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
            im = im.convert("RGB")
            arr = np.array(im)

        result = ocr.ocr(arr, cls=False)  # cls=False to save memory
        if not result or not result[0]:
            return ""
        return "\n".join([line[1][0] for line in result[0]])

    except (MemoryError, np.exceptions._ArrayMemoryError) as e:  # type: ignore[attr-defined]
        print(f"⚠️  Memory error for {path}, trying smaller size: {e}")
        try:
            # Try again with much smaller image
            with Image.open(path) as im:
                im.thumbnail((512, 512), Image.Resampling.LANCZOS)
                im = im.convert("RGB")
                arr = np.array(im)
            result = ocr.ocr(arr, cls=False)
            if not result or not result[0]:
                return ""
            return "\n".join([line[1][0] for line in result[0]])
        except (MemoryError, np.exceptions._ArrayMemoryError):  # type: ignore[attr-defined]
            print(f"⚠️  Still out of memory for {path} even at reduced size")
            return ""
        except (OSError, ValueError, RuntimeError) as e:
            print(
                f"⚠️  Failed to process {path} even at reduced size: {type(e).__name__}: {e}"
            )
            return ""
    except (OSError, ValueError) as e:
        print(f"⚠️  Image processing failed for {path}: {type(e).__name__}: {e}")
        return ""
    except RuntimeError as e:
        print(f"⚠️  OCR failed for {path}: {e}")
        return ""


def _extract_text(path: str, ocr) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _extract_from_pdf(path, ocr)
    elif ext in {".html", ".htm"}:
        return _html_to_text(path)
    else:
        return _extract_from_image(path, ocr)


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def _should_skip_file(path: str, ext: str, processed: set) -> tuple[bool, str]:
    """Determine if a file should be skipped during processing.

    Args:
        path: Full file path
        ext: File extension (lowercase)
        processed: Set of already processed file paths

    Returns:
        tuple[bool, str]: (should_skip, reason)
    """
    # Check if filename is in skip list
    filename = os.path.basename(path)
    if filename in CONFIG.skip_files:
        return True, f"file in hard-coded skip list."

    # Unsupported extension
    if ext not in CONFIG.doc_extensions:
        return True, f"unsupported file type: {ext}"

    if path in processed:
        return True, f"already processed"

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


def _load_existing_index() -> tuple[Optional[faiss.Index], list[dict]]:
    """Load existing FAISS index and metadata if they exist."""
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


def _load_processed_files() -> set[str]:
    """Load the set of already processed files from the log."""
    processed = set()
    if CONFIG.processed_log and CONFIG.processed_log.exists():
        # Try UTF-8 first, fall back to system default if corrupted
        try:
            if CONFIG.processed_log:
                with open(CONFIG.processed_log, "r", encoding="utf-8") as f:
                    processed = set(line.strip() for line in f)
        except UnicodeDecodeError:
            print("⚠️  Converting processed_files.txt to UTF-8...")
            # Read with system default encoding and rewrite as UTF-8
            if CONFIG.processed_log:
                with open(
                    CONFIG.processed_log, "r", encoding="cp1252", errors="ignore"
                ) as f:
                    lines = [line.strip() for line in f]
                with open(CONFIG.processed_log, "w", encoding="utf-8") as f:
                    for line in lines:
                        f.write(f"{line}\n")
            processed = set(lines)
    return processed


def _scan_and_process_files(
    root_path, ocr, embedder, index, metadata, processed
) -> tuple[Optional[faiss.Index], int, int, int, int, dict[str, int]]:
    """Scan directory and process all supported files."""
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
                    index, chunk_count = _process_file(
                        path, ocr, embedder, index, metadata
                    )
                    chunk_total += chunk_count

                # Log processed file
                if CONFIG.processed_log:
                    with open(CONFIG.processed_log, "a", encoding="utf-8") as f:
                        f.write(f"{path}\n")

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
                with open("crash_log.txt", "a", encoding="utf-8") as f:
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

    return (
        index,
        file_count,
        chunk_total,
        skipped_already_processed,
        skipped_problems,
        skip_reasons,
    )


def _print_summary(
    file_count, chunk_total, skipped_already_processed, skipped_problems, skip_reasons
) -> None:
    """Print processing summary statistics."""
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
    root_path = Path(root_folder)

    if not root_path.is_dir():
        print(f"❌ Folder not found: {root_path}")
        sys.exit(1)

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
    ocr = _init_ocr()
    embedder = _init_embedder()

    # Load existing index and metadata
    index, metadata = _load_existing_index()

    # Load processed files
    processed = _load_processed_files()

    # Scan and process files
    (
        index,
        file_count,
        chunk_total,
        skipped_already_processed,
        skipped_problems,
        skip_reasons,
    ) = _scan_and_process_files(root_path, ocr, embedder, index, metadata, processed)

    if index is None or chunk_total == 0:
        print("❌ No text extracted. Nothing indexed.")
        sys.exit(2)

    # Print summary
    _print_summary(
        file_count,
        chunk_total,
        skipped_already_processed,
        skipped_problems,
        skip_reasons,
    )


def _process_file(
    path, ocr, embedder, index, metadata
) -> tuple[Optional[faiss.Index], int]:
    """
    Process a single file: extract text, chunk, embed, and add to FAISS + metadata.
    Returns (index, chunk_count) - index may be newly created if it was None.
    """
    raw_text = _extract_text(path, ocr)
    text = _clean_text(raw_text)
    chunks = _chunk_text(text)

    if not chunks:
        return index, 0  # zero chunks processed

    # Embeddings
    embs: Optional[np.ndarray] = None
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.float16, enabled=torch.cuda.is_available()
    ):
        embs = embedder.encode(
            chunks,
            batch_size=config.BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

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

    return index, len(chunks)


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Ingest folder -> FAISS (PDF/HTML/Images with OCR fallback)"
    )
    ap.add_argument(
        "folder",
        nargs="?",
        default=CONFIG.default_folder,
        help=f"Root folder of documents (default: {CONFIG.default_folder})",
    )
    args = ap.parse_args()
    build_index(args.folder)
