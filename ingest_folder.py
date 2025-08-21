# ===============================
# Standard Library
# ===============================
import logging
import os
import sys
import pickle
import traceback
import math
from io import BytesIO

# ===============================
# Core Numerical / Utility
# ===============================
import numpy as np
import psutil

# ===============================
# GPU Frameworks (order critical)
# ===============================
import torch

print("Torch loaded:", torch.__version__)
from sentence_transformers import SentenceTransformer

# ===============================
# Vector Databases / Indexing
# ===============================
import faiss

# ===============================
# Other AI Frameworks
# ===============================
import paddle
from paddleocr import PaddleOCR

# ===============================
# Imaging / Document Parsers
# ===============================
from PIL import Image
import fitz  # PyMuPDF
from bs4 import BeautifulSoup

DOC_EXTS = {
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
CHUNK_SIZE = 1600  # characters (increased for email threads and complex documents)
CHUNK_OVERLAP = 200  # characters (proportionally increased)
EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "local_faiss.index"
META_PATH = "documents.pkl"
TOP_PRINT_EVERY = 5  # print every N files
PROCESSED_LOG = "processed_files.txt"
TOTAL_RAM_GB = psutil.virtual_memory().total / (1024**3)

if TOTAL_RAM_GB >= 32:
    # High-end systems
    MAX_PIXELS = 1_800_000  # ~1.8 MP per render
    TILE_SIZE = 1200
elif TOTAL_RAM_GB >= 16:
    # Mid-range systems- very conservative due to memory fragmentation
    MAX_PIXELS = 400_000  # ~0.4 MP per render (reduced further)
    TILE_SIZE = 600
else:
    # Low-memory systems
    MAX_PIXELS = 400_000  # ~0.4 MP per render
    TILE_SIZE = 600

MAX_SIDE = 2000  # hard cap on either side
TILE_OVERLAP = 40  # small overlap so words at tile edges aren't cut
USE_OCR_CLS = False  # angle classifier off to save memory

print(
    f"🖥️  Detected {TOTAL_RAM_GB:.1f}GB RAM - using MAX_PIXELS={MAX_PIXELS:,}, TILE_SIZE={TILE_SIZE}"
)

# Configurable skip criteria
MAX_FILE_MB = (
    200  # skip PDFs/images bigger than this (increased for Proximal Origins emails)
)
MAX_PDF_PAGES = 200  # skip PDFs with more pages (increased for important documents)
SKIP_FORM_PDFS = True  # skip PDFs containing form fields


def init_ocr():

    # Suppress PaddleOCR warnings
    logging.getLogger("paddleocr").setLevel(logging.ERROR)

    # Force CPU; angle classifier off (we handle orientation OK in most docs)
    ocr = PaddleOCR(lang="en", use_angle_cls=False, use_gpu=False)

    try:
        dev = getattr(getattr(paddle, "device", None), "get_device", lambda: "cpu")()
        print(f"ℹ️ PaddlePaddle: {paddle.__version__} | Device: {dev}")
    except Exception:
        print("⚠️ Could not print Paddle version/device.")
    return ocr


def init_embedder():
    return SentenceTransformer(EMBED_MODEL)


def clean_text(s: str) -> str:
    # Collapse whitespace; keep newlines sparsely
    return " ".join(s.split())


def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
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


def html_to_text(path: str) -> str:
    # Prefer lxml if available; fall back gracefully
    parser = "lxml"
    try:
        import lxml  # noqa: F401
    except Exception:
        parser = "html.parser"
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, parser)
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    return soup.get_text(separator="\n")


def _safe_dpi_for_page(page, max_pixels=MAX_PIXELS, max_side=MAX_SIDE, base_dpi=150):

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


def _ocr_pil_image(ocr, pil_img):
    arr = np.array(pil_img.convert("RGB"))  # Paddle expects RGB ndarray
    result = ocr.ocr(arr, cls=USE_OCR_CLS)
    if not result or not result[0]:
        return ""
    return "\n".join([line[1][0] for line in result[0]])


def _ocr_page_tiled(ocr, page, dpi, tile_px=TILE_SIZE, overlap=TILE_OVERLAP):
    rect = page.rect
    s = dpi / 72.0
    full_w = int(rect.width * s)
    full_h = int(rect.height * s)

    texts = []
    # number of tiles in each dimension
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
                if tile_px > 800:
                    return _ocr_page_tiled(
                        ocr, page, dpi, tile_px=tile_px // 2, overlap=overlap
                    )
                else:
                    continue
            except Exception:
                continue

    return "\n".join(texts)


def _ocr_embedded_images(doc, page, ocr):
    out = []
    try:
        imgs = page.get_images(full=True) or []
        for xref, *_ in imgs:
            try:
                img = doc.extract_image(xref)
                im = Image.open(BytesIO(img["image"]))
                out.append(_ocr_pil_image(ocr, im))
            except Exception:
                continue
    except Exception:
        pass
    return "\n".join([t for t in out if t.strip()])


def pdf_page_text_or_ocr(page, ocr, doc=None) -> str:
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
        page, max_pixels=MAX_PIXELS, max_side=MAX_SIDE, base_dpi=150
    )

    rect = page.rect
    s = dpi / 72.0
    w_px = int(rect.width * s)
    h_px = int(rect.height * s)

    if w_px <= TILE_SIZE and h_px <= TILE_SIZE:
        try:
            pix = page.get_pixmap(
                matrix=fitz.Matrix(s, s), colorspace=fitz.csGRAY, alpha=False
            )
            png_bytes = pix.getPNGdata()
            im = Image.open(BytesIO(png_bytes))
            return _ocr_pil_image(ocr, im)
        except MemoryError:
            return _ocr_page_tiled(ocr, page, dpi)
        except Exception:
            return ""

    # Larger pages → tiled OCR
    return _ocr_page_tiled(ocr, page, dpi)


def extract_from_pdf(path: str, ocr) -> str:
    out = []
    with fitz.open(path) as doc:
        for p in doc:
            try:
                out.append(pdf_page_text_or_ocr(p, ocr, doc=doc))
            except Exception:
                traceback.print_exc()
                continue
    return "\n".join(out)


def extract_from_image(path: str, ocr) -> str:
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

    except (MemoryError, np.core._exceptions._ArrayMemoryError) as e:
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
        except Exception:
            print(f"⚠️  Failed to process {path} even at reduced size")
            return ""
    except Exception as e:
        print(f"⚠️  OCR failed for {path}: {e}")
        return ""


def extract_text(path: str, ocr) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_from_pdf(path, ocr)
    elif ext in {".html", ".htm"}:
        return html_to_text(path)
    else:
        return extract_from_image(path, ocr)


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def should_skip_file(path: str, ext: str, processed: set) -> tuple[bool, str]:

    # Unsupported extension
    if ext not in DOC_EXTS:
        return True, f"unsupported file type: {ext}"

    if path in processed:
        return True, f"already processed"

    # File size
    file_size_mb = os.path.getsize(path) / 1024 / 1024
    if file_size_mb > MAX_FILE_MB:
        return True, f"large file ({file_size_mb:.1f} MB)"

    # PDF-specific checks
    if ext == ".pdf":
        try:
            with fitz.open(path) as doc:
                if doc.page_count > MAX_PDF_PAGES:
                    return True, f"PDF with {doc.page_count} pages"
                if SKIP_FORM_PDFS:
                    try:
                        if doc.widgets():
                            return True, "form-heavy PDF (has interactive fields)"
                    except AttributeError:
                        # Older PyMuPDF versions don't have widgets() method
                        pass
        except Exception as e:
            return True, f"cannot open PDF: {e}"

    return False, ""  # do not skip


# -----------------------
# Main build
# -----------------------
def build_index(root_folder: str):

    if not os.path.isdir(root_folder):
        print(f"❌ Folder not found: {root_folder}")
        sys.exit(1)

    print(f"📁 FAISS exists: {os.path.exists(INDEX_PATH)}")
    print(f"📝 Processed log exists: {os.path.exists(PROCESSED_LOG)}")

    # Clear existing index if resuming
    if os.path.exists(PROCESSED_LOG) and os.path.exists(INDEX_PATH):
        print("ℹ️ Resuming - keeping existing index")
    elif os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
        os.remove(META_PATH)
        print("ℹ🧹 Cleared existing index for fresh run")

    print(f"🔎 Scanning: {root_folder}")
    ocr = init_ocr()
    embedder = init_embedder()

    # Load existing index and metadata if resuming
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        print("📂 Loading existing index and metadata...")
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            metadata = pickle.load(f)
        print(
            f"   Loaded {index.ntotal} existing chunks from {len(set(m['source'] for m in metadata))} files"
        )
    else:
        index = None
        metadata = []

    file_count = 0
    chunk_total = len(metadata) if metadata else 0
    skipped_already_processed = 0
    skipped_problems = 0
    skip_reasons = {}

    # Read processed files log to avoid reprocessing
    processed = set()
    if os.path.exists(PROCESSED_LOG):
        # Try UTF-8 first, fall back to system default if corrupted
        try:
            with open(PROCESSED_LOG, "r", encoding="utf-8") as f:
                processed = set(line.strip() for line in f)
        except UnicodeDecodeError:
            print("⚠️  Converting processed_files.txt to UTF-8...")
            # Read with system default encoding and rewrite as UTF-8
            with open(PROCESSED_LOG, "r", encoding="cp1252", errors="ignore") as f:
                lines = [line.strip() for line in f]
            with open(PROCESSED_LOG, "w", encoding="utf-8") as f:
                for line in lines:
                    f.write(f"{line}\n")
            processed = set(lines)

    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:

            path = os.path.join(dirpath, fname)
            ext = os.path.splitext(fname)[1].lower()

            skip, reason = should_skip_file(path, ext, processed)
            if skip:
                if reason == "already processed":
                    skipped_already_processed += 1
                    if (
                        file_count % TOP_PRINT_EVERY == 0
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
                index, chunk_count = process_file(path, ocr, embedder, index, metadata)
                chunk_total += chunk_count

                # Force garbage collection after each file to prevent memory fragmentation
                import gc

                gc.collect()

                # Log processed file
                with open(PROCESSED_LOG, "a", encoding="utf-8") as f:
                    f.write(f"{path}\n")

                if file_count % TOP_PRINT_EVERY == 0:
                    total_skipped = skipped_already_processed + skipped_problems
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
                import gc

                gc.collect()

    if index is None or chunk_total == 0:
        print("❌ No text extracted. Nothing indexed.")
        sys.exit(2)

    print("-------------------------------------------------")
    print(f"✅ Done. Files processed: {file_count} | Chunks: {chunk_total}")
    print(f"📋 Already processed: {skipped_already_processed}")
    print(f"⚠️  Problem files: {skipped_problems}")

    if skip_reasons:
        print(f"📊 Skip breakdown:")
        for reason, count in sorted(skip_reasons.items()):
            print(f"   • {reason}: {count}")

    print(f"📝  Index: {INDEX_PATH}")
    print(f"📝 Metadata: {META_PATH}")
    print("   (Re-run this script after adding new documents.)")


def process_file(path, ocr, embedder, index, metadata):
    """
    Process a single file: extract text, chunk, embed, and add to FAISS + metadata.
    Returns (index, chunk_count) - index may be newly created if it was None.
    """
    raw_text = extract_text(path, ocr)
    text = clean_text(raw_text)
    chunks = chunk_text(text)
    if not chunks:
        return index, 0  # zero chunks processed

    # Embeddings
    embs = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=False)
    embs = embs.astype("float32")
    embs = l2_normalize(embs)

    # Initialize FAISS if needed (only happens on very first file)
    if index is None:
        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        print(f"🔧 Created new FAISS index with dimension {dim}")

    # Add to index
    index.add(embs)

    # Add metadata
    for i, ch in enumerate(chunks):
        metadata.append({"source": path, "chunk_index": i, "text": ch})

    # Incremental save
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

    return index, len(chunks)


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    # ap = argparse.ArgumentParser(
    #     description="Ingest folder -> FAISS (PDF/HTML/Images with OCR fallback)"
    # )
    # ap.add_argument("folder", help="Root folder of documents")
    # args = ap.parse_args()
    # build_index(args.folder)

    build_index("C:\\Users\\psarn\\OneDrive\\Documents\\Covid-19")
