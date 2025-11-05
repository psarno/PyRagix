"""High-level OCR orchestration with PaddleOCR.

The ingestion pipeline leans on this module to keep Paddle's global state
contained: noisy glog output is muted before the native libraries load, cache
locations are surfaced to the operator, and CUDA DLL search paths are patched
on Windows so PaddleOCR can discover NVIDIA runtimes.  Public methods favour
defensive memory management because OCR commonly runs alongside PDF rendering
and FAISS embedding in the same process.
"""

import gc
import logging
import math
import os
import sys
import warnings
from io import BytesIO
from typing import TYPE_CHECKING

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

# Suppress misleading PaddlePaddle ccache warning BEFORE any imports
# (only relevant when building from source, not using pre-built wheels)
warnings.filterwarnings("ignore", message=".*ccache.*", category=UserWarning)

# Suppress C++ library logging (glog) before importing paddle
# These must be set before paddle's C++ libs initialize
_ = os.environ.setdefault("GLOG_minloglevel", "2")
_ = os.environ.setdefault("GLOG_v", "0")
_ = os.environ.setdefault("FLAGS_v", "0")

import paddle  # noqa: E402
from paddleocr import PaddleOCR  # noqa: E402

# Check if PPOCR_HOME is set - if not, warn user about the default location
if "PPOCR_HOME" not in os.environ:
    default_cache = os.path.normpath(os.path.expanduser("~/.paddlex"))
    _ = sys.stderr.write(
        (
            f"WARNING: PPOCR_HOME not set. PaddleOCR will download ~500MB-1GB of models to:\n"
            f"         {default_cache}\n"
            "         To change this location, set the PPOCR_HOME environment variable.\n"
            "         Example: export PPOCR_HOME=/your/custom/path/.paddlex\n\n"
        )
    )

if TYPE_CHECKING:
    from classes.ProcessingConfig import ProcessingConfig
    from ingestion.models import PDFDocument, PDFPage, PILImage

if sys.platform == "win32":
    base = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
    for sub in [
        "cudnn",
        "cublas",
        "cufft",
        "curand",
        "cusolver",
        "cusparse",
        "cuda_runtime",
    ]:
        bin_path = os.path.join(base, sub, "bin")
        if os.path.isdir(bin_path):
            # Ensure Paddle finds CUDA DLLs when Python is installed via the store.
            _ = os.add_dll_directory(bin_path)

# Set up logger
logger = logging.getLogger(__name__)

ImageInfo = tuple[int, ...]


class OCRProcessor:
    """Handles all OCR operations with PaddleOCR."""

    def __init__(self, config: "ProcessingConfig"):
        super().__init__()
        self.config = config
        self.ocr = self._init_ocr()

    def _init_ocr(self) -> PaddleOCR:
        """Initialize PaddleOCR with project defaults and log device metadata.

        PaddleOCR must be constructed after environment variables are enforced
        (see `ingestion.environment.EnvironmentManager`) otherwise threads,
        CUDA visibility, or glog verbosity may drift from configured values.
        """
        # Suppress PaddleOCR warnings
        logging.getLogger("paddleocr").setLevel(logging.ERROR)

        # Angle classifier off (we handle orientation OK in most docs)
        # Note: use_gpu removed in newer PaddleOCR versions, it auto-detects
        ocr = PaddleOCR(lang="en", use_angle_cls=False)

        try:
            dev = getattr(
                getattr(paddle, "device", None), "get_device", lambda: "cpu"
            )()
            logger.info(
                f"ℹ️ PaddlePaddle: {getattr(paddle, '__version__', 'unknown')} | Device: {dev}"
            )
        except (AttributeError, TypeError):
            logger.warning("⚠️ Could not print Paddle version/device.")
        return ocr

    def _cleanup_memory(self) -> None:
        """Force garbage collection to free up memory."""
        _ = gc.collect()
        # Clear Paddle's memory cache if available
        try:
            paddle.device.cuda.empty_cache()
        except (AttributeError, RuntimeError):
            # Not using CUDA or method not available
            pass

    def ocr_pil_image(self, pil_img: "PILImage") -> str:
        """Extract text from PIL image using OCR."""
        try:
            # Convert and process
            rgb_img = pil_img.convert("RGB")
            arr = np.array(rgb_img)

            # Clear RGB image from memory
            if rgb_img is not pil_img:
                rgb_img.close()

            result = self.ocr.predict(arr)

            # Clear array from memory
            del arr

            if not result:
                return ""
            first_result = result[0] if len(result) > 0 else None
            if not first_result:
                return ""
            return "\n".join([line[1][0] for line in first_result])
        except (RuntimeError, KeyboardInterrupt, OSError, MemoryError) as e:
            logger.error(f"⚠️  OCR failed on PIL image: {e}")
            return ""

    def ocr_page_tiled(
        self,
        page: "PDFPage",
        dpi: int,
        tile_px: int | None = None,
        overlap: int | None = None,
    ) -> str:
        """OCR a page by splitting it into tiles to manage memory usage."""
        if tile_px is None:
            tile_px = self.config.tile_size
        overlap_value = overlap if overlap is not None else self.config.tile_overlap

        rect = page.rect
        s = dpi / 72.0
        full_w = int(rect.width * s)
        full_h = int(rect.height * s)

        texts: list[str] = []
        # number of tiles in each dimension
        if tile_px is None:
            tile_px = (
                self.config.tile_size or 600
            )  # fallback if CONFIG.tile_size is also None
        nx = max(1, math.ceil(full_w / tile_px))
        ny = max(1, math.ceil(full_h / tile_px))

        # tile size in page coordinates (points)
        tile_w_pts = tile_px / s
        tile_h_pts = tile_px / s
        ov_pts = overlap_value / s

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
                pix = None  # Initialize to avoid unbound variable

                try:
                    # GRAY, no alpha massively reduces memory (n=1 channel)
                    pix = page.get_pixmap(
                        matrix=fitz.Matrix(s, s),
                        colorspace=fitz.csGRAY,
                        alpha=False,
                        clip=clip,
                    )
                    # Avoid pix.samples → use compressed PNG bytes
                    png_bytes = pix.tobytes("png")
                    im = Image.open(BytesIO(png_bytes))
                    txt = self.ocr_pil_image(im)
                    if txt.strip():
                        texts.append(txt)
                    # Explicit cleanup
                    pix = None
                    im.close()
                    del im, png_bytes

                    # Force memory cleanup every 10 tiles
                    if (iy * nx + ix + 1) % 10 == 0:
                        self._cleanup_memory()

                except (MemoryError, RuntimeError):
                    # Handle both MemoryError and "could not create a primitive" RuntimeError
                    if pix:
                        pix = None
                    self._cleanup_memory()  # Force cleanup on error
                    # If a tile still fails (rare), try halving tile size once
                    if tile_px > 800:
                        return self.ocr_page_tiled(
                            page, dpi, tile_px=tile_px // 2, overlap=overlap_value
                        )
                    else:
                        continue
                except (OSError, RuntimeError, ValueError):
                    # OCR/image processing errors
                    continue

        return "\n".join(texts)

    def ocr_embedded_images(self, doc: "PDFDocument", page: "PDFPage") -> str:
        """Extract text from embedded images in PDF page."""
        out: list[str] = []
        try:
            imgs = page.get_images(full=True) or []
            for xref, *_ in imgs:
                try:
                    img = doc.extract_image(xref)
                    if img is not None and "image" in img:
                        im = Image.open(BytesIO(img["image"]))
                        out.append(self.ocr_pil_image(im))
                except (KeyError, OSError, ValueError, TypeError):
                    # Image extraction/processing errors
                    continue
        except (AttributeError, RuntimeError):
            # PDF processing errors
            pass
        filtered_texts = [text for text in out if text.strip()]
        return "\n".join(filtered_texts)

    def extract_from_image(self, path: str) -> str:
        """Extract text from image file using OCR with memory error handling."""
        try:
            with Image.open(path) as im:
                # Skip tiny images (likely test fixtures, icons, or UI elements)
                # Minimum 100x100 pixels required for meaningful OCR
                min_dimension = 100
                if im.width < min_dimension or im.height < min_dimension:
                    logger.info(
                        f"⏭️  Skipping tiny image {path} ({im.width}x{im.height}px) - below {min_dimension}x{min_dimension}px threshold"
                    )
                    return ""
                # Ultra conservative sizing for stability
                max_pixels = 256 * 256  # 0.065MP max - very small
                if im.width * im.height > max_pixels:
                    scale = (max_pixels / (im.width * im.height)) ** 0.5
                    new_w = max(64, int(im.width * scale))  # Don't go too small
                    new_h = max(64, int(im.height * scale))
                    im = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
                im = im.convert("RGB")
                arr = np.array(im)

            try:
                result = self.ocr.predict(arr)
                del arr  # Free array memory immediately
                if not result:
                    return ""
                first_result = result[0] if len(result) > 0 else None
                if not first_result:
                    return ""
                text_result = "\n".join([line[1][0] for line in first_result])
                del result, first_result  # Free result memory
                return text_result
            except (RuntimeError, KeyboardInterrupt, MemoryError) as e:
                logger.error(f"⚠️  OCR failed for {path}: {e}")
                self._cleanup_memory()
                return ""

        except (MemoryError, RuntimeError) as e:
            logger.warning(f"⚠️  Memory error for {path}, trying smaller size: {e}")
            try:
                # Try again with much smaller image
                with Image.open(path) as im:
                    im.thumbnail((128, 128), Image.Resampling.LANCZOS)
                    im = im.convert("RGB")
                    arr = np.array(im)
                try:
                    result = self.ocr.predict(arr)
                    del arr  # Free array memory immediately
                    if not result:
                        return ""
                    first_result = result[0] if len(result) > 0 else None
                    if not first_result:
                        return ""
                    text_result = "\n".join([line[1][0] for line in first_result])
                    del result, first_result  # Free result memory
                    return text_result
                except (RuntimeError, KeyboardInterrupt, MemoryError) as e:
                    logger.error(f"⚠️  OCR retry failed for {path}: {e}")
                    self._cleanup_memory()
                    return ""
            except (MemoryError, RuntimeError):
                logger.error(f"⚠️  Still out of memory for {path} even at reduced size")
                return ""
            except (OSError, ValueError, RuntimeError) as e:
                logger.error(
                    f"⚠️  Failed to process {path} even at reduced size: {type(e).__name__}: {e}"
                )
                return ""
        except (OSError, ValueError) as e:
            logger.error(
                f"⚠️  Image processing failed for {path}: {type(e).__name__}: {e}"
            )
            return ""
