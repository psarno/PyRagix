# ======================================
# OCR Processing Module
# Handles all OCR operations with PaddleOCR
# ======================================

import gc
import logging
import math
from io import BytesIO
from typing import Any, Optional

import fitz  # PyMuPDF
import numpy as np
import paddle
from PIL import Image
from paddleocr import PaddleOCR

import os, sys
base = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
for sub in ["cudnn", "cublas", "cufft", "curand", "cusolver", "cusparse", "cuda_runtime"]:
    bin_path = os.path.join(base, sub, "bin")
    if os.path.isdir(bin_path):
        os.add_dll_directory(bin_path)

# Set up logger
logger = logging.getLogger(__name__)

class OCRProcessor:
    """Handles all OCR operations with PaddleOCR."""

    def __init__(self, config):
        self.config = config
        self.ocr = self._init_ocr()

    def _init_ocr(self) -> PaddleOCR:
        """Initialize PaddleOCR with appropriate settings."""
        # Suppress PaddleOCR warnings
        logging.getLogger("paddleocr").setLevel(logging.ERROR)

        # Force CPU; angle classifier off (we handle orientation OK in most docs)
        ocr = PaddleOCR(lang="en", use_angle_cls=False, use_gpu=False)

        try:
            dev = getattr(
                getattr(paddle, "device", None), "get_device", lambda: "cpu"
            )()
            logger.info(f"ℹ️ PaddlePaddle: {paddle.__version__} | Device: {dev}")
        except (AttributeError, TypeError):
            logger.warning("⚠️ Could not print Paddle version/device.")
        return ocr

    def _cleanup_memory(self) -> None:
        """Force garbage collection to free up memory."""
        gc.collect()
        # Clear Paddle's memory cache if available
        try:
            paddle.device.cuda.empty_cache()
        except (AttributeError, RuntimeError):
            # Not using CUDA or method not available
            pass

    def ocr_pil_image(self, pil_img: Any) -> str:
        """Extract text from PIL image using OCR."""
        try:
            # Convert and process
            rgb_img = pil_img.convert("RGB")
            arr = np.array(rgb_img)
            
            # Clear RGB image from memory
            if rgb_img is not pil_img:
                rgb_img.close()
            
            result = self.ocr.ocr(arr, cls=self.config.use_ocr_cls)
            
            # Clear array from memory
            del arr
            
            if not result or not isinstance(result, list) or not result:
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
        page: Any,
        dpi: int,
        tile_px: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> str:
        """OCR a page by splitting it into tiles to manage memory usage."""
        if tile_px is None:
            tile_px = self.config.tile_size
        if overlap is None:
            overlap = self.config.tile_overlap

        rect = page.rect
        s = dpi / 72.0
        full_w = int(rect.width * s)
        full_h = int(rect.height * s)

        texts = []
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
        ov_pts = overlap / s  # type: ignore

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
                    pix = page.get_pixmap(  # type: ignore[attr-defined]
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
                        
                except (MemoryError, RuntimeError) as e:
                    # Handle both MemoryError and "could not create a primitive" RuntimeError
                    if pix:
                        pix = None
                    self._cleanup_memory()  # Force cleanup on error
                    # If a tile still fails (rare), try halving tile size once
                    if tile_px is not None and tile_px > 800:
                        return self.ocr_page_tiled(
                            page, dpi, tile_px=tile_px // 2, overlap=overlap
                        )
                    else:
                        continue
                except (OSError, RuntimeError, ValueError):
                    # OCR/image processing errors
                    continue

        return "\n".join(texts)

    def ocr_embedded_images(self, doc: Any, page: Any) -> str:
        """Extract text from embedded images in PDF page."""
        out = []
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
        return "\n".join([t for t in out if t.strip()])

    def extract_from_image(self, path: str) -> str:
        """Extract text from image file using OCR with memory error handling."""
        try:
            with Image.open(path) as im:
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
                result = self.ocr.ocr(arr, cls=False)  # cls=False to save memory
                del arr  # Free array memory immediately
                if not result or not isinstance(result, list) or not result:
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
                    result = self.ocr.ocr(arr, cls=False)
                    del arr  # Free array memory immediately
                    if not result or not isinstance(result, list) or not result:
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
