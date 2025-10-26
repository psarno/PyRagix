from __future__ import annotations

import logging
import math
from io import BytesIO
from pathlib import Path
from typing import Any, TYPE_CHECKING

from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from PIL import Image

import config
from classes.ProcessingConfig import ProcessingConfig
from ingestion.models import EmbeddingModel, PDFPage

if TYPE_CHECKING:
    from classes.OCRProcessor import OCRProcessor

logger = logging.getLogger(__name__)


def clean_text(value: str) -> str:
    """Normalize whitespace to keep downstream chunkers predictable."""
    return " ".join(value.split())


def chunk_text(
    text: str,
    cfg: ProcessingConfig,
    *,
    size: int | None = None,
    overlap: int | None = None,
    embedder: EmbeddingModel | None = None,
) -> list[str]:
    """Chunk text using semantic or fixed-size strategy."""
    if size is None:
        size = cfg.chunk_size
    if overlap is None:
        overlap = cfg.chunk_overlap

    text = text.strip()
    if not text:
        return []

    if config.ENABLE_SEMANTIC_CHUNKING:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            if embedder is None:
                logger.warning(
                    "⚠️ Semantic chunking enabled but no embedder supplied; using fixed-size"
                )
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config.SEMANTIC_CHUNK_MAX_SIZE,
                    chunk_overlap=config.SEMANTIC_CHUNK_OVERLAP,
                    length_function=len,
                    separators=[
                        "\n\n",
                        "\n",
                        ". ",
                        "? ",
                        "! ",
                        "; ",
                        ", ",
                        " ",
                        "",
                    ],
                )
                return splitter.split_text(text)
        except ImportError:
            logger.warning(
                "⚠️ langchain-text-splitters missing. Install via `uv add langchain-text-splitters` for semantic chunking."
            )
        except Exception as exc:
            logger.warning(f"⚠️ Semantic chunking failed: {exc}. Using fixed-size.")

    chunks: list[str] = []
    step = max(1, size - overlap)
    for start in range(0, len(text), step):
        chunks.append(text[start : start + size])
    return chunks


def extract_text(path: str, ocr: OCRProcessor, cfg: ProcessingConfig) -> str:
    """Extract text from PDF, HTML, or image sources."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return _extract_from_pdf(path, ocr, cfg)
    if ext in {".html", ".htm"}:
        return _html_to_text(path)
    return ocr.extract_from_image(path)


def _html_to_text(path: str) -> str:
    parser = "lxml"
    try:
        import lxml  # type: ignore[import-untyped]

        _ = lxml
    except ImportError:
        parser = "html.parser"

    with open(path, "r", encoding="utf-8", errors="ignore") as resource:
        soup = BeautifulSoup(resource, parser)

    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    return soup.get_text(separator="\n")


def _safe_dpi_for_page(
    page: PDFPage,
    cfg: ProcessingConfig,
    *,
    max_pixels: int | None = None,
    max_side: int | None = None,
    base_dpi: int | None = None,
) -> int:
    if max_pixels is None:
        max_pixels = cfg.max_pixels
    if max_side is None:
        max_side = cfg.max_side
    if base_dpi is None:
        base_dpi = config.BASE_DPI

    rect = page.rect
    if rect.width == 0 or rect.height == 0:
        return 96

    def px_for(dpi: int) -> tuple[float, float]:
        scale = dpi / 72.0
        return rect.width * scale, rect.height * scale

    width_px, height_px = px_for(base_dpi)
    scale = 1.0

    if max_pixels is not None and width_px * height_px > max_pixels:
        scale *= math.sqrt(max_pixels / (width_px * height_px))
    if width_px * scale > max_side:
        scale *= max_side / (width_px * scale)
    if height_px * scale > max_side:
        scale *= max_side / (height_px * scale)

    return max(72, int(base_dpi * scale))


def _pdf_page_text_or_ocr(
    page: Any,
    ocr: OCRProcessor,
    cfg: ProcessingConfig,
    *,
    doc: Any | None = None,
) -> str:
    text = page.get_text("text") or ""
    if len(text.strip()) > 20:
        return text

    if doc is not None:
        embedded = ocr.ocr_embedded_images(doc, page)
        if embedded.strip():
            return embedded

    dpi = _safe_dpi_for_page(
        page,
        cfg,
        max_pixels=cfg.max_pixels,
        max_side=cfg.max_side,
        base_dpi=config.BASE_DPI,
    )
    scale = dpi / 72.0
    width_px = int(page.rect.width * scale)
    height_px = int(page.rect.height * scale)

    if (
        cfg.tile_size is not None
        and width_px <= cfg.tile_size
        and height_px <= cfg.tile_size
    ):
        try:
            pix = page.get_pixmap(
                matrix=fitz.Matrix(scale, scale),
                colorspace=fitz.csGRAY,
                alpha=False,
            )
            png_bytes = pix.getPNGdata()
            image = Image.open(BytesIO(png_bytes))
            return ocr.ocr_pil_image(image)
        except MemoryError:
            return ocr.ocr_page_tiled(page, dpi)
        except (OSError, RuntimeError, ValueError):
            return ""

    return ocr.ocr_page_tiled(page, dpi)


def _extract_from_pdf(path: str, ocr: OCRProcessor, cfg: ProcessingConfig) -> str:
    pages: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            try:
                pages.append(_pdf_page_text_or_ocr(page, ocr, cfg, doc=doc))
            except (RuntimeError, MemoryError, OSError) as exc:
                logger.error(f"⚠️ Error processing PDF page: {type(exc).__name__}: {exc}")
                logger.debug("Full traceback:", exc_info=True)
                continue
    return "\n".join(pages)


__all__ = [
    "chunk_text",
    "clean_text",
    "extract_text",
]
