from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from venv import logger

import psutil

import config


@dataclass
class ProcessingConfig:
    """Configuration for document processing and ingestion."""

    # Supported file extensions
    doc_extensions: set[str] = field(default_factory=set)

    # Text processing
    chunk_size: int = 1600  # characters
    chunk_overlap: int = 200  # characters
    embed_model: str = config.EMBED_MODEL

    # File paths
    index_path: Path = field(default_factory=lambda: Path("local_faiss.index"))
    db_path: Path = field(default_factory=lambda: Path("documents.db"))
    processed_log: Path = field(default_factory=lambda: Path("processed_files.txt"))
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
    skip_files: set[str] = field(
        default_factory=set
    )  # Hard-coded list of files to skip


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
            self.max_pixels = (
                200_000  # ~0.2 MP per render (reduced further for stability)
            )
            self.tile_size = 400
        else:
            # Low-memory systems
            self.max_pixels = 400_000  # ~0.4 MP per render
            self.tile_size = 600

        logger.info(
            f"🖥️  Detected {total_ram_gb:.1f}GB RAM - using MAX_PIXELS={self.max_pixels:,}, TILE_SIZE={self.tile_size}"
        )
