from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import config
from classes.ProcessingConfig import ProcessingConfig
from ingestion.environment import EnvironmentManager
from ingestion.file_filters import load_processed_files
from ingestion.file_scanner import Chunker, DocumentExtractor, FileScanner
from ingestion.metadata_store import build_bm25_index, load_metadata
from ingestion.models import IngestionContext
from ingestion.stale_cleaner import StaleDocumentCleaner
from types_models import MetadataDict

if TYPE_CHECKING:
    import faiss

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.INGESTION_LOG_FILE, encoding="utf-8"),
    ],
)


def load_existing_index(
    ctx: IngestionContext,
) -> tuple[faiss.Index | None, list[MetadataDict]]:
    """Load existing FAISS index and metadata if present on disk."""
    import faiss

    cfg = ctx.config
    if cfg.index_path.exists() and cfg.db_path.exists():
        print("📂 Loading existing index and metadata...")
        index = faiss.read_index(str(cfg.index_path))

        metadata = load_metadata(cfg.db_path)

        index = ctx.faiss_manager.prepare_loaded_index(index)

        print(
            f"   Loaded {index.ntotal} existing chunks from {len(set(m.source for m in metadata))} files"
        )
        return index, metadata
    return None, []


def print_summary(
    file_count: int,
    chunk_total: int,
    skipped_already_processed: int,
    skipped_problems: int,
    skip_reasons: dict[str, int],
    cfg: ProcessingConfig,
) -> None:
    """Emit final processing statistics."""
    print("-------------------------------------------------")
    print(f"✅ Done. Files processed: {file_count} | Chunks: {chunk_total}")
    print(f"📋 Already processed: {skipped_already_processed}")
    print(f"⚠️  Problem files: {skipped_problems}")

    if skip_reasons:
        print("📊 Skip breakdown:")
        for reason, count in sorted(skip_reasons.items()):
            print(f"   • {reason}: {count}")

    print(f"📝  Index: {cfg.index_path}")
    print(f"📝 Database: {cfg.db_path}")
    if config.INDEX_TYPE.lower() == "ivf":
        print(f"🎯 Index type: IVF (nlist={config.NLIST}, nprobe={config.NPROBE})")
    else:
        print("🎯 Index type: Flat")
    print("   (Re-run this script after adding new documents.)")


def build_index(
    root_folder: str,
    ctx: IngestionContext,
    *,
    env: EnvironmentManager | None = None,
    fresh_start: bool = False,
    recurse_subdirs: bool = True,
    filetypes: str | None = None,
) -> None:
    """Build (or update) the FAISS index for the supplied directory."""
    cfg = ctx.config
    root_path = Path(root_folder)
    env_manager = env or EnvironmentManager()
    extractor = DocumentExtractor(cfg, ctx.ocr)
    chunker = Chunker(cfg, ctx.embedder)
    scanner = FileScanner(ctx, extractor=extractor, chunker=chunker, env=env_manager)

    if not root_path.is_dir():
        print(f"❌ Folder not found: {root_path}")
        sys.exit(1)

    crash_log_path = Path(config.CRASH_LOG_FILE)
    if crash_log_path.exists():
        crash_log_path.unlink()
        print("🗑️  Cleared previous crash log")

    print(f"📁 FAISS exists: {cfg.index_path.exists()}")
    print(f"📝 Processed log exists: {cfg.processed_log.exists()}")
    print(f"⚙️ Index type: {config.INDEX_TYPE.upper()}")
    if config.INDEX_TYPE.lower() == "ivf":
        print(f"🎯 IVF settings: nlist={config.NLIST}, nprobe={config.NPROBE}")

    if config.GPU_ENABLED:
        gpu_status = "Active" if ctx.faiss_manager.gpu_ready else "Failed"
        print(f"🎮 FAISS GPU acceleration: {gpu_status} (device {config.GPU_DEVICE})")
        if not ctx.faiss_manager.gpu_ready:
            print(f"   💡 {ctx.faiss_manager.status}")

    pickle_file = root_path / "documents.pkl"
    if pickle_file.exists():
        pickle_file.unlink()
        print("🗑️  Removed obsolete documents.pkl file")

    if fresh_start:
        print("🆕 Fresh start requested - clearing all existing files")
        if cfg.index_path.exists():
            cfg.index_path.unlink()
            print("   Removed existing FAISS index")
        if cfg.db_path.exists():
            cfg.db_path.unlink()
            print("   Removed existing metadata database")
        if cfg.processed_log.exists():
            cfg.processed_log.unlink()
            print("   Removed processed files log")
    elif cfg.processed_log.exists() and cfg.index_path.exists():
        print("ℹ️ Resuming - keeping existing index")
    elif cfg.index_path.exists():
        cfg.index_path.unlink()
        if cfg.db_path.exists():
            cfg.db_path.unlink()
        print("ℹ🧹 Cleared existing index for fresh run")

    if filetypes:
        try:
            cfg.set_allowed_extensions(filetypes)
        except ValueError as exc:
            print(f"❌ Invalid file types: {exc}")
            sys.exit(1)

    effective_extensions = cfg.get_effective_extensions()
    printable = ", ".join(sorted(ext.lstrip(".") for ext in effective_extensions))
    if cfg.allowed_extensions is not None:
        print(f"📄 Processing file types: {printable}")
    else:
        print(f"📄 Processing all supported file types: {printable}")

    print(f"🔎 Scanning: {root_path}")

    index, metadata = load_existing_index(ctx)
    processed = load_processed_files(cfg)

    if not fresh_start and processed:
        current_files: list[Path] = []
        for ext in effective_extensions:
            glob_pattern = f"*{ext}"
            if recurse_subdirs:
                current_files.extend(root_path.rglob(glob_pattern))
            else:
                current_files.extend(root_path.glob(glob_pattern))

        filtered_files = [
            file_path
            for file_path in current_files
            if not any(file_path.match(pattern) for pattern in config.SKIP_FILES)
        ]

        cleaner = StaleDocumentCleaner(cfg)
        fresh_start_requested, processed = cleaner.check_and_handle_stale_documents(
            processed, filtered_files
        )

        if fresh_start_requested:
            fresh_start = True
            index = None
            metadata.clear()

    if recurse_subdirs:
        print("🔄 Starting file scan and processing (including subdirectories)...")
    else:
        print("🔄 Starting file scan and processing (root folder only)...")

    try:
        stats = scanner.scan(
            root_path,
            index,
            metadata,
            processed,
            recurse_subdirs=recurse_subdirs,
        )
    except Exception as exc:
        print(f"❌ Fatal error during file processing: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)

    index = stats["index"]
    file_count = stats["file_count"]
    chunk_total = stats["chunk_total"]
    skipped_already_processed = stats["skipped_already_processed"]
    skipped_problems = stats["skipped_problems"]
    skip_reasons = stats["skip_reasons"]

    print(f"✅ File processing completed. Got {file_count} files, {chunk_total} chunks")

    if index is None or chunk_total == 0:
        print("❌ No text extracted. Nothing indexed.")
        sys.exit(2)

    build_bm25_index(metadata)

    print("📊 Generating summary...")
    print_summary(
        file_count,
        chunk_total,
        skipped_already_processed,
        skipped_problems,
        skip_reasons,
        cfg,
    )
    print("🎉 Script completed successfully!")


__all__ = [
    "build_index",
    "load_existing_index",
    "print_summary",
]
