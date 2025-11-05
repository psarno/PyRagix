"""Top-level orchestration for the ingestion pipeline.

The Python and .NET implementations share the same high-level sequence: prepare
environment guards, load existing FAISS/metadata artifacts, reconcile the
processed-ledger, scan and chunk new documents, then atomically persist FAISS,
SQLite, and BM25 resources. Comments below call out the ordering guarantees so
schema or retry changes stay mirrored across runtimes.
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path
from typing import cast

import config
from classes.ProcessingConfig import ProcessingConfig
from ingestion.environment import EnvironmentManager
from ingestion.file_filters import load_processed_files
from ingestion.file_scanner import Chunker, DocumentExtractor, FileScanner
from ingestion.metadata_store import build_bm25_index, load_metadata
from ingestion.models import IngestionContext
from ingestion.progress import ConsoleSpinnerProgress, IngestionStage
from ingestion.stale_cleaner import StaleDocumentCleaner
from types_models import MetadataDict
from utils.faiss_importer import faiss
from utils.faiss_types import FaissIndex

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.INGESTION_LOG_FILE, encoding="utf-8"),
    ],
)
# Mirror ingestion logs to stdout and to ingestion.log so long runs have durable diagnostics.


def load_existing_index(
    ctx: IngestionContext,
) -> tuple[FaissIndex | None, list[MetadataDict]]:
    """Load existing FAISS index and metadata if present on disk.

    The pair keeps the FAISS vectors and metadata rows in lock-step so the scan
    stage can append new chunks without re-ingesting everything.
    """
    cfg = ctx.config
    if cfg.index_path.exists() and cfg.db_path.exists():
        print("📂 Loading existing index and metadata...")
        index = cast(FaissIndex, faiss.read_index(str(cfg.index_path)))

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
            plural = "files" if count != 1 else "file"
            print(f"   • {count} {plural} skipped due to {reason}")

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
    verbose: bool = False,
) -> None:
    """Build (or update) the FAISS index for the supplied directory.

    High-level flow:
      1. Validate configuration, environment, and crash logs.
      2. Prime in-memory state from existing FAISS/SQLite/processed-ledger artifacts.
      3. Scan, extract, chunk, and embed documents via `FileScanner`.
      4. Persist updated indexes (FAISS + BM25) and report summary statistics.
    Each stage mirrors the .NET ingest command so behaviour stays cross-platform.
    """
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
        # Clear the previous crash report to avoid confusion with a successful retry.
        crash_log_path.unlink()
        print("🗑️  Cleared previous crash log")

    print(f"📁 FAISS exists: {cfg.index_path.exists()}")
    print(f"📝 Processed log exists: {cfg.processed_log.exists()}")
    print(f"⚙️ Index type: {config.INDEX_TYPE.upper()}")
    if config.INDEX_TYPE.lower() == "ivf":
        print(f"🎯 IVF settings: nlist={config.NLIST}, nprobe={config.NPROBE}")

    if config.GPU_ENABLED:
        # Share the GPU readiness check so users know immediately if FAISS will fall back to CPU.
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
    progress = ConsoleSpinnerProgress(enabled=not verbose)
    file_count = 0
    chunk_total = len(metadata)
    skipped_already_processed = 0
    skipped_problems = 0
    skip_reasons: dict[str, int] = {}
    total_candidates = 0

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
        # Detect whether the on-disk corpus diverged from the processed ledger before ingesting.
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
        if progress.enabled:
            # Spinner keeps the terminal responsive while the pipeline runs extraction/embedding.
            progress.start(
                stage=IngestionStage.SCANNING, message="Scanning documents..."
            )
            progress.update(
                files_completed=0,
                chunk_total=chunk_total,
                skipped_already_processed=skipped_already_processed,
                skipped_problems=skipped_problems,
            )
        # Scanner drives the full ingestion loop (extract -> chunk -> embed -> persist-to-FAISS).
        stats = scanner.scan(
            root_path,
            index,
            metadata,
            processed,
            recurse_subdirs=recurse_subdirs,
            progress=progress,
            verbose=verbose,
        )

        index = stats["index"]
        file_count = stats["file_count"]
        chunk_total = stats["chunk_total"]
        skipped_already_processed = stats["skipped_already_processed"]
        skipped_problems = stats["skipped_problems"]
        skip_reasons = stats["skip_reasons"]
        total_candidates = stats["candidate_total"]
        files_seen = file_count + skipped_already_processed + skipped_problems

        if progress.enabled:
            progress.update(
                stage=IngestionStage.PERSISTING,
                message="Persisting metadata and indexes...",
                files_completed=files_seen,
                total_files=total_candidates,
                chunk_total=chunk_total,
                skipped_already_processed=skipped_already_processed,
                skipped_problems=skipped_problems,
            )

        # BM25 persistence runs after FAISS/metadata so hybrid search sees consistent artifacts.
        build_bm25_index(metadata)

        if progress.enabled:
            # Final pulse ensures the spinner prints completion metrics before stopping.
            progress.update(
                stage=IngestionStage.COMPLETED,
                message=f"Ingestion complete. Files: {file_count}, chunks: {chunk_total}",
                files_completed=files_seen,
                total_files=total_candidates,
                chunk_total=chunk_total,
                skipped_already_processed=skipped_already_processed,
                skipped_problems=skipped_problems,
            )
    except Exception as exc:
        error_message = (
            f"❌ Fatal error during file processing: {type(exc).__name__}: {exc}"
        )
        if progress.enabled:
            progress.update(stage=IngestionStage.ERROR, message=error_message)
            progress.write_line(error_message)
        else:
            print(error_message)
        progress.stop()
        traceback.print_exc()
        sys.exit(1)
    finally:
        progress.stop()

    print(f"✅ File processing completed. Got {file_count} files, {chunk_total} chunks")

    if index is None or chunk_total == 0:
        print("❌ No text extracted. Nothing indexed.")
        sys.exit(2)

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
