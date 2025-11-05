"""Directory traversal and chunk persistence pipeline.

This module stitches together extraction, chunking, embedding, and FAISS/BM25
persistence.  Its control flow mirrors the .NET `FileScanner` so both runtimes
apply identical retry policies, skip rules, and progress reporting while
writing metadata and updating the processed ledger.
"""

from __future__ import annotations

import logging
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import config
from ingestion.environment import EnvironmentManager
from ingestion.file_filters import calculate_file_hash, should_skip_file
from ingestion.metadata_store import ChunkRecord, insert_chunk_records
from ingestion.models import (
    EmbeddingModel,
    IngestionContext,
    OCRProcessorProtocol,
    ProcessingResult,
    ProcessingStats,
)
from ingestion.progress import ConsoleSpinnerProgress, IngestionStage
from ingestion.text_processing import clean_text, chunk_text, extract_text
from types_models import MetadataDict
from classes.ProcessingConfig import ProcessingConfig
from utils.faiss_types import FaissIndex, ensure_nprobe


def _get_torch():
    import torch

    return torch


logger = logging.getLogger(__name__)


class DocumentExtractor:
    """High-level text extraction facade."""

    def __init__(self, cfg: ProcessingConfig, ocr: OCRProcessorProtocol) -> None:
        super().__init__()
        self._cfg = cfg
        self._ocr = ocr

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _extract_with_retry(self, path: str) -> str:
        return extract_text(path, self._ocr, self._cfg)

    def extract(self, path: str) -> str:
        try:
            raw_text = self._extract_with_retry(path)
        except Exception as exc:
            raise RuntimeError(
                f"Extraction failed for {Path(path).name}: {exc}"
            ) from exc
        return clean_text(raw_text)


class Chunker:
    """Chunk text according to the configured strategy."""

    def __init__(self, cfg: ProcessingConfig, embedder: EmbeddingModel) -> None:
        super().__init__()
        self._cfg = cfg
        self._embedder = embedder

    def chunk(self, text: str) -> list[str]:
        return chunk_text(text, self._cfg, embedder=self._embedder)


class FileScanner:
    """Walks directories and processes documents into the FAISS index."""

    def __init__(
        self,
        ctx: IngestionContext,
        *,
        extractor: DocumentExtractor,
        chunker: Chunker,
        env: EnvironmentManager,
    ) -> None:
        super().__init__()
        self._ctx = ctx
        self._extractor = extractor
        self._chunker = chunker
        self._env = env
        self._faiss_manager = ctx.faiss_manager

    @staticmethod
    def _normalize_skip_reason(reason: str) -> str:
        """Collapse specific skip messages into grouped summary categories."""
        base = reason.split(":", 1)[0].strip().rstrip(".")
        mapping = {
            "unsupported file type": "incompatible file type",
            "file type not in filter": "filtered file type",
        }
        return mapping.get(base, base)

    def _emit_message(
        self,
        message: str,
        progress: ConsoleSpinnerProgress | None,
    ) -> None:
        """Write a progress-aware message to stdout."""
        if progress and progress.enabled:
            progress.write_line(message)
        else:
            print(message)

    def _encode_embeddings(
        self,
        embedder: EmbeddingModel,
        chunks: list[str],
        *,
        batch_size: int,
    ) -> np.ndarray:
        """Encode chunks to np.ndarray using torch autocast to keep GPU RAM in check."""
        torch = _get_torch()
        with (
            torch.inference_mode(),
            torch.autocast(
                "cuda",
                dtype=torch.float16,
                enabled=torch.cuda.is_available(),
            ),
        ):
            embs_raw = embedder.encode(
                chunks,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        embs = np.asarray(embs_raw, dtype=np.float32)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return embs

    def _embed_with_fallback(self, path: str, chunks: list[str]) -> np.ndarray:
        """Encode chunks with batch-size fallback when CUDA reports OOM."""
        embedder = self._ctx.embedder
        torch = _get_torch()
        try:
            return self._encode_embeddings(
                embedder,
                chunks,
                batch_size=config.BATCH_SIZE,
            )
        except torch.OutOfMemoryError as exc:
            logger.error("⚠️  CUDA out of memory during embedding for %s: %s", path, exc)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            smaller_batch = max(1, config.BATCH_SIZE // config.BATCH_SIZE_RETRY_DIVISOR)
            logger.info(
                "🔄 Retrying embedding with smaller batch size: %s", smaller_batch
            )
            try:
                return self._encode_embeddings(
                    embedder,
                    chunks,
                    batch_size=smaller_batch,
                )
            except Exception as retry_exc:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"Embedding failed for {Path(path).name} after reducing batch size: {retry_exc}"
                ) from retry_exc
        except Exception as exc:
            raise RuntimeError(
                f"Embedding failed for {Path(path).name}: {exc}"
            ) from exc

    @retry(
        retry=retry_if_exception_type((RuntimeError, ValueError, OSError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _embed_chunks_with_retry(self, path: str, chunks: list[str]) -> np.ndarray:
        return self._embed_with_fallback(path, chunks)

    @retry(
        retry=retry_if_exception_type((OSError, RuntimeError, ValueError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _save_index_with_retry(self, index: FaissIndex) -> None:
        try:
            self._faiss_manager.save(index, self._ctx.config.index_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to persist FAISS index to {self._ctx.config.index_path}: {exc}"
            ) from exc

    def process_file(
        self,
        path: str,
        index: FaissIndex | None,
        metadata: list[MetadataDict],
        *,
        progress: ConsoleSpinnerProgress | None = None,
        verbose: bool = False,
    ) -> ProcessingResult:
        """Extract, chunk, embed, and persist a single document."""
        start_time = time.perf_counter()
        text = self._extractor.extract(path)
        extract_time = time.perf_counter() - start_time

        chunk_start = time.perf_counter()
        chunks = self._chunker.chunk(text)
        chunk_time = time.perf_counter() - chunk_start

        if not chunks:
            if verbose:
                self._emit_message(
                    f"[verbose] {os.path.basename(path)} produced no chunks (extract {extract_time:.2f}s).",
                    progress,
                )
                return {"index": index, "chunk_count": 0}

        embed_start = time.perf_counter()
        embs = self._embed_chunks_with_retry(path, chunks)
        embed_time = time.perf_counter() - embed_start

        index_start = time.perf_counter()

        if index is None:
            dim = embs.shape[1]
            index, actual_type = self._faiss_manager.create(dim)

            if actual_type == "ivf":
                with self._env.memory_guard():
                    training_success = self._faiss_manager.train(index, embs)
                if not training_success:
                    logger.info("🔄 Creating flat index as fallback...")
                    index, _ = self._faiss_manager.create(dim, index_type="flat")
                else:
                    ivf_index = ensure_nprobe(index, context="FaissManager.create")
                    ivf_index.nprobe = config.NPROBE

        assert index is not None

        if config.INDEX_TYPE.lower() == "ivf" and index.is_trained:
            with self._env.memory_guard():
                index = self._faiss_manager.maybe_retrain(index, embs)
        else:
            index.add(embs)

        index_time = time.perf_counter() - index_start

        file_hash = calculate_file_hash(path)
        created_at = datetime.now().isoformat()
        file_type = Path(path).suffix.lstrip(".").lower() or "unknown"
        total_chunks = len(chunks)

        chunk_records: list[ChunkRecord] = []
        metadata_entries: list[MetadataDict] = []
        # Mirror chunk payloads into SQLite records and in-memory metadata for downstream retrieval.
        for i, chunk in enumerate(chunks):
            chunk_records.append(
                {
                    "source": path,
                    "chunk_index": i,
                    "text": chunk,
                    "file_hash": file_hash,
                    "created_at": created_at,
                    "file_type": file_type,
                    "total_chunks": total_chunks,
                }
            )
            metadata_entries.append(
                MetadataDict(
                    source=path,
                    chunk_index=i,
                    text=chunk,
                    file_type=file_type,
                    total_chunks=total_chunks,
                )
            )

        persist_start = time.perf_counter()
        metadata.extend(metadata_entries)
        insert_chunk_records(self._ctx.config.db_path, chunk_records)

        self._save_index_with_retry(index)
        self._env.cleanup()
        persist_time = time.perf_counter() - persist_start

        if verbose:
            self._emit_message(
                (
                    f"[verbose] {os.path.basename(path)} "
                    f"extract={extract_time:.2f}s "
                    f"chunk={chunk_time:.2f}s "
                    f"embed={embed_time:.2f}s "
                    f"index={index_time:.2f}s "
                    f"persist={persist_time:.2f}s "
                    f"chunks={total_chunks}"
                ),
                progress,
            )

        return {"index": index, "chunk_count": len(chunks)}

    def _collect_candidate_files(
        self,
        root_path: Path,
        effective_extensions: set[str],
        excluded_dirs: set[str],
        *,
        recurse_subdirs: bool,
    ) -> list[str]:
        """Return an ordered list of file paths that match the configured extensions."""
        candidates: list[str] = []

        if recurse_subdirs:
            for dirpath, subdirs, filenames in os.walk(root_path):
                subdirs[:] = [d for d in subdirs if d not in excluded_dirs]
                for fname in filenames:
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in effective_extensions:
                        candidates.append(os.path.join(dirpath, fname))
        else:
            for fname in os.listdir(root_path):
                candidate_path = os.path.join(root_path, fname)
                if not os.path.isfile(candidate_path):
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext in effective_extensions:
                    candidates.append(candidate_path)

        return candidates

    def scan(
        self,
        root_path: Path,
        index: FaissIndex | None,
        metadata: list[MetadataDict],
        processed: set[str],
        *,
        recurse_subdirs: bool = True,
        progress: ConsoleSpinnerProgress | None = None,
        verbose: bool = False,
    ) -> ProcessingStats:
        """Walk the filesystem and process supported files."""
        cfg = self._ctx.config
        file_count = 0
        chunk_total = len(metadata)
        skipped_already_processed = 0
        skipped_problems = 0
        skip_reasons: dict[str, int] = {}
        effective_extensions = cfg.get_effective_extensions()
        progress_obj = progress if progress and progress.enabled else None

        excluded_dirs = {
            ".venv",
            "venv",
            ".git",
            ".hg",
            ".svn",
            "node_modules",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".tox",
            "dist",
            "build",
            ".eggs",
        }

        try:
            candidate_files = self._collect_candidate_files(
                root_path,
                effective_extensions,
                excluded_dirs,
                recurse_subdirs=recurse_subdirs,
            )
        except OSError:
            message = f"❌ Cannot access directory: {root_path}"
            if progress_obj:
                progress_obj.update(
                    stage=IngestionStage.ERROR,
                    message=message,
                    files_completed=0,
                    total_files=0,
                    chunk_total=chunk_total,
                )
                progress_obj.write_line(message)
            else:
                print(message)
            return {
                "index": index,
                "file_count": 0,
                "chunk_total": chunk_total,
                "skipped_already_processed": 0,
                "skipped_problems": 0,
                "skip_reasons": {},
                "candidate_total": 0,
            }

        total_candidates = len(candidate_files)
        files_seen = 0

        if progress_obj:
            discovery_message = (
                f"Preparing worklist ({total_candidates} candidate files)..."
                if total_candidates
                else "No matching files discovered."
            )
            progress_obj.start(
                stage=IngestionStage.DISCOVERY,
                message=discovery_message,
                total_files=total_candidates,
            )
            progress_obj.update(
                files_completed=files_seen,
                chunk_total=chunk_total,
                skipped_already_processed=skipped_already_processed,
                skipped_problems=skipped_problems,
            )

        for path in candidate_files:
            ext = os.path.splitext(path)[1].lower()
            skip, reason = should_skip_file(path, ext, processed, cfg)
            filename = os.path.basename(path)

            if skip:
                if reason == "already processed":
                    skipped_already_processed += 1
                else:
                    skipped_problems += 1
                    normalized_reason = self._normalize_skip_reason(reason)
                    # Track unique skip buckets so summary output highlights repeated issues.
                    skip_reasons[normalized_reason] = (
                        skip_reasons.get(normalized_reason, 0) + 1
                    )
                files_seen += 1
                if progress_obj:
                    label = (
                        "already processed"
                        if reason == "already processed"
                        else self._normalize_skip_reason(reason)
                    )
                    progress_obj.update(
                        stage=IngestionStage.FILE_SKIPPED,
                        message=f"Skipped {filename} ({label})",
                        files_completed=files_seen,
                        chunk_total=chunk_total,
                        skipped_already_processed=skipped_already_processed,
                        skipped_problems=skipped_problems,
                    )
                if verbose:
                    label = (
                        "already processed"
                        if reason == "already processed"
                        else self._normalize_skip_reason(reason)
                    )
                    self._emit_message(
                        f"[verbose] Skipped {filename} ({label})", progress_obj
                    )
                continue

            if progress_obj:
                progress_obj.update(
                    stage=IngestionStage.FILE_STARTED,
                    message=f"Processing {filename}...",
                    files_completed=files_seen,
                    chunk_total=chunk_total,
                    skipped_already_processed=skipped_already_processed,
                    skipped_problems=skipped_problems,
                )
            else:
                print(f"Processing: {path}")
            if verbose:
                self._emit_message(f"[verbose] Starting {filename}", progress_obj)

            try:
                file_count += 1
                with self._env.memory_guard():
                    result = self.process_file(
                        path,
                        index,
                        metadata,
                        progress=progress_obj,
                        verbose=verbose,
                    )
                index = result["index"]
                chunk_count = result["chunk_count"]
                chunk_total += chunk_count

                if cfg.processed_log:
                    file_hash = calculate_file_hash(path)
                    if file_hash:
                        # Append file checksum ledger so subsequent runs can skip unchanged sources.
                        with open(cfg.processed_log, "a", encoding="utf-8") as handle:
                            _ = handle.write(f"{file_hash}|{filename}\n")

                files_seen += 1
                if progress_obj:
                    progress_obj.update(
                        stage=IngestionStage.FILE_COMPLETED,
                        message=f"Completed {filename}",
                        files_completed=files_seen,
                        chunk_total=chunk_total,
                        skipped_already_processed=skipped_already_processed,
                        skipped_problems=skipped_problems,
                    )

                if verbose:
                    self._emit_message(
                        f"[verbose] Completed {filename} with {chunk_count} chunks",
                        progress_obj,
                    )

                if cfg.top_print_every and file_count % cfg.top_print_every == 0:
                    status_line = (
                        "⚙️ Processed "
                        + f"{file_count} files | total chunks: {chunk_total} | "
                        + f"already done: {skipped_already_processed} | problems: {skipped_problems}"
                    )
                    if progress_obj:
                        progress_obj.write_line(status_line)
                    else:
                        print(status_line)

            except Exception as exc:
                skipped_problems += 1
                files_seen += 1
                error_type = type(exc).__name__
                error_msg = str(exc)
                friendly_reason = self._normalize_skip_reason(error_type)
                message = f"⚠️ Failed: {filename} - {error_type}: {error_msg[:100]}"

                if progress_obj:
                    progress_obj.update(
                        stage=IngestionStage.ERROR,
                        message=message,
                        files_completed=files_seen,
                        chunk_total=chunk_total,
                        skipped_already_processed=skipped_already_processed,
                        skipped_problems=skipped_problems,
                    )
                    progress_obj.write_line(message)
                else:
                    print(message)

                with open(config.CRASH_LOG_FILE, "a", encoding="utf-8") as handle:
                    _ = handle.write(f"\n{'=' * 60}\n")
                    _ = handle.write(f"CRASHED FILE: {path}\n")
                    _ = handle.write(f"ERROR TYPE: {error_type}\n")
                    _ = handle.write(f"ERROR: {error_msg}\n")
                    _ = handle.write("TRACEBACK:\n")
                    _ = handle.write(traceback.format_exc())
                    _ = handle.write(f"\n{'=' * 60}\n")

                # Aggregate fatal failure reasons so callers see problem hotspots in summary output.
                skip_reasons[friendly_reason] = skip_reasons.get(friendly_reason, 0) + 1
                self._env.cleanup()

        return {
            "index": index,
            "file_count": file_count,
            "chunk_total": chunk_total,
            "skipped_already_processed": skipped_already_processed,
            "skipped_problems": skipped_problems,
            "skip_reasons": skip_reasons,
            "candidate_total": total_candidates,
        }


__all__ = ["DocumentExtractor", "Chunker", "FileScanner"]
