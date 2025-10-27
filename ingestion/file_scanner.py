import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Iterable, TYPE_CHECKING

import numpy as np
import torch

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
from ingestion.text_processing import clean_text, chunk_text, extract_text
from types_models import MetadataDict
from classes.ProcessingConfig import ProcessingConfig

if TYPE_CHECKING:
    import faiss

logger = logging.getLogger(__name__)


class DocumentExtractor:
    """High-level text extraction facade."""

    def __init__(self, cfg: ProcessingConfig, ocr: OCRProcessorProtocol) -> None:
        self._cfg = cfg
        self._ocr = ocr

    def extract(self, path: str) -> str:
        raw_text = extract_text(path, self._ocr, self._cfg)
        return clean_text(raw_text)


class Chunker:
    """Chunk text according to the configured strategy."""

    def __init__(self, cfg: ProcessingConfig, embedder: EmbeddingModel) -> None:
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
        self._ctx = ctx
        self._extractor = extractor
        self._chunker = chunker
        self._env = env
        self._faiss_manager = ctx.faiss_manager

    def process_file(
        self,
        path: str,
        index: faiss.Index | None,
        metadata: list[MetadataDict],
    ) -> ProcessingResult:
        """Extract, chunk, embed, and persist a single document."""
        text = self._extractor.extract(path)
        chunks = self._chunker.chunk(text)

        if not chunks:
            return {"index": index, "chunk_count": 0}

        embs: np.ndarray | None = None
        embedder = self._ctx.embedder
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
                embs = np.asarray(embs_raw, dtype=np.float32).astype(np.float32)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except torch.OutOfMemoryError as exc:
            logger.error(
                "⚠️  CUDA out of memory during embedding for %s: %s", path, exc
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                smaller_batch = max(
                    1, config.BATCH_SIZE // config.BATCH_SIZE_RETRY_DIVISOR
                )
                logger.info(
                    "🔄 Retrying embedding with smaller batch size: %s", smaller_batch
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
                    embs = np.asarray(embs_raw, dtype=np.float32).astype(np.float32)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except Exception as retry_exc:
                logger.error(
                    "⚠️  Failed to embed even with smaller batch for %s: %s",
                    path,
                    retry_exc,
                )
                return {"index": index, "chunk_count": 0}

        assert embs is not None

        if index is None:
            dim = embs.shape[1]
            index, actual_type = self._faiss_manager.create(dim)

            if actual_type == "ivf":
                with self._env.memory_guard():
                    training_success = self._faiss_manager.train(index, embs)
                if not training_success:
                    logger.info("🔄 Creating flat index as fallback...")
                    index, _ = self._faiss_manager.create(dim, index_type="flat")
                elif hasattr(index, "nprobe"):
                    index.nprobe = config.NPROBE

        assert index is not None

        if (
            config.INDEX_TYPE.lower() == "ivf"
            and hasattr(index, "is_trained")
            and getattr(index, "is_trained", False)
        ):
            with self._env.memory_guard():
                index = self._faiss_manager.maybe_retrain(index, embs)
        else:
            index.add(embs)

        file_hash = calculate_file_hash(path)
        created_at = datetime.now().isoformat()

        chunk_records: list[ChunkRecord] = []
        metadata_entries: list[MetadataDict] = []
        for i, chunk in enumerate(chunks):
            chunk_records.append(
                {
                    "source": path,
                    "chunk_index": i,
                    "text": chunk,
                    "file_hash": file_hash,
                    "created_at": created_at,
                }
            )
            metadata_entries.append(MetadataDict(source=path, chunk_index=i, text=chunk))

        metadata.extend(metadata_entries)
        insert_chunk_records(self._ctx.config.db_path, chunk_records)

        self._faiss_manager.save(index, self._ctx.config.index_path)
        self._env.cleanup()

        return {"index": index, "chunk_count": len(chunks)}

    def scan(
        self,
        root_path: str | Path,
        index: faiss.Index | None,
        metadata: list[MetadataDict],
        processed: set[str],
        *,
        recurse_subdirs: bool = True,
    ) -> ProcessingStats:
        """Walk the filesystem and process supported files."""
        cfg = self._ctx.config
        file_count = 0
        chunk_total = len(metadata)
        skipped_already_processed = 0
        skipped_problems = 0
        skip_reasons: dict[str, int] = {}

        if recurse_subdirs:
            walker: Iterable[tuple[str, list[str], list[str]]] = os.walk(root_path)
        else:
            try:
                root_files: list[str] = [
                    f
                    for f in os.listdir(root_path)
                    if os.path.isfile(os.path.join(root_path, f))
                ]
                walker = [(str(root_path), [], root_files)]
            except OSError:
                print(f"❌ Cannot access directory: {root_path}")
                return {
                    "index": index,
                    "file_count": 0,
                    "chunk_total": chunk_total,
                    "skipped_already_processed": 0,
                    "skipped_problems": 0,
                    "skip_reasons": {},
                }

        for dirpath, _, filenames in walker:
            for fname in filenames:
                path = os.path.join(dirpath, fname)
                ext = os.path.splitext(fname)[1].lower()

                skip, reason = should_skip_file(path, ext, processed, cfg)
                if skip:
                    if reason == "already processed":
                        skipped_already_processed += 1
                        if file_count % cfg.top_print_every == 0:
                            print(f"✓ Already processed: {fname}")
                    else:
                        skipped_problems += 1
                        print(f"💨 Skipping {fname}: {reason}")
                        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                    continue

                try:
                    print(f"Processing: {path}")

                    file_count += 1
                    with self._env.memory_guard():
                        result = self.process_file(path, index, metadata)
                    index = result["index"]
                    chunk_count = result["chunk_count"]
                    chunk_total += chunk_count

                    if cfg.processed_log:
                        file_hash = calculate_file_hash(path)
                        filename = os.path.basename(path)
                        if file_hash:
                            with open(cfg.processed_log, "a", encoding="utf-8") as handle:
                                _ = handle.write(f"{file_hash}|{filename}\n")

                    if file_count % cfg.top_print_every == 0:
                        print(
                            "⚙️ Processed " +
                            f"{file_count} files | total chunks: {chunk_total} | " +
                            f"already done: {skipped_already_processed} | problems: {skipped_problems}"
                        )

                except Exception as exc:
                    skipped_problems += 1
                    error_type = type(exc).__name__
                    error_msg = str(exc)
                    print(
                        f"⚠️ Failed: {os.path.basename(path)} - {error_type}: {error_msg[:100]}"
                    )

                    with open(config.CRASH_LOG_FILE, "a", encoding="utf-8") as handle:
                        _ = handle.write(f"\n{'='*60}\n")
                        _ = handle.write(f"CRASHED FILE: {path}\n")
                        _ = handle.write(f"ERROR TYPE: {error_type}\n")
                        _ = handle.write(f"ERROR: {error_msg}\n")
                        _ = handle.write("TRACEBACK:\n")
                        _ = handle.write(traceback.format_exc())
                        _ = handle.write(f"\n{'='*60}\n")

                    skip_reasons[error_type] = skip_reasons.get(error_type, 0) + 1
                    self._env.cleanup()

        return {
            "index": index,
            "file_count": file_count,
            "chunk_total": chunk_total,
            "skipped_already_processed": skipped_already_processed,
            "skipped_problems": skipped_problems,
            "skip_reasons": skip_reasons,
        }


__all__ = ["DocumentExtractor", "Chunker", "FileScanner"]
