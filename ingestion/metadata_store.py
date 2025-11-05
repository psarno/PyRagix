"""SQLite-backed metadata persistence shared across the ingestion pipelines.

The table layout mirrors the .NET port (`pyragix-net`) so both runtimes can
reason about identical chunk metadata (hashes, ordering, file types). Any
schema change here must be reflected in the sibling repository and the
downstream retrieval code that consumes these fields.
"""

import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Sequence, TypedDict, cast

import sqlite_utils
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import config
from types_models import MetadataDict

logger = logging.getLogger(__name__)


class ChunkRecord(TypedDict):
    """Raw record layout persisted to SQLite for each chunk entry."""

    source: str
    chunk_index: int
    text: str
    file_hash: str
    created_at: str
    file_type: str
    total_chunks: int


def load_metadata(db_path: Path) -> list[MetadataDict]:
    """Load all chunk metadata from SQLite, if available."""
    if not db_path.exists():
        return []

    try:
        db = sqlite_utils.Database(str(db_path))
    except Exception as exc:
        logger.warning(f"⚠️ Could not open metadata database: {exc}")
        return []

    if "chunks" not in db.table_names():
        return []

    chunks_table = db["chunks"]
    rows_iter = cast(Iterable[Mapping[str, Any]], chunks_table.rows)
    records: list[MetadataDict] = []
    # sqlite-utils exposes rows as an iterable property, not a callable
    for row in rows_iter:
        try:
            records.append(
                MetadataDict(
                    source=str(row["source"]),
                    chunk_index=int(row["chunk_index"]),
                    text=str(row["text"]),
                    file_type=str(row["file_type"]),
                    total_chunks=int(row["total_chunks"]),
                )
            )
        except KeyError as exc:
            # Fail fast: new schema required for parity with the .NET ingestion pipeline.
            error_msg = (
                f"Database schema is outdated. Missing required field: {exc}. "
                + "Please re-ingest your documents with: uv run python ingest_folder.py --fresh ./docs"
            )
            raise RuntimeError(error_msg) from exc
    return records


def insert_chunk_records(db_path: Path, chunk_records: Sequence[ChunkRecord]) -> None:
    """Ensure the chunks table exists and insert the provided records.

    The schema must stay aligned with `ChunkRecord` and `MetadataDict` to
    guarantee retrieval components—and the .NET port—see a consistent view.
    """
    if not chunk_records:
        return

    db = sqlite_utils.Database(str(db_path))
    chunks_table = db["chunks"]

    if "chunks" not in db.table_names():
        chunks_table.create(
            {
                "id": int,
                "source": str,
                "chunk_index": int,
                "text": str,
                "file_hash": str,
                "created_at": str,
                "file_type": str,
                "total_chunks": int,
            },
            pk="id",
        )
        # Indexes accelerate restarts where we look up by source or detect stale hashes.
        # Keep index names consistent with the .NET port so migrations remain trivial.
        chunks_table.create_index(["source"])
        chunks_table.create_index(["file_hash"])

    chunks_table.insert_all(cast(Sequence[dict[str, Any]], chunk_records))


def build_bm25_index(metadata: Sequence[MetadataDict]) -> None:
    """Build and persist the BM25 keyword index when enabled.

    The resulting pickle must live alongside the FAISS index so hybrid retrieval
    stays transactionally consistent with chunk metadata.
    """
    if not config.ENABLE_HYBRID_SEARCH:
        logger.info("BM25 indexing disabled (ENABLE_HYBRID_SEARCH=False)")
        return

    print("🔨 Building BM25 keyword index...")
    try:
        from utils.bm25_index import build_bm25_index as build, save_bm25_index
    except ImportError:
        print(
            "⚠️ BM25 indexing failed: rank-bm25 not installed. "
            + "Run: uv add rank-bm25"
        )
        return

    texts = [meta.text for meta in metadata]
    if not texts:
        print("⚠️ BM25 indexing skipped: no chunk metadata available.")
        logger.info("BM25 indexing skipped because metadata list is empty")
        return

    @retry(
        retry=retry_if_exception_type((RuntimeError, ValueError, OSError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _build_and_save() -> None:
        # Build the BM25 index out-of-process and atomically persist it so hybrid search stays consistent.
        bm25_index = build(texts)
        bm25_path = Path(config.BM25_INDEX_PATH)
        save_bm25_index(bm25_index, bm25_path)
        print(f"✅ BM25 index saved: {bm25_path} ({len(bm25_index)} documents)")

    try:
        _build_and_save()
    except Exception as exc:
        print(f"⚠️ BM25 indexing failed after retries: {exc}")
        logger.error("BM25 index build error after retries: %s", exc, exc_info=True)


__all__ = ["ChunkRecord", "load_metadata", "insert_chunk_records", "build_bm25_index"]
