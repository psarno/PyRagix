from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence, TypedDict

import sqlite_utils

import config
from types_models import MetadataDict

logger = logging.getLogger(__name__)


class ChunkRecord(TypedDict):
    source: str
    chunk_index: int
    text: str
    file_hash: str
    created_at: str


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
    records: list[MetadataDict] = []
    for row in chunks_table.rows:
        try:
            records.append(
                {
                    "source": str(row["source"]),
                    "chunk_index": int(row["chunk_index"]),
                    "text": str(row["text"]),
                }
            )
        except KeyError:
            continue
    return records


def insert_chunk_records(db_path: Path, chunk_records: Sequence[ChunkRecord]) -> None:
    """Ensure the chunks table exists and insert the provided records."""
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
            },
            pk="id",
        )
        chunks_table.create_index(["source"])
        chunks_table.create_index(["file_hash"])

    chunks_table.insert_all(chunk_records)


def build_bm25_index(metadata: Sequence[MetadataDict]) -> None:
    """Build and persist the BM25 keyword index when enabled."""
    if not config.ENABLE_HYBRID_SEARCH:
        logger.info("BM25 indexing disabled (ENABLE_HYBRID_SEARCH=False)")
        return

    print("🔨 Building BM25 keyword index...")
    try:
        from utils.bm25_index import build_bm25_index as build, save_bm25_index

        texts = [meta["text"] for meta in metadata]
        if not texts:
            print("⚠️ BM25 indexing skipped: no chunk metadata available.")
            logger.info("BM25 indexing skipped because metadata list is empty")
            return

        bm25_index = build(texts)
        bm25_path = Path(config.BM25_INDEX_PATH)
        save_bm25_index(bm25_index, bm25_path)
        print(f"✅ BM25 index saved: {bm25_path} ({len(bm25_index)} documents)")
    except ImportError:
        print(
            "⚠️ BM25 indexing failed: rank-bm25 not installed. "
            "Run: uv add rank-bm25"
        )
    except Exception as exc:
        print(f"⚠️ BM25 indexing failed: {exc}")
        logger.error(f"BM25 index build error: {exc}", exc_info=True)


__all__ = ["ChunkRecord", "load_metadata", "insert_chunk_records", "build_bm25_index"]
