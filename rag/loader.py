"""Load FAISS index, metadata, and embedding model."""

from collections.abc import Iterable, Mapping
from typing import Any, cast

import sqlite_utils
from sentence_transformers import SentenceTransformer

import faiss

from types_models import MetadataDict, RAGConfig
import config as global_config
from utils.faiss_types import SupportsDevice, ensure_nprobe

def _row_to_metadata(row: Mapping[str, Any]) -> MetadataDict:
    return MetadataDict(
        source=str(row["source"]),
        chunk_index=int(row["chunk_index"]),
        text=str(row["text"]),
        file_type=str(row["file_type"]),
        total_chunks=int(row["total_chunks"]),
    )


def load_rag_system(
    config: RAGConfig,
) -> tuple[faiss.Index, list[MetadataDict], SentenceTransformer]:
    """Load FAISS index, metadata, and embedder for querying."""
    print("Loading FAISS index and metadata...")

    if not config.index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {config.index_path}")
    if not config.db_path.exists():
        raise FileNotFoundError(f"Database file not found: {config.db_path}")

    try:
        index = faiss.read_index(str(config.index_path))
        try:
            ivf_index = ensure_nprobe(index, context="rag.loader.load_rag_system")
        except TypeError:
            is_ivf_index = False
        else:
            is_ivf_index = True
            ivf_index.nprobe = global_config.NPROBE
            print(f"Set IVF nprobe to {global_config.NPROBE}")

        db = sqlite_utils.Database(str(config.db_path))
        metadata: list[MetadataDict] = []

        if "chunks" not in db.table_names():
            raise ValueError("Database exists but contains no chunks table")

        rows_iter = cast(Iterable[Mapping[str, Any]], db["chunks"].rows)
        for row in rows_iter:
            metadata.append(_row_to_metadata(row))

        embedder = SentenceTransformer(config.embed_model)

        if index.ntotal != len(metadata):
            raise ValueError(
                f"Index/metadata mismatch: {index.ntotal} vectors vs "
                + f"{len(metadata)} metadata entries"
            )

        unique_sources = len(set(m.source for m in metadata))
        device_info = "GPU" if isinstance(index, SupportsDevice) and index.device >= 0 else "CPU"

        if is_ivf_index:
            current_ivf = ensure_nprobe(index, context="rag.loader.load_rag_system (report)")
            nprobe = current_ivf.nprobe
            print(
                f"Loaded {index.ntotal} chunks from {unique_sources} files "
                + f"(IVF index on {device_info}, nprobe={nprobe})"
            )
        else:
            print(
                f"Loaded {index.ntotal} chunks from {unique_sources} files "
                + f"(Flat index on {device_info})"
            )

        return index, metadata, embedder

    except Exception as exc:
        raise Exception(f"Failed to load RAG system: {exc}") from exc


__all__ = ["load_rag_system"]
