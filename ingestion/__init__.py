"""Ingestion subsystem helpers.

This package gradually absorbs functionality from `ingest_folder.py` so the
pipeline can be composed from smaller, testable modules.
"""

from . import (
    cli,
    environment,
    faiss_manager,
    file_filters,
    file_scanner,
    metadata_store,
    models,
    pipeline,
    stale_cleaner,
    text_processing,
)

__all__ = [
    "cli",
    "environment",
    "faiss_manager",
    "file_filters",
    "file_scanner",
    "metadata_store",
    "models",
    "pipeline",
    "stale_cleaner",
    "text_processing",
]
