"""
Ingestion subsystem helpers.

This package gradually absorbs functionality from `ingest_folder.py` so the
pipeline can be composed from smaller, testable modules.
"""

from ingestion import (
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
    "models",
    "faiss_manager",
    "file_filters",
    "metadata_store",
    "file_scanner",
    "text_processing",
    "pipeline",
    "cli",
    "environment",
    "stale_cleaner",
]
