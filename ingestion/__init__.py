from __future__ import annotations

"""
Ingestion subsystem helpers.

This package gradually absorbs functionality from `ingest_folder.py` so the
pipeline can be composed from smaller, testable modules.
"""

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
