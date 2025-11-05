"""Detect and reconcile stale ingestion artifacts before new runs.

The processed ledger (`processed_files.txt`) must stay aligned with the on-disk
corpus; otherwise hybrid retrieval will surface dead chunks.  This helper
mirrors the .NET port's behaviour so both runtimes prompt operators with the
same options (fresh start, clean, append) when hashes drift or files go missing.
"""

import logging
from pathlib import Path

from classes.ProcessingConfig import ProcessingConfig
from ingestion.file_filters import (
    clean_stale_entries,
    detect_stale_documents,
    prompt_user_stale_action,
)

logger = logging.getLogger(__name__)


class StaleDocumentCleaner:
    """Manages detection and cleanup of stale document references."""

    def __init__(self, config: ProcessingConfig) -> None:
        super().__init__()
        self.config = config

    def check_and_handle_stale_documents(
        self,
        processed_hashes: set[str],
        current_files: list[Path],
    ) -> tuple[bool, set[str]]:
        """Check for stale documents and prompt the operator to reconcile them.

        Returns a tuple ``(fresh_start_requested, updated_hashes)`` so the
        caller can decide whether to blow away indices or continue appending.
        ``updated_hashes`` reflects on-disk mutations when the operator chooses
        to remove stale entries.
        """
        if not processed_hashes:
            logger.info("No processed files to check for staleness")
            return False, processed_hashes

        print("🔍 Checking for stale document references...")

        stale_files = detect_stale_documents(
            processed_hashes,
            current_files,
            self.config,
        )

        if not stale_files:
            print("✅ No stale documents detected")
            return False, processed_hashes

        # Prompt user for action
        user_choice = prompt_user_stale_action(stale_files)

        if user_choice == "quit":
            import sys

            print("Operation cancelled by user.")
            sys.exit(0)

        if user_choice == "fresh":
            print("🆕 User chose fresh start - clearing all existing files")
            self._clear_all_data()
            return True, set()

        if user_choice == "clean":
            print("🧹 User chose clean - removing stale entries")
            clean_stale_entries(stale_files, self.config)
            # Reload processed files after cleanup.
            from ingestion.file_filters import load_processed_files

            # Reloading keeps the caller aligned with whatever rows survived the cleanup.
            return False, load_processed_files(self.config)

        # user_choice == "append" - keep existing data
        return False, processed_hashes

    def _clear_all_data(self) -> None:
        """Remove all existing index, database, and processed files log."""
        if self.config.index_path.exists():
            self.config.index_path.unlink()
            print("   Removed existing FAISS index")

        if self.config.db_path.exists():
            self.config.db_path.unlink()
            print("   Removed existing metadata database")

        if self.config.processed_log.exists():
            self.config.processed_log.unlink()
            print("   Removed processed files log")


__all__ = ["StaleDocumentCleaner"]
