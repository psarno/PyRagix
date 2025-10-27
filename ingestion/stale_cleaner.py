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
        """
        Check for stale documents and prompt user for action.

        Returns:
            Tuple of (fresh_start_requested, updated_processed_hashes)
            where fresh_start_requested=True means user chose fresh start.
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
            # Reload processed files after cleanup
            from ingestion.file_filters import load_processed_files

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
