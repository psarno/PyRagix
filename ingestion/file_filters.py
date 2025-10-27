import hashlib
import logging
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING, cast

import fitz  # PyMuPDF
import sqlite_utils

from classes.ProcessingConfig import ProcessingConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sqlite_utils import Cursor


def calculate_file_hash(path: str) -> str:
    """Calculate SHA256 hash of file contents for duplicate detection."""
    try:
        file_size = os.path.getsize(path)
    except OSError as exc:
        logger.error(f"⚠️ Could not read file size for hashing: {exc}")
        return ""

    if file_size < 1024 * 1024:
        chunk_size = 4096
    elif file_size < 10 * 1024 * 1024:
        chunk_size = 64 * 1024
    elif file_size < 100 * 1024 * 1024:
        chunk_size = 256 * 1024
    else:
        chunk_size = 1024 * 1024

    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, MemoryError) as exc:
        logger.error(f"⚠️ Could not hash {os.path.basename(path)}: {exc}")
        return ""


def should_skip_file(
    path: str,
    ext: str,
    processed: set[str],
    cfg: ProcessingConfig,
) -> tuple[bool, str]:
    """Determine whether a file should be skipped."""
    filename = os.path.basename(path)
    if filename in cfg.skip_files:
        return True, "file in hard-coded skip list."

    effective_extensions = cfg.get_effective_extensions()
    if ext not in effective_extensions:
        if cfg.allowed_extensions is not None:
            return True, f"file type not in filter: {ext}"
        return True, f"unsupported file type: {ext}"

    file_hash = calculate_file_hash(path)
    if file_hash and file_hash in processed:
        return True, "already processed"

    try:
        file_size_mb = os.path.getsize(path) / 1024 / 1024
    except OSError:
        return True, "cannot stat file"
    if file_size_mb > cfg.max_file_mb:
        return True, f"large file ({file_size_mb:.1f} MB)"

    if ext == ".pdf":
        try:
            with fitz.open(path) as doc:
                if doc.page_count > cfg.max_pdf_pages:
                    return True, f"PDF with {doc.page_count} pages"
                if cfg.skip_form_pdfs:
                    try:
                        if getattr(doc, "widgets", lambda: cast(list[Any], []))():
                            return True, "form-heavy PDF (has interactive fields)"
                    except AttributeError:
                        pass
        except (OSError, RuntimeError, ValueError) as exc:
            return True, f"cannot open PDF: {exc}"

    return False, ""


def load_processed_files(cfg: ProcessingConfig) -> set[str]:
    """Load hashes recorded in processed_files.txt."""
    processed_hashes: set[str] = set()
    log_path = cfg.processed_log
    if not log_path or not log_path.exists():
        return processed_hashes

    try:
        with open(log_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line and "|" in line:
                    try:
                        file_hash, _ = line.split("|", 1)
                        processed_hashes.add(file_hash)
                    except ValueError:
                        continue
    except UnicodeDecodeError:
        logger.warning("⚠️ processed_files.txt not UTF-8, rewriting...")
        with open(log_path, "r", encoding="cp1252", errors="ignore") as handle:
            lines = [line.strip() for line in handle if line.strip()]
        with open(log_path, "w", encoding="utf-8") as handle:
            for line in lines:
                _ = handle.write(f"{line}\n")
        return load_processed_files(cfg)

    return processed_hashes


def detect_stale_documents(
    processed_files: set[str],
    current_files: list[Path],
    cfg: ProcessingConfig,
) -> list[str]:
    """Detect processed files whose sources no longer exist."""
    if not processed_files:
        return []

    current_hashes: set[str] = set()
    for file_path in current_files:
        if file_path.is_file():
            try:
                file_hash = calculate_file_hash(str(file_path))
                if file_hash:
                    current_hashes.add(file_hash)
            except (OSError, PermissionError):
                continue

    stale_hashes = processed_files - current_hashes
    stale_paths: list[str] = []
    if stale_hashes and cfg.processed_log and cfg.processed_log.exists():
        try:
            with open(cfg.processed_log, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line and "|" in line:
                        parts = line.split("|", 1)
                        if len(parts) == 2:
                            file_hash, file_path = parts[0].strip(), parts[1].strip()
                            if file_hash in stale_hashes:
                                stale_paths.append(file_path)
        except (OSError, UnicodeDecodeError):
            pass

    return stale_paths


def prompt_user_stale_action(stale_files: list[str]) -> str:
    """Prompt the operator for how to handle missing files."""
    print("\n⚠️  Stale document references detected!")
    print(f"\nFound {len(stale_files)} processed files that no longer exist:")
    for file_path in stale_files[:5]:
        print(f"• {file_path}")
    if len(stale_files) > 5:
        print(f"• (+ {len(stale_files) - 5} more...)")

    print("\nOptions:")
    print("[F]resh start - Clear all data and reprocess from scratch")
    print("[C]lean - Remove stale entries and process new/changed files")
    print("[A]ppend - Keep existing data and add new files only")
    print("[Q]uit - Exit without changes")

    while True:
        try:
            choice = input("\nChoose [F/C/A/Q]: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print("\nOperation cancelled.")
            return "quit"

        if choice in ["F", "FRESH"]:
            return "fresh"
        if choice in ["C", "CLEAN"]:
            return "clean"
        if choice in ["A", "APPEND"]:
            return "append"
        if choice in ["Q", "QUIT"]:
            return "quit"
        print("Please enter F, C, A, or Q")


def clean_stale_entries(stale_files: list[str], cfg: ProcessingConfig) -> None:
    """Remove stale entries from processed log + SQLite database."""
    if not stale_files:
        return

    print(f"🧹 Cleaning {len(stale_files)} stale entries...")
    stale_hashes: set[str] = set()
    log_path = cfg.processed_log

    if log_path and log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line and "|" in line:
                        parts = line.split("|", 1)
                        if len(parts) == 2:
                            file_hash, file_path = parts[0].strip(), parts[1].strip()
                            if file_path in stale_files:
                                stale_hashes.add(file_hash)
        except (OSError, UnicodeDecodeError):
            print("   Warning: Could not read processed files log")

    if stale_hashes and log_path:
        try:
            valid_entries: list[str] = []
            with open(log_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line and "|" in line:
                        parts = line.split("|", 1)
                        if len(parts) == 2:
                            file_hash, _ = parts[0].strip(), parts[1].strip()
                            if file_hash not in stale_hashes:
                                valid_entries.append(line)

            with open(log_path, "w", encoding="utf-8") as handle:
                for entry in valid_entries:
                    _ = handle.write(f"{entry}\n")

            print(
                f"   Cleaned processed files log ({len(valid_entries)} entries remain)"
            )
        except (OSError, UnicodeDecodeError) as exc:
            print(f"   Warning: Could not clean processed files log: {exc}")

    if cfg.db_path and cfg.db_path.exists():
        try:
            db = sqlite_utils.Database(cfg.db_path)
            stale_sources: list[str] = []
            for path_str in stale_files:
                stale_sources.extend(
                    [
                        Path(path_str).name,
                        path_str,
                        (
                            str(Path(path_str).relative_to(Path.cwd()))
                            if Path(path_str).is_absolute()
                            else path_str
                        ),
                    ]
                )

            if stale_sources:
                placeholders = ",".join(["?" for _ in stale_sources])
                query = f"DELETE FROM chunks WHERE source IN ({placeholders})"
                _ = db.execute(query, stale_sources)

                cursor: Cursor = db.execute("SELECT changes()")
                row = cursor.fetchone()
                deleted_count: int = row[0] if row else 0
                print(f"   Cleaned database ({deleted_count} chunks removed)")
        except Exception as exc:
            print(f"   Warning: Could not clean database: {exc}")

    if cfg.index_path and cfg.index_path.exists():
        cfg.index_path.unlink()
        print("   Removed FAISS index (will be rebuilt)")

    print("✅ Stale entries cleaned successfully")


__all__ = [
    "calculate_file_hash",
    "clean_stale_entries",
    "detect_stale_documents",
    "load_processed_files",
    "prompt_user_stale_action",
    "should_skip_file",
]
