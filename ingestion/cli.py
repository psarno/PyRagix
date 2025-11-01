import argparse
from io import TextIOWrapper
import os
from pathlib import Path
import sys
from typing import Sequence, cast

import config
from ingestion.environment import EnvironmentManager
from ingestion.pipeline import build_index


def validate_input_path(folder_path: str) -> None:
    """Validate the input path to prevent accidentally indexing dangerous locations.

    Args:
        folder_path: Path to validate

    Raises:
        SystemExit: If the path is potentially dangerous
    """
    path = Path(folder_path).resolve()

    # Check if path exists
    if not path.exists():
        print(f"❌ Error: Path does not exist: {folder_path}")
        sys.exit(1)

    if not path.is_dir():
        print(f"❌ Error: Path is not a directory: {folder_path}")
        sys.exit(1)

    # Warn about root/home directory
    home = Path.home().resolve()
    # Check if path is a filesystem root (parent == itself)
    is_root = path.parent == path

    if is_root:
        print(f"❌ Error: Cannot index filesystem root directory: {path}")
        print("   This would scan your entire system.")
        print("   Please specify a specific folder (e.g., './docs')")
        sys.exit(1)

    if path == home:
        print(f"❌ Error: Cannot index home directory: {path}")
        print("   This would scan all your personal files.")
        print("   Please specify a specific folder (e.g., './docs' or './Documents')")
        sys.exit(1)

    # Warn if indexing a parent of home (even more dangerous)
    try:
        _ = home.relative_to(path)
        print(f"❌ Error: Path {path} contains your home directory")
        print("   This would scan all your personal files and system directories.")
        print("   Please specify a specific document folder.")
        sys.exit(1)
    except ValueError:
        # path is not a parent of home - this is fine
        pass


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for the ingestion pipeline."""
    cast(TextIOWrapper, sys.stdout).reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Ingest folder -> FAISS (PDF/HTML/Images with OCR fallback)"
    )
    _ = parser.add_argument(
        "folder",
        help="Root folder of documents to process (e.g., './docs' or '.')",
    )
    _ = parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start from scratch, clearing existing index and processed files log",
    )
    _ = parser.add_argument(
        "--no-recurse",
        action="store_true",
        help="Only process files in the root folder, skip subdirectories",
    )
    _ = parser.add_argument(
        "--filetypes",
        type=str,
        help="Comma-separated list of file extensions to process (e.g., 'pdf,png,jpg').",
    )
    _ = parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional diagnostics about each ingestion stage.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    _ = os.system("cls" if os.name == "nt" else "clear")

    print(f"Using settings from {config.SETTINGS_FILE}")

    # Validate path before processing
    validate_input_path(args.folder)

    env = EnvironmentManager()
    env.apply()
    ctx = env.initialize()

    print(f"Processing folder: {args.folder}")

    build_index(
        args.folder,
        ctx,
        env=env,
        fresh_start=args.fresh,
        recurse_subdirs=not args.no_recurse,
        filetypes=args.filetypes,
        verbose=args.verbose,
    )


__all__ = ["main"]
