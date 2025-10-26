from __future__ import annotations

import argparse
from io import TextIOWrapper
import os
import sys
from typing import Sequence, cast

import config
from ingestion.environment import EnvironmentManager
from ingestion.pipeline import build_index


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for the ingestion pipeline."""
    cast(TextIOWrapper, sys.stdout).reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Ingest folder -> FAISS (PDF/HTML/Images with OCR fallback)"
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=".",
        help="Root folder of documents to process (default: current directory)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start from scratch, clearing existing index and processed files log",
    )
    parser.add_argument(
        "--no-recurse",
        action="store_true",
        help="Only process files in the root folder, skip subdirectories",
    )
    parser.add_argument(
        "--filetypes",
        type=str,
        help="Comma-separated list of file extensions to process (e.g., 'pdf,png,jpg').",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    os.system("cls" if os.name == "nt" else "clear")

    print(f"Using settings from {config.SETTINGS_FILE}")

    env = EnvironmentManager()
    env.apply()
    ctx = env.initialize()

    if args.folder == ".":
        print("No folder specified, using current directory")
    else:
        print(f"Processing folder: {args.folder}")

    build_index(
        args.folder,
        ctx,
        env=env,
        fresh_start=args.fresh,
        recurse_subdirs=not args.no_recurse,
        filetypes=args.filetypes,
    )


__all__ = ["main"]
