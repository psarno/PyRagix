"""CLI entrypoint for querying the RAG system."""

from __future__ import annotations

import io
import sys
import traceback
import warnings

# Suppress misleading PaddlePaddle ccache warning BEFORE any imports
# (only relevant when building from source, not using pre-built wheels)
warnings.filterwarnings("ignore", message=".*ccache.*", category=UserWarning)

from typing import TYPE_CHECKING  # noqa: E402

from __version__ import __version__  # noqa: E402
from rag.configuration import DEFAULT_CONFIG, validate_config  # noqa: E402
from rag.loader import load_rag_system  # noqa: E402
from rag.retrieval import query_rag as _run_query_rag  # noqa: E402
from types_models import MetadataDict, RAGConfig  # noqa: E402

if TYPE_CHECKING:
    import faiss
    from sentence_transformers import SentenceTransformer


_load_rag_system = load_rag_system
_validate_config = validate_config


def _ensure_utf8_stdio() -> None:
    """Force UTF-8 encoding for Windows Git Bash consoles."""
    if sys.platform != "win32" or "pytest" in sys.modules:
        return

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        buffer = getattr(stream, "buffer", None)
        if buffer is None:
            continue
        try:
            wrapper = io.TextIOWrapper(
                buffer,
                encoding="utf-8",
                errors="replace",
                line_buffering=True,
            )
            setattr(sys, stream_name, wrapper)
        except (LookupError, OSError, ValueError):
            continue


def _query_rag(
    query: str,
    index: "faiss.Index",
    metadata: list[MetadataDict],
    embedder: "SentenceTransformer",
    config: RAGConfig,
    top_k: int | None = None,
    show_sources: bool = True,
    debug: bool = True,
) -> str | None:
    """Compatibility wrapper around the refactored query routine."""
    return _run_query_rag(
        query=query,
        index=index,
        metadata=metadata,
        embedder=embedder,
        config_obj=config,
        top_k=top_k,
        show_sources=show_sources,
        debug=debug,
    )


def main(config: RAGConfig | None = None) -> None:
    """Main function to run the RAG query system."""
    _ensure_utf8_stdio()

    if config is None:
        config = DEFAULT_CONFIG.model_copy()

    try:
        _validate_config(config)
        index, metadata, embedder = _load_rag_system(config)

        print(f"\nRAG Query System Ready! (Version {__version__})")
        print("Type your questions (or 'quit' to exit)")

        while True:
            try:
                query = input("\nQuery: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n👋 Goodbye!")
                break

            if query.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break

            if not query:
                continue

            _ = _query_rag(
                query, index, metadata, embedder, config, show_sources=True, debug=True
            )

    except FileNotFoundError as exc:
        print(f"❌ File not found: {exc}")
        print("Make sure you've run ingest_folder.py first!")
        sys.exit(1)
    except ValueError as exc:
        print(f"❌ Configuration error: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"❌ Unexpected error: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
