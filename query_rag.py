"""CLI entrypoint for querying the RAG system."""

from __future__ import annotations

import io
import sys
import traceback
import warnings

# Suppress misleading PaddlePaddle ccache warning BEFORE any imports
# (only relevant when building from source, not using pre-built wheels)
warnings.filterwarnings("ignore", message=".*ccache.*", category=UserWarning)

from typing import TYPE_CHECKING, Callable  # noqa: E402

from __version__ import __version__  # noqa: E402
from rag.configuration import DEFAULT_CONFIG, validate_config  # noqa: E402
from types_models import MetadataDict, RAGConfig  # noqa: E402
from utils.ollama_status import (  # noqa: E402
    OllamaUnavailableError,
    ensure_ollama_model_available,
)
from utils.spinner import Spinner  # noqa: E402

if TYPE_CHECKING:
    import faiss
    from sentence_transformers import SentenceTransformer

LoadRagSystemFn = Callable[
    [RAGConfig], tuple["faiss.Index", list[MetadataDict], "SentenceTransformer"]
]
RunQueryFn = Callable[
    [
        str,
        "faiss.Index",
        list[MetadataDict],
        "SentenceTransformer",
        RAGConfig,
        int | None,
        bool,
        bool,
    ],
    str | None,
]

_load_rag_system: LoadRagSystemFn | None = None
_run_query_rag: RunQueryFn | None = None


def _ensure_pipeline_loaded() -> tuple[LoadRagSystemFn, RunQueryFn]:
    global _load_rag_system, _run_query_rag

    if _load_rag_system is None or _run_query_rag is None:
        from rag.loader import load_rag_system as _loader
        from rag.retrieval import query_rag as _query_runner

        _load_rag_system = _loader
        _run_query_rag = _query_runner

    # Type checkers: the tuple unpacks to non-None after assignment above.
    assert _load_rag_system is not None and _run_query_rag is not None
    return _load_rag_system, _run_query_rag


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


def _configure_readline() -> None:
    """Enable readline so arrow keys and editing work in the prompt."""
    if sys.platform == "win32":
        return

    try:
        import readline  # noqa: F401

        # Basic bindings for a familiar shell experience.
        readline.parse_and_bind("tab: complete")
        readline.parse_and_bind("set enable-meta-key on")
    except ImportError:
        # Some minimal Python builds omit readline; fall back gracefully.
        pass


def _query_rag(
    query: str,
    index: faiss.Index,
    metadata: list[MetadataDict],
    embedder: SentenceTransformer,
    config: RAGConfig,
    top_k: int | None = None,
    show_sources: bool = True,
    debug: bool = True,
) -> str | None:
    """Compatibility wrapper around the refactored query routine."""
    _, query_runner = _ensure_pipeline_loaded()
    return query_runner(
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
    _configure_readline()

    if config is None:
        config = DEFAULT_CONFIG.model_copy()

    try:
        validate_config(config)

        spinner_enabled = sys.stdout.isatty()

        if not spinner_enabled:
            print("Checking Ollama availability...")

        with Spinner(
            "Checking Ollama availability...",
            enabled=spinner_enabled,
            final_message="✅ Ollama ready." if spinner_enabled else None,
        ):
            ensure_ollama_model_available(
                config.ollama_base_url, config.ollama_model
            )

        if not spinner_enabled:
            print("✅ Ollama ready.")

        if not spinner_enabled:
            print("Initializing RAG pipeline...")

        with Spinner(
            "Initializing RAG pipeline...",
            enabled=spinner_enabled,
            final_message="✅ RAG pipeline ready." if spinner_enabled else None,
        ):
            loader, _ = _ensure_pipeline_loaded()
            index, metadata, embedder = loader(config)

        if not spinner_enabled:
            print("✅ RAG pipeline ready.")

        print(f"Pyragix query system (Version {__version__})")
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

    except OllamaUnavailableError as exc:
        print(f"❌ {exc}")
        sys.exit(1)
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
