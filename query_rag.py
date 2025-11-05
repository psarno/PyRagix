"""CLI entrypoint for querying the RAG system."""

from __future__ import annotations

import argparse
import io
import platform
import sys
import time
import traceback
import warnings

# Suppress misleading PaddlePaddle ccache warning BEFORE any imports
# (only relevant when building from source, not using pre-built wheels)
warnings.filterwarnings("ignore", message=".*ccache.*", category=UserWarning)

from typing import TYPE_CHECKING, Callable, Sequence, TypeVar  # noqa: E402

from __version__ import __version__  # noqa: E402
from rag.configuration import DEFAULT_CONFIG, validate_config  # noqa: E402
from types_models import MetadataDict, RAGConfig  # noqa: E402
from utils.ollama_status import (  # noqa: E402
    OllamaUnavailableError,
    ensure_ollama_model_available,
)
from utils.spinner import Spinner  # noqa: E402
from utils.faiss_types import FaissIndex  # noqa: E402

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

LoadRagSystemFn = Callable[
    [RAGConfig], tuple[FaissIndex, list[MetadataDict], "SentenceTransformer"]
]
RunQueryFn = Callable[
    [
        str,
        FaissIndex,
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

T = TypeVar("T")


def _parse_cli_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive console for querying the PyRagix knowledge base.",
    )
    _ = parser.add_argument(
        "--no-spinner",
        action="store_true",
        help="Disable startup spinners (useful for logging or slow terminals).",
    )
    _ = parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print timing information for startup steps.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


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


def _configure_windows_utf8() -> None:
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


def _ensure_utf8_stdio() -> None:
    """Force UTF-8 encoding for Windows Git Bash consoles."""
    if platform.system().lower().startswith("win") and "pytest" not in sys.modules:
        _configure_windows_utf8()


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
    index: FaissIndex,
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
        query,
        index,
        metadata,
        embedder,
        config,
        top_k,
        show_sources,
        debug,
    )


def _run_with_feedback(
    label: str,
    final_message: str | None,
    *,
    spinner_enabled: bool,
    verbose: bool,
    func: Callable[[], T],
) -> T:
    """Wrap a long-running step with spinner output and optional timing."""
    if spinner_enabled:
        print(label, flush=True)
    else:
        print(label)

    start = time.perf_counter()
    with Spinner(label, enabled=spinner_enabled, final_message=None):
        result = func()

    elapsed = time.perf_counter() - start

    if final_message:
        print(final_message, flush=True)

    if verbose:
        print(f"[verbose] {label} completed in {elapsed:.2f}s", flush=True)

    return result


def main(
    config: RAGConfig | None = None,
    *,
    enable_spinner: bool | None = None,
    verbose: bool = False,
) -> None:
    """Main function to run the RAG query system."""
    _ensure_utf8_stdio()
    _configure_readline()

    if config is None:
        config = DEFAULT_CONFIG.model_copy()

    try:
        validate_config(config)

        spinner_enabled = (
            sys.stdout.isatty() if enable_spinner is None else enable_spinner
        )

        _ = _run_with_feedback(
            "Checking Ollama availability...",
            "✅ Ollama ready.",
            spinner_enabled=spinner_enabled,
            verbose=verbose,
            func=lambda: ensure_ollama_model_available(
                config.ollama_base_url, config.ollama_model
            ),
        )

        def _load_pipeline() -> tuple[
            FaissIndex, list[MetadataDict], SentenceTransformer
        ]:
            loader, _ = _ensure_pipeline_loaded()
            return loader(config)

        index, metadata, embedder = _run_with_feedback(
            "Initializing RAG pipeline...",
            "✅ RAG pipeline ready.",
            spinner_enabled=spinner_enabled,
            verbose=verbose,
            func=_load_pipeline,
        )

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
                query,
                index,
                metadata,
                embedder,
                config,
                show_sources=True,
                debug=True,
            )

    except OllamaUnavailableError as exc:
        print(f"❌ {exc}")
        sys.exit(1)
    except FileNotFoundError as exc:
        print(f"❌ File not found: {exc}")
        message = str(exc)
        if "FAISS index" in message or "Metadata database" in message:
            print("Make sure you've run ingest_folder.py first!")
        sys.exit(1)
    except ValueError as exc:
        print(f"❌ Configuration error: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"❌ Unexpected error: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)


def cli(argv: Sequence[str] | None = None) -> None:
    """CLI entry point that parses arguments before loading heavy dependencies."""
    args = _parse_cli_args(argv)
    spinner_flag: bool | None = False if args.no_spinner else None
    main(enable_spinner=spinner_flag, verbose=args.verbose)


if __name__ == "__main__":
    cli()
