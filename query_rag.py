"""CLI entrypoint for querying the RAG system."""

import sys
import traceback
from typing import TYPE_CHECKING

from __version__ import __version__
from rag.configuration import DEFAULT_CONFIG, validate_config
from rag.loader import load_rag_system
from rag.retrieval import query_rag as _run_query_rag
from types_models import MetadataDict, RAGConfig

if TYPE_CHECKING:
    import faiss
    from sentence_transformers import SentenceTransformer


_load_rag_system = load_rag_system
_validate_config = validate_config


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
