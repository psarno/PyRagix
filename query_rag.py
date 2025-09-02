# ======================================
# RAG Query System
# - Retrieval-Augmented Generation interface
# - FAISS vector search with Ollama LLM
# - Memory-optimized embedding operations
# ======================================

# ===============================
# Standard Library
# ===============================
import sqlite_utils
import sys
import traceback
from pathlib import Path
from typing import (
    List,
    Optional,
    Tuple,
    Dict,
    Any,
    Protocol,
)
from typing_extensions import TypedDict
from contextlib import contextmanager

# ===============================
# Third-party Libraries
# ===============================
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# Import our config for machine-specific settings
import config


# ===============================
# Type Definitions
# ===============================
class RAGConfig(TypedDict):
    """Configuration for RAG system."""

    embed_model: str
    index_path: Path
    db_path: Path
    ollama_base_url: str
    ollama_model: str
    default_top_k: int
    request_timeout: int
    temperature: float
    top_p: float
    max_tokens: int


class SearchResult(TypedDict):
    """Single search result from FAISS."""

    source: str
    chunk_idx: int
    score: float
    text: str


class OllamaResponse(Protocol):
    """Protocol for Ollama API response."""

    status_code: int

    def json(self) -> Dict[str, Any]: ...


# ===============================
# Configuration
# ===============================
from __version__ import __version__

# Default configuration - uses config.py for machine-specific settings
DEFAULT_CONFIG: RAGConfig = {
    "embed_model": config.EMBED_MODEL,
    "index_path": Path("local_faiss.index"),
    "db_path": Path("documents.db"),
    "ollama_base_url": config.OLLAMA_BASE_URL,
    "ollama_model": config.OLLAMA_MODEL,
    "default_top_k": config.DEFAULT_TOP_K,
    "request_timeout": config.REQUEST_TIMEOUT,
    "temperature": config.TEMPERATURE,
    "top_p": config.TOP_P,
    "max_tokens": config.MAX_TOKENS,
}


# ===============================
# Helper Functions
# ===============================
@contextmanager
def _memory_cleanup():
    """Context manager for automatic memory cleanup after operations."""
    try:
        yield
    finally:
        # Force garbage collection to prevent memory fragmentation
        import gc

        gc.collect()


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Normalize embeddings using L2 norm.

    Args:
        mat: Input embedding matrix of shape (n_samples, n_features)

    Returns:
        np.ndarray: L2-normalized embeddings
    """
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def _generate_answer_with_ollama(
    query: str, context_chunks: List[str], config: RAGConfig
) -> str:
    """Generate a human-like answer using Ollama based on retrieved context.

    Args:
        query: User's question
        context_chunks: List of relevant text chunks from vector search
        config: RAG configuration containing Ollama settings

    Returns:
        str: Generated answer or error message
    """
    # Combine context chunks
    context = "\n\n".join(
        [f"Document {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)]
    )

    # Create the prompt
    prompt = f"""Analyze these documents to answer the question comprehensively. Use ONLY what is written in the documents.

DOCUMENTS:
{context}

QUESTION: {query}

Instructions:
- Review ALL documents above for relevant information
- Synthesize information across multiple documents if available
- Provide a comprehensive answer based on patterns you see
- Quote specific examples from the documents
- If no relevant information exists, respond: "No information found in documents"

Response:"""

    try:
        # Call Ollama API
        response = requests.post(
            f"{config['ollama_base_url']}/api/generate",
            json={
                "model": config["ollama_model"],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config["temperature"],
                    "top_p": config["top_p"],
                    "num_predict": config["max_tokens"],
                },
            },
            timeout=config["request_timeout"],
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("response", "Sorry, I couldn't generate an answer.")
        else:
            return f"Error calling Ollama API: {response.status_code}"

    except requests.exceptions.ConnectionError:
        return "WARNING: Ollama is not running. Please start Ollama first with: ollama serve"
    except requests.exceptions.Timeout:
        return "WARNING: Request timed out. The model might be loading or the query is too complex."
    except requests.exceptions.RequestException as e:
        return f"WARNING: Request error: {str(e)}"
    except (KeyError, ValueError) as e:
        return f"WARNING: Configuration or response parsing error: {str(e)}"
    except Exception as e:
        return f"WARNING: Unexpected error generating answer: {str(e)}"


def _load_rag_system(
    config: RAGConfig,
) -> Tuple[faiss.Index, List[Dict[str, Any]], SentenceTransformer]:
    """Load the FAISS index and metadata.

    Args:
        config: RAG configuration

    Returns:
        tuple: (FAISS index, metadata list, embedder model)

    Raises:
        FileNotFoundError: If index or metadata files don't exist
        Exception: If loading fails
    """
    print("Loading FAISS index and metadata...")

    # Validate files exist
    if not config["index_path"].exists():
        raise FileNotFoundError(f"FAISS index not found: {config['index_path']}")
    if not config["db_path"].exists():
        raise FileNotFoundError(f"Database file not found: {config['db_path']}")

    try:
        # Load FAISS index
        index = faiss.read_index(str(config["index_path"]))

        # Configure IVF index if applicable
        if hasattr(index, "nprobe"):
            # This is an IVF index, set nprobe for search
            nprobe = getattr(config, "NPROBE", 16)  # Default to 16 if not set
            index.nprobe = nprobe  # type: ignore
            print(f"Set IVF nprobe to {nprobe}")

        # Load metadata from database
        db = sqlite_utils.Database(str(config["db_path"]))
        metadata = []
        if "chunks" in db.table_names():
            for row in db["chunks"].rows:
                metadata.append({
                    "source": row["source"],
                    "chunk_index": row["chunk_index"],
                    "text": row["text"]
                })
        else:
            raise ValueError("Database exists but contains no chunks table")

        # Load embedder
        embedder = SentenceTransformer(config["embed_model"])

        # Validate data consistency
        if index.ntotal != len(metadata):
            raise ValueError(
                f"Index/metadata mismatch: {index.ntotal} vectors vs {len(metadata)} metadata entries"
            )

        unique_sources = len(set(m["source"] for m in metadata))
        index_type = "IVF" if hasattr(index, "nprobe") else "Flat"
        device_info = (
            "GPU"
            if hasattr(index, "device") and getattr(index, "device", -1) >= 0
            else "CPU"
        )

        if index_type == "IVF":
            nprobe = getattr(index, "nprobe", "unknown")
            print(
                f"Loaded {index.ntotal} chunks from {unique_sources} files (IVF index on {device_info}, nprobe={nprobe})"
            )
        else:
            print(
                f"Loaded {index.ntotal} chunks from {unique_sources} files (Flat index on {device_info})"
            )
        return index, metadata, embedder

    except Exception as e:
        raise Exception(f"Failed to load RAG system: {str(e)}") from e


def _query_rag(
    query: str,
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    embedder: SentenceTransformer,
    config: RAGConfig,
    top_k: Optional[int] = None,
    show_sources: bool = True,
    debug: bool = True,
) -> Optional[str]:
    """Query the RAG system and generate a human-like answer.

    Args:
        query: User's question
        index: FAISS vector index
        metadata: List of document metadata
        embedder: Sentence transformer model
        config: RAG configuration
        top_k: Number of top results to retrieve
        show_sources: Whether to display source information
        debug: Whether to show debug information

    Returns:
        str: Generated answer, or None if no relevant documents found
    """
    if not query.strip():
        print("⚠️ Empty query provided")
        return None

    if top_k is None:
        top_k = config["default_top_k"]

    print(f"\nQuery: {query}")

    try:
        with _memory_cleanup():
            # Embed the query
            query_emb = embedder.encode(
                [query], convert_to_numpy=True, normalize_embeddings=False
            )
            # Convert to numpy array with proper type
            query_emb_array = np.asarray(query_emb, dtype=np.float32)
            query_emb_normalized = _l2_normalize(query_emb_array)

            # Search FAISS - returns (distances, labels)
            distances, labels = index.search(query_emb_normalized, top_k)  # type: ignore
            # For cosine similarity (IndexFlatIP), distances are actually scores
            scores = distances
            indices = labels

        # Collect relevant chunks
        context_chunks: List[str] = []
        sources_info: List[SearchResult] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue

            if idx >= len(metadata):
                print(f"⚠️ Invalid index {idx} (metadata has {len(metadata)} entries)")
                continue

            meta = metadata[idx]
            source = meta["source"]
            chunk_idx = meta["chunk_index"]
            text = meta["text"]

            context_chunks.append(text)
            sources_info.append(
                {
                    "source": source,
                    "chunk_idx": chunk_idx,
                    "score": float(score),
                    "text": text,
                }
            )

        if not context_chunks:
            print("\nNo relevant documents found.")
            return None

        # Debug: show what chunks are being sent
        if debug:
            print(f"\nSending {len(context_chunks)} chunks to LLM:")
            for i, chunk in enumerate(context_chunks[:2]):  # Show first 2
                print(f"  Chunk {i+1} (len={len(chunk)}): {repr(chunk[:100])}...")
            print()

        # Generate answer using Ollama
        print("🤖 Generating answer...")
        answer = _generate_answer_with_ollama(query, context_chunks, config)

        print("\nAnswer:")
        print("=" * 60)
        print(answer)
        print("=" * 60)

        # Optionally show sources
        if show_sources:
            print("\nSources:")
            for i, info in enumerate(sources_info):
                source_path = Path(info["source"])
                print(
                    f"{i+1}. {source_path.name} (chunk {info['chunk_idx']}, score: {info['score']:.3f})"
                )
            print("-" * 60)

        return answer

    except Exception as e:
        print(f"❌ Error during query processing: {type(e).__name__}: {str(e)}")
        if debug:
            traceback.print_exc()
        return None


def _validate_config(config: RAGConfig) -> None:
    """Validate RAG configuration.

    Args:
        config: Configuration to validate

    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = [
        "embed_model",
        "index_path",
        "db_path",
        "ollama_base_url",
        "ollama_model",
        "default_top_k",
        "request_timeout",
    ]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    if config["default_top_k"] <= 0:
        raise ValueError("default_top_k must be positive")

    if config["request_timeout"] <= 0:
        raise ValueError("request_timeout must be positive")

    # Validate paths are Path objects
    for path_key in ["index_path", "db_path"]:
        if not isinstance(config[path_key], Path):
            raise ValueError(f"{path_key} must be a Path object")


def main(config: Optional[RAGConfig] = None) -> None:
    """Main function to run the RAG query system.

    Args:
        config: Optional RAG configuration. Uses defaults if not provided.
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

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

            _query_rag(
                query, index, metadata, embedder, config, show_sources=True, debug=True
            )

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Make sure you've run ingest_folder.py first!")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
