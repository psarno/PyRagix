# ======================================
# RAG Query System
# - Retrieval-Augmented Generation interface
# - FAISS vector search with Ollama LLM
# - Memory-optimized embedding operations
# ======================================

from __version__ import __version__

# ===============================
# Standard Library
# ===============================
import sqlite_utils
import sys
import traceback
from pathlib import Path
from typing import (
    Any,
    TYPE_CHECKING,
)
from contextlib import contextmanager

# ===============================
# Third-party Libraries
# ===============================
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# Type checking imports for heavy dependencies
if TYPE_CHECKING:
    import faiss
else:
    # Import at runtime
    import faiss

# Import our config for machine-specific settings
import config

# Import our strong types
from types_models import RAGConfig, SearchResult, MetadataDict

# Import v2 features (conditionally used based on config)
from utils.query_expander import expand_query
from utils.reranker import Reranker

# Global reranker instance (lazy-loaded)
_reranker: Reranker | None = None

# Global BM25 index instance (lazy-loaded)
_bm25_index: Any | None = None  # BM25Index type from utils.bm25_index


def _get_reranker() -> Reranker:
    """Get or create global reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker(model_name=config.RERANKER_MODEL)
    return _reranker


def _get_bm25_index() -> Any | None:
    """Get or load global BM25 index instance.

    Returns:
        BM25Index instance or None if not available/enabled
    """
    global _bm25_index

    if not config.ENABLE_HYBRID_SEARCH:
        return None

    if _bm25_index is None:
        try:
            from utils.bm25_index import load_bm25_index
            bm25_path = Path(config.BM25_INDEX_PATH)
            _bm25_index = load_bm25_index(bm25_path)
            if _bm25_index is None:
                print(f"⚠️ BM25 index not found at {bm25_path}. Hybrid search disabled.")
                print("   Run ingestion with ENABLE_HYBRID_SEARCH=true to build index.")
        except ImportError:
            print("⚠️ rank-bm25 not installed. Hybrid search disabled.")
            print("   Run: uv add rank-bm25")
        except Exception as e:
            print(f"⚠️ Failed to load BM25 index: {e}")

    return _bm25_index


# ===============================
# Configuration
# ===============================
# Default configuration - uses config.py for machine-specific settings
DEFAULT_CONFIG = RAGConfig(
    embed_model=config.EMBED_MODEL,
    index_path=Path("local_faiss.index"),
    db_path=Path("documents.db"),
    ollama_base_url=config.OLLAMA_BASE_URL,
    ollama_model=config.OLLAMA_MODEL,
    default_top_k=config.DEFAULT_TOP_K,
    request_timeout=config.REQUEST_TIMEOUT,
    temperature=config.TEMPERATURE,
    top_p=config.TOP_P,
    max_tokens=config.MAX_TOKENS,
    # Phase 1 (v2) settings
    enable_query_expansion=config.ENABLE_QUERY_EXPANSION,
    query_expansion_count=config.QUERY_EXPANSION_COUNT,
    enable_reranking=config.ENABLE_RERANKING,
    reranker_model=config.RERANKER_MODEL,
    rerank_top_k=config.RERANK_TOP_K,
    # Phase 2 (v2) settings
    enable_hybrid_search=config.ENABLE_HYBRID_SEARCH,
    hybrid_alpha=config.HYBRID_ALPHA,
    bm25_index_path=config.BM25_INDEX_PATH,
)


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
    query: str, context_chunks:list[str], config: RAGConfig
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
            f"{config.ollama_base_url}/api/generate",
            json={
                "model": config.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "num_predict": config.max_tokens,
                },
            },
            timeout=config.request_timeout,
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
) -> tuple[faiss.Index, list[MetadataDict], SentenceTransformer]:
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
    if not config.index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {config.index_path}")
    if not config.db_path.exists():
        raise FileNotFoundError(f"Database file not found: {config.db_path}")

    try:
        # Load FAISS index
        index = faiss.read_index(str(config.index_path))

        # Configure IVF index if applicable
        if hasattr(index, "nprobe"):
            # This is an IVF index, set nprobe for search
            nprobe = getattr(config, "NPROBE", 16)  # Default to 16 if not set
            index.nprobe = nprobe 
            print(f"Set IVF nprobe to {nprobe}")

        # Load metadata from database
        db = sqlite_utils.Database(str(config.db_path))
        metadata: list[MetadataDict] = []
        if "chunks" in db.table_names():
            for row in db["chunks"].rows:
                metadata.append({
                    "source": str(row["source"]),
                    "chunk_index": int(row["chunk_index"]),
                    "text": str(row["text"])
                })
        else:
            raise ValueError("Database exists but contains no chunks table")

        # Load embedder
        embedder = SentenceTransformer(config.embed_model)

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
    metadata: list[MetadataDict],
    embedder: SentenceTransformer,
    config: RAGConfig,
    top_k: int | None = None,
    show_sources: bool = True,
    debug: bool = True,
) -> str | None:
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
        top_k = config.default_top_k

    print(f"\nQuery: {query}")

    # Phase 1 Feature: Multi-Query Expansion
    queries_to_search = [query]
    if config.enable_query_expansion:
        print("🔄 Expanding query into multiple variants...")
        import config as cfg_module
        queries_to_search = expand_query(
            query,
            cfg_module.OLLAMA_BASE_URL,
            cfg_module.OLLAMA_MODEL,
            num_variants=cfg_module.QUERY_EXPANSION_COUNT,
            timeout=30,
        )
        if len(queries_to_search) > 1:
            print(f"   Generated {len(queries_to_search)} query variants:")
            for i, q in enumerate(queries_to_search):
                print(f"   {i+1}. {q}")

    # Determine retrieval count based on reranking
    enable_reranking = config.enable_reranking
    import config as cfg_module
    retrieval_k = cfg_module.RERANK_TOP_K if enable_reranking else top_k

    try:
        # Collect results from all query variants
        all_results: dict[int, SearchResult] = {}  # Use dict to dedupe by index

        for query_variant in queries_to_search:
            with _memory_cleanup():
                # Embed the query variant
                query_emb = embedder.encode(
                    [query_variant], convert_to_numpy=True, normalize_embeddings=False
                )
                # Convert to numpy array with proper type
                query_emb_array = np.asarray(query_emb, dtype=np.float32)
                query_emb_normalized = _l2_normalize(query_emb_array)

                # Search FAISS - returns (distances, labels)
                distances, labels = index.search(query_emb_normalized, retrieval_k) 
                # For cosine similarity (IndexFlatIP), distances are actually scores
                scores = distances
                indices = labels

            # Collect results from this variant
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for missing results
                    continue

                if idx >= len(metadata):
                    print(f"⚠️ Invalid index {idx} (metadata has {len(metadata)} entries)")
                    continue

                meta = metadata[idx]

                # Keep highest score if we've seen this chunk before
                if idx not in all_results or score > all_results[idx].score:
                    all_results[idx] = SearchResult(
                        source=meta["source"],
                        chunk_idx=meta["chunk_index"],
                        score=float(score),
                        text=meta["text"],
                    )

        # Convert dict to list
        sources_info = list(all_results.values())

        # Sort by FAISS score initially
        sources_info.sort(key=lambda x: x.score, reverse=True)

        if debug and config.enable_query_expansion and len(queries_to_search) > 1:
            print(f"\n📊 Retrieved {len(sources_info)} unique chunks from all variants")

        # Phase 2 Feature: Hybrid BM25 + FAISS Score Fusion
        if config.enable_hybrid_search:
            bm25_index = _get_bm25_index()
            if bm25_index is not None:
                print("🔀 Fusing FAISS semantic scores with BM25 keyword scores...")
                try:
                    # Query BM25 index (use original query, not variants)
                    bm25_results = bm25_index.search(query, top_k=retrieval_k)

                    if debug:
                        print(f"   BM25 returned {len(bm25_results)} keyword matches")

                    # Build BM25 score map: metadata_index -> bm25_score
                    from utils.bm25_index import normalize_bm25_scores
                    bm25_normalized = normalize_bm25_scores(bm25_results)
                    bm25_score_map = {idx: score for idx, score in bm25_normalized}

                    # Fuse scores: need to match FAISS results with BM25 by metadata index
                    # The challenge: sources_info came from all_results dict keyed by idx
                    # We need to rebuild that mapping

                    # Rebuild metadata index for each FAISS result
                    alpha = config.hybrid_alpha
                    fused_results = []

                    for faiss_result in sources_info:
                        # Find this result's metadata index
                        # Match by source + chunk_idx to find metadata index
                        meta_idx = None
                        for idx, meta in enumerate(metadata):
                            if (meta["source"] == faiss_result.source and
                                meta["chunk_index"] == faiss_result.chunk_idx):
                                meta_idx = idx
                                break

                        # Get scores
                        faiss_score = faiss_result.score
                        # Normalize FAISS cosine similarity [0.5, 1.0] → [0.0, 1.0]
                        faiss_normalized = max(0.0, min(1.0, (faiss_score - 0.5) * 2.0))

                        bm25_score = bm25_score_map.get(meta_idx, 0.0) if meta_idx is not None else 0.0

                        # Fuse scores
                        fused_score = alpha * faiss_normalized + (1 - alpha) * bm25_score

                        # Create fused result - copy and update scores
                        fused = faiss_result.model_copy()
                        fused.faiss_score = faiss_score
                        fused.bm25_score = bm25_score
                        fused.fused_score = fused_score
                        fused.score = fused_score

                        fused_results.append(fused)

                    # Sort by fused score
                    def _get_fused_score(result: SearchResult) -> float:
                        return result.fused_score if result.fused_score is not None else 0.0

                    fused_results.sort(key=_get_fused_score, reverse=True)
                    sources_info = fused_results

                    if debug:
                        print(f"   Fused with alpha={alpha} (semantic={alpha}, keyword={1-alpha})")

                except Exception as e:
                    print(f"⚠️ Hybrid fusion failed: {e}, using FAISS-only results")
                    import logging
                    logging.getLogger(__name__).error(f"Hybrid fusion error: {e}", exc_info=True)

        # Phase 1 Feature: Cross-Encoder Reranking
        if enable_reranking and sources_info:
            print(f"🎯 Reranking top {len(sources_info)} results...")
            reranker = _get_reranker()
            sources_info = reranker.rerank(query, sources_info, top_k=top_k)
            if debug:
                print(f"   After reranking, using top {len(sources_info)} results")

        # Extract context chunks for LLM
        context_chunks = [result.text for result in sources_info]

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
                source_path = Path(info.source)
                print(
                    f"{i+1}. {source_path.name} (chunk {info.chunk_idx}, score: {info.score:.3f})"
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

    Note:
        With Pydantic models, most validation is automatic.
        This function provides additional runtime checks.
    """
    # Pydantic already validates types and constraints
    # Additional validations can be added here if needed
    if config.default_top_k <= 0:
        raise ValueError("default_top_k must be positive")

    if config.request_timeout <= 0:
        raise ValueError("request_timeout must be positive")

    # Paths are already Path objects from config


def main(config: RAGConfig | None = None) -> None:
    """Main function to run the RAG query system.

    Args:
        config: Optional RAG configuration. Uses defaults if not provided.
    """
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
