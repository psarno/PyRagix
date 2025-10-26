"""Query pipeline: retrieval, hybrid fusion, reranking, and generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer

import config
from rag.embeddings import (
    FloatArray,
    get_sentence_encoder,
    l2_normalize,
    memory_cleanup,
)
from rag.llm import generate_answer_with_ollama
from types_models import MetadataDict, RAGConfig, SearchResult
from utils.query_expander import expand_query
from utils.reranker import Reranker

if TYPE_CHECKING:
    import faiss

_reranker: Reranker | None = None
_bm25_index: Any | None = None  # utils.bm25_index.BM25Index


def _get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker(model_name=config.RERANKER_MODEL)
    return _reranker


def _get_bm25_index() -> Any | None:
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
        except Exception as exc:
            print(f"⚠️ Failed to load BM25 index: {exc}")

    return _bm25_index


def query_rag(
    query: str,
    index: "faiss.Index",
    metadata: list[MetadataDict],
    embedder: SentenceTransformer,
    config_obj: RAGConfig,
    top_k: int | None = None,
    show_sources: bool = True,
    debug: bool = True,
) -> str | None:
    """Execute the end-to-end Retrieval-Augmented Generation flow."""
    if not query.strip():
        print("⚠️ Empty query provided")
        return None

    effective_top_k = top_k or config_obj.default_top_k

    print(f"\nQuery: {query}")

    queries_to_search = [query]
    if config_obj.enable_query_expansion:
        print("🔄 Expanding query into multiple variants...")
        queries_to_search = expand_query(
            query,
            config_obj.ollama_base_url,
            config_obj.ollama_model,
            num_variants=config_obj.query_expansion_count,
            timeout=30,
        )
        if len(queries_to_search) > 1:
            print(f"   Generated {len(queries_to_search)} query variants:")
            for i, variant in enumerate(queries_to_search, start=1):
                print(f"   {i}. {variant}")

    enable_reranking = config_obj.enable_reranking
    retrieval_k = config_obj.rerank_top_k if enable_reranking else effective_top_k

    try:
        encode_queries = get_sentence_encoder(embedder)
        all_results: dict[int, SearchResult] = {}

        for query_variant in queries_to_search:
            with memory_cleanup():
                query_emb: FloatArray = encode_queries(
                    [query_variant],
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                )
                query_emb_array: FloatArray = np.asarray(query_emb, dtype=np.float32)
                query_emb_normalized = l2_normalize(query_emb_array)

                distances, labels = index.search(query_emb_normalized, retrieval_k)
                scores = distances
                indices = labels

            for score_raw, idx_raw in zip(scores[0], indices[0]):
                idx = int(idx_raw)
                if idx == -1:
                    continue

                if idx >= len(metadata):
                    print(f"⚠️ Invalid index {idx} (metadata has {len(metadata)} entries)")
                    continue

                meta = metadata[idx]
                score = float(score_raw)

                if idx not in all_results or score > all_results[idx].score:
                    all_results[idx] = SearchResult(
                        source=meta["source"],
                        chunk_idx=meta["chunk_index"],
                        score=score,
                        text=meta["text"],
                    )

        sources_info: list[SearchResult] = list(all_results.values())
        sources_info.sort(key=lambda x: x.score, reverse=True)

        if debug and config_obj.enable_query_expansion and len(queries_to_search) > 1:
            print(f"\n📊 Retrieved {len(sources_info)} unique chunks from all variants")

        if config_obj.enable_hybrid_search:
            bm25_index = _get_bm25_index()
            if bm25_index is not None:
                print("🔀 Fusing FAISS semantic scores with BM25 keyword scores...")
                try:
                    from utils.bm25_index import normalize_bm25_scores

                    bm25_results = bm25_index.search(query, top_k=retrieval_k)
                    if debug:
                        print(f"   BM25 returned {len(bm25_results)} keyword matches")

                    bm25_normalized = normalize_bm25_scores(bm25_results)
                    bm25_score_map: dict[int, float] = {
                        int(idx): float(score) for idx, score in bm25_normalized
                    }

                    fused_results: list[SearchResult] = []
                    alpha = config_obj.hybrid_alpha

                    for faiss_result in sources_info:
                        meta_idx: int | None = None
                        for idx, meta in enumerate(metadata):
                            if (
                                meta["source"] == faiss_result.source
                                and meta["chunk_index"] == faiss_result.chunk_idx
                            ):
                                meta_idx = idx
                                break

                        faiss_score = faiss_result.score
                        faiss_normalized = max(
                            0.0, min(1.0, (faiss_score - 0.5) * 2.0)
                        )
                        bm25_score = (
                            bm25_score_map.get(meta_idx, 0.0)
                            if meta_idx is not None
                            else 0.0
                        )
                        fused_score = alpha * faiss_normalized + (1 - alpha) * bm25_score

                        fused = faiss_result.model_copy()
                        fused.faiss_score = faiss_score
                        fused.bm25_score = bm25_score
                        fused.fused_score = fused_score
                        fused.score = fused_score
                        fused_results.append(fused)

                    fused_results.sort(
                        key=lambda result: result.fused_score or 0.0,
                        reverse=True,
                    )
                    sources_info = fused_results

                    if debug:
                        print(
                            f"   Fused with alpha={alpha} "
                            f"(semantic={alpha}, keyword={1 - alpha})"
                        )

                except Exception as exc:
                    print(f"⚠️ Hybrid fusion failed: {exc}, using FAISS-only results")
                    import logging

                    logging.getLogger(__name__).error(
                        f"Hybrid fusion error: {exc}", exc_info=True
                    )

        if enable_reranking and sources_info:
            print(f"🎯 Reranking top {len(sources_info)} results...")
            reranker = _get_reranker()
            sources_info = reranker.rerank(query, sources_info, top_k=effective_top_k)
            if debug:
                print(f"   After reranking, using top {len(sources_info)} results")

        context_chunks = [result.text for result in sources_info]

        if not context_chunks:
            print("\nNo relevant documents found.")
            return None

        if debug:
            print(f"\nSending {len(context_chunks)} chunks to LLM:")
            for i, chunk in enumerate(context_chunks[:2], start=1):
                print(f"  Chunk {i} (len={len(chunk)}): {repr(chunk[:100])}...")
            print()

        print("🤖 Generating answer...")
        answer = generate_answer_with_ollama(query, context_chunks, config_obj)

        print("\nAnswer:")
        print("=" * 60)
        print(answer)
        print("=" * 60)

        if show_sources:
            print("\nSources:")
            for i, info in enumerate(sources_info, start=1):
                source_path = Path(info.source)
                print(
                    f"{i}. {source_path.name} "
                    f"(chunk {info.chunk_idx}, score: {info.score:.3f})"
                )
            print("-" * 60)

        return answer

    except Exception as exc:
        print(f"❌ Error during query processing: {type(exc).__name__}: {exc}")
        if debug:
            import traceback

            traceback.print_exc()
        return None


__all__ = ["query_rag"]
