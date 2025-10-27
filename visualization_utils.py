# ======================================
# RAG Visualization Utilities  
# - Dimensionality reduction for embedding visualization
# - FAISS index embedding extraction
# - Data preparation for frontend visualization
# ======================================

# ===============================
# Standard Library
# ===============================
from __future__ import annotations

from typing import Any, Sequence

# ===============================
# Third-party Libraries
# ===============================
import numpy as np
import numpy.typing as npt
import faiss
from sklearn.manifold import TSNE
import umap

from rag.embeddings import memory_cleanup
from types_models import MetadataDict

PointDict = dict[str, Any]
FloatArray = npt.NDArray[np.float32]
EmbeddingArray = npt.NDArray[np.floating[Any]]

# ===============================
# FAISS Embedding Extraction
# ===============================
def _extract_faiss_embeddings(index: faiss.Index, max_points: int = 1000) -> FloatArray:
    """Extract embeddings from FAISS index for visualization."""
    try:
        if hasattr(index, 'reconstruct_n'):
            # IVF indices - reconstruct vectors
            total_vectors = index.ntotal
            
            if total_vectors > max_points:
                indices = np.linspace(0, total_vectors - 1, max_points, dtype=int)
            else:
                indices = np.arange(total_vectors)
            
            reconstructed_vectors: list[FloatArray] = []
            for i in indices:
                try:
                    vector: FloatArray = np.zeros(index.d, dtype=np.float32)
                    index.reconstruct(int(i), vector)
                    reconstructed_vectors.append(vector)
                except RuntimeError:
                    continue
                    
            if reconstructed_vectors:
                stacked_vectors = np.stack(reconstructed_vectors).astype(np.float32)
                return stacked_vectors
            else:
                raise ValueError("Could not reconstruct vectors from IVF index")
                
        elif hasattr(index, 'xb') and getattr(index, 'xb', None) is not None:
            # Flat indices - direct access  
            embeddings_raw = faiss.vector_to_array(getattr(index, 'xb'))
            embedding_dim = index.d
            embeddings_array: FloatArray = (
                embeddings_raw.reshape(-1, embedding_dim).astype(np.float32)
            )
            
            if len(embeddings_array) > max_points:
                indices = np.linspace(0, len(embeddings_array) - 1, max_points, dtype=int)
                embeddings_array = embeddings_array[indices]
                
            return embeddings_array
        else:
            raise ValueError(f"Unsupported FAISS index type: {type(index)}")
            
    except Exception as e:
        print(f"❌ Error extracting embeddings: {e}")
        raise


# ===============================
# Dimensionality Reduction
# ===============================
def _apply_dimensionality_reduction(
    embeddings: EmbeddingArray,
    method: str = "umap",
    dimensions: int = 2,
    random_state: int = 42
) -> EmbeddingArray:
    """Apply UMAP or t-SNE dimensionality reduction."""

    if method.lower() == "umap":
        reducer = umap.UMAP(
            n_components=dimensions,
            random_state=random_state,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine'
        )
    elif method.lower() == "tsne":
        reducer = TSNE(
            n_components=dimensions,
            random_state=random_state,
            perplexity=min(30, len(embeddings) - 1),
            max_iter=1000
        )
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    try:
        with memory_cleanup():
            return reducer.fit_transform(embeddings)
    except Exception as e:
        print(f"❌ Error in {method}: {e}")
        raise


# ===============================
# Main Visualization Function
# ===============================
def create_embedding_visualization(
    query: str,
    query_embedding: EmbeddingArray,
    index: faiss.Index,
    metadata: Sequence[MetadataDict],
    retrieved_indices: Sequence[int],
    scores: Sequence[float],
    method: str = "umap",
    dimensions: int = 2,
    max_points: int = 1000
) -> dict[str, Any]:
    """Create visualization data for RAG embeddings."""
    try:
        # Extract document embeddings 
        doc_embeddings = _extract_faiss_embeddings(index, max_points - 1)
        
        # Combine query + documents
        all_embeddings = np.vstack([query_embedding, doc_embeddings])
        
        # Apply dimensionality reduction
        reduced_embeddings = _apply_dimensionality_reduction(all_embeddings, method, dimensions)
        
        # Prepare points for frontend
        points: list[PointDict] = []
        
        # Query point (first)
        query_coords = reduced_embeddings[0]
        points.append({
            "id": 0,
            "x": float(query_coords[0]),
            "y": float(query_coords[1]),
            "z": float(query_coords[2]) if dimensions == 3 else None,
            "source": "[QUERY]",
            "chunk_idx": 0,
            "score": 1.0,
            "text": query,
            "is_query": True
        })
        
        # Document points
        retrieved_scores = {
            int(idx): float(score) for idx, score in zip(retrieved_indices, scores)
        }
        
        for i, coords in enumerate(reduced_embeddings[1:], 1):
            meta_idx = min(i - 1, len(metadata) - 1)
            
            if meta_idx < len(metadata):
                meta = metadata[meta_idx]
                score = retrieved_scores.get(meta_idx, 0.0)
                text = meta.text

                points.append({
                    "id": i,
                    "x": float(coords[0]),
                    "y": float(coords[1]),
                    "z": float(coords[2]) if dimensions == 3 else None,
                    "source": meta.source,
                    "chunk_idx": meta.chunk_index,
                    "score": float(score),
                    "text": text[:200] + "..." if len(text) > 200 else text,
                    "is_query": False
                })
        
        return {
            "points": points,
            "query": query,
            "method": method,
            "dimensions": dimensions,
            "total_points": len(points),
            "retrieved_count": len(retrieved_indices)
        }
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        return {
            "points": [],
            "query": query,
            "method": method, 
            "dimensions": dimensions,
            "total_points": 0,
            "retrieved_count": 0,
            "error": str(e)
        }
