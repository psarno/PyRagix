# ======================================
# RAG Web Server
# - FastAPI server for RAG query interface  
# - REST API endpoints for web frontend
# - CORS enabled for local development
# ======================================

# ===============================
# Standard Library
# ===============================
import sys
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

# ===============================
# Third-party Libraries
# ===============================
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# ===============================
# Local Imports
# ===============================
from query_rag import (
    _load_rag_system,
    _query_rag,
    _validate_config,
    DEFAULT_CONFIG,
    RAGConfig,
    SearchResult,
)


# ===============================
# Request/Response Models
# ===============================
class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str
    top_k: Optional[int] = None
    show_sources: bool = True
    debug: bool = False


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str
    sources: List[SearchResult]
    query: str
    top_k: int
    success: bool
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    rag_loaded: bool
    total_documents: int
    index_type: str


# ===============================
# Global State
# ===============================
rag_system = {
    "index": None,
    "metadata": None, 
    "embedder": None,
    "config": None,
    "loaded": False,
}


# ===============================
# Startup/Shutdown Handlers
# ===============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load RAG system on startup, cleanup on shutdown."""
    # Startup
    try:
        print("🚀 Loading RAG system...")
        config = DEFAULT_CONFIG.copy()
        _validate_config(config)
        
        index, metadata, embedder = _load_rag_system(config)
        
        rag_system["index"] = index
        rag_system["metadata"] = metadata
        rag_system["embedder"] = embedder
        rag_system["config"] = config
        rag_system["loaded"] = True
        
        print("✅ RAG system loaded successfully!")
        
    except Exception as e:
        print(f"❌ Failed to load RAG system: {e}")
        print("Make sure you've run ingest_folder.py first!")
        # Don't exit - let the server start but mark as not loaded
        
    yield
    
    # Shutdown
    print("🔄 Shutting down RAG system...")
    rag_system.clear()


# ===============================
# FastAPI App
# ===============================
app = FastAPI(
    title="RAG Query API",
    description="REST API for Retrieval-Augmented Generation queries",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===============================
# API Endpoints
# ===============================
@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect to the web interface."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta http-equiv="refresh" content="0; url=/web/">
    </head>
    <body>
        <p>Redirecting to <a href="/web/">RAG Web Interface</a>...</p>
    </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if not rag_system["loaded"]:
        return HealthResponse(
            status="error",
            version="0.1.0",
            rag_loaded=False,
            total_documents=0,
            index_type="none"
        )
    
    index = rag_system["index"]
    metadata = rag_system["metadata"]
    
    index_type = "IVF" if hasattr(index, "nprobe") else "Flat"
    
    return HealthResponse(
        status="healthy",
        version="0.1.0", 
        rag_loaded=True,
        total_documents=len(metadata),
        index_type=index_type
    )


@app.post("/query", response_model=QueryResponse)
async def query_rag_endpoint(request: QueryRequest):
    """Query the RAG system."""
    if not rag_system["loaded"]:
        raise HTTPException(
            status_code=503,
            detail="RAG system not loaded. Make sure you've run ingest_folder.py first!"
        )
    
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )
    
    try:
        # Get top_k from request or use default
        top_k = request.top_k or rag_system["config"]["default_top_k"]
        
        # Perform RAG query (modified to return sources)
        answer, sources = _query_rag_with_sources(
            query=request.query,
            index=rag_system["index"],
            metadata=rag_system["metadata"],
            embedder=rag_system["embedder"],
            config=rag_system["config"],
            top_k=top_k,
            show_sources=request.show_sources,
            debug=request.debug
        )
        
        if answer is None:
            return QueryResponse(
                answer="No relevant documents found for your query.",
                sources=[],
                query=request.query,
                top_k=top_k,
                success=False,
                error="No relevant documents found"
            )
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            query=request.query, 
            top_k=top_k,
            success=True
        )
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        if request.debug:
            error_msg += f"\n{traceback.format_exc()}"
            
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )


def _query_rag_with_sources(
    query: str,
    index,
    metadata: List[Dict[str, Any]], 
    embedder,
    config: RAGConfig,
    top_k: int,
    show_sources: bool = True,
    debug: bool = False
) -> tuple[Optional[str], List[SearchResult]]:
    """Modified version of _query_rag that returns sources separately."""
    from query_rag import _memory_cleanup, _l2_normalize, _generate_answer_with_ollama
    import numpy as np
    
    if not query.strip():
        return None, []
    
    try:
        with _memory_cleanup():
            # Embed the query
            query_emb = embedder.encode(
                [query], convert_to_numpy=True, normalize_embeddings=False
            )
            query_emb_array = np.asarray(query_emb, dtype=np.float32)
            query_emb_normalized = _l2_normalize(query_emb_array)

            # Search FAISS
            distances, labels = index.search(query_emb_normalized, top_k)
            scores = distances
            indices = labels

        # Collect relevant chunks and sources
        context_chunks: List[str] = []
        sources_info: List[SearchResult] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(metadata):
                continue

            meta = metadata[idx]
            source = meta["source"]
            chunk_idx = meta["chunk_index"]  
            text = meta["text"]

            context_chunks.append(text)
            sources_info.append({
                "source": source,
                "chunk_idx": chunk_idx,
                "score": float(score),
                "text": text,
            })

        if not context_chunks:
            return None, []

        # Generate answer using Ollama
        answer = _generate_answer_with_ollama(query, context_chunks, config)
        
        return answer, sources_info

    except Exception as e:
        if debug:
            print(f"❌ Error during query processing: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
        return None, []


# ===============================
# Static Files & Web Interface
# ===============================
# Mount static files for web interface (will be created next)
web_dir = Path("web")
if web_dir.exists():
    app.mount("/web", StaticFiles(directory="web", html=True), name="web")


# ===============================
# Server Entry Point  
# ===============================
def main():
    """Run the FastAPI server."""
    print("🌐 Starting RAG Web Server...")
    print("📖 Web interface will be available at: http://localhost:8000/web/")
    print("🔍 API docs available at: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        access_log=True
    )


if __name__ == "__main__":
    main()