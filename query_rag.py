import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from typing import List, Dict

# Version for debugging
VERSION = "0.0.3"

# Use same settings as ingest
EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "local_faiss.index"
META_PATH = "documents.pkl"

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b-instruct-q4_0"


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def generate_answer_with_ollama(query: str, context_chunks: List[str]) -> str:
    """Generate a human-like answer using Ollama based on retrieved context"""

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
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "top_p": 0.9, "max_tokens": 500},
            },
            timeout=60,
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
    except Exception as e:
        return f"WARNING: Error generating answer: {str(e)}"


def load_rag_system():
    """Load the FAISS index and metadata"""
    print("Loading FAISS index and metadata...")

    # Load FAISS index
    index = faiss.read_index(INDEX_PATH)

    # Load metadata
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    # Load embedder
    embedder = SentenceTransformer(EMBED_MODEL)

    print(
        f"Loaded {index.ntotal} chunks from {len(set(m['source'] for m in metadata))} files"
    )
    return index, metadata, embedder


def query_rag(
    query: str, index, metadata, embedder, top_k=7, show_sources=True, debug=True
):
    """Query the RAG system and generate a human-like answer"""
    print(f"\nQuery: {query}")

    # Embed the query
    query_emb = embedder.encode(
        [query], convert_to_numpy=True, normalize_embeddings=False
    )
    query_emb = query_emb.astype("float32")
    query_emb = l2_normalize(query_emb)

    # Search FAISS
    scores, indices = index.search(query_emb, top_k)

    # Collect relevant chunks
    context_chunks = []
    sources_info = []

    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx == -1:  # FAISS returns -1 for missing results
            continue

        meta = metadata[idx]
        source = meta["source"]
        chunk_idx = meta["chunk_index"]
        text = meta["text"]

        context_chunks.append(text)
        sources_info.append({"source": source, "chunk_idx": chunk_idx, "score": score})

    if not context_chunks:
        print("\nNo relevant documents found.")
        return

    # Debug: show what chunks are being sent
    if debug:
        print(f"\nSending {len(context_chunks)} chunks to LLM:")
        for i, chunk in enumerate(context_chunks[:2]):  # Show first 2
            print(f"  Chunk {i+1} (len={len(chunk)}): {repr(chunk[:100])}...")
        print()

    # Generate answer using Ollama
    print(f"\n🤖 Generating answer...")
    answer = generate_answer_with_ollama(query, context_chunks)

    print(f"\nAnswer:")
    print("=" * 60)
    print(answer)
    print("=" * 60)

    # Optionally show sources
    if show_sources:
        print(f"\nSources:")
        for i, info in enumerate(sources_info):
            print(
                f"{i+1}. {info['source']} (chunk {info['chunk_idx']}, score: {info['score']:.3f})"
            )
        print("-" * 60)


def main():
    try:
        index, metadata, embedder = load_rag_system()

        print(f"\nRAG Query System Ready! (Version {VERSION})")
        print("Type your questions (or 'quit' to exit)")

        while True:
            query = input("\nQuery: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break

            if not query:
                continue

            query_rag(query, index, metadata, embedder, top_k=7)

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Make sure you've run ingest_folder.py first!")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
