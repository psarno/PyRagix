"""LLM interaction helpers."""

from pathlib import Path
from textwrap import dedent
from collections import OrderedDict

import requests

from types_models import RAGConfig, SearchResult


def generate_answer_with_ollama(
    query: str,
    search_results: list[SearchResult],
    config: RAGConfig,
) -> str:
    """Call the Ollama HTTP API with contextualized prompt including metadata."""
    formatted_chunks: list[str] = []
    doc_index_map: "OrderedDict[str, int]" = OrderedDict()
    for result in search_results:
        source_name = Path(result.source).name
        if source_name not in doc_index_map:
            doc_index_map[source_name] = len(doc_index_map) + 1

        doc_id = doc_index_map[source_name]
        metadata_header = (
            f"[Document {doc_id} — Source: {source_name} | "
            f"Chunk {result.chunk_idx + 1}/{result.total_chunks} | "
            f"Type: {result.file_type.upper()}]"
        )
        formatted_chunks.append(f"{metadata_header}\n{result.text}")

    context = "\n\n".join(formatted_chunks)
    legend = "\n".join(
        f"- Document {doc_id}: {source_name}"
        for source_name, doc_id in doc_index_map.items()
    )

    prompt = dedent(
        f"""Analyze these excerpts to answer the question. Use ONLY the provided information.

        DOCUMENT REFERENCE:
        {legend}

        EXCERPTS:
        {context}

        QUESTION: {query}

        Instructions:
        - Review ALL excerpts for relevant information
        - Synthesize information across multiple documents when appropriate
        - Clearly cite sources using the form `Document X (FileName)`
        - Quote specific passages verbatim when they materially support the answer
        - If no relevant information exists, respond exactly: "No information found in documents"

        Response:"""
    )

    try:
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

        return f"Error calling Ollama API: {response.status_code}"

    except requests.exceptions.ConnectionError:
        return "WARNING: Ollama is not running. Please start Ollama first with: ollama serve"
    except requests.exceptions.Timeout:
        return "WARNING: Request timed out. The model might be loading or the query is too complex."
    except requests.exceptions.RequestException as exc:
        return f"WARNING: Request error: {exc}"
    except (KeyError, ValueError) as exc:
        return f"WARNING: Configuration or response parsing error: {exc}"
    except Exception as exc:
        return f"WARNING: Unexpected error generating answer: {exc}"


__all__ = ["generate_answer_with_ollama"]
