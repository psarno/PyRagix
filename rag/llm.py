"""LLM interaction helpers."""

from collections import OrderedDict
from pathlib import Path
from textwrap import dedent
from typing import Any

import requests
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from types_models import RAGConfig, SearchResult


class _RetryableOllamaError(RuntimeError):
    """Exception raised when Ollama responds with a retryable status code."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


@retry(
    retry=retry_if_exception_type(
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            _RetryableOllamaError,
        )
    ),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    reraise=True,
)
def _post_with_retry(
    url: str, payload: dict[str, Any], timeout: int
) -> requests.Response:
    """Issue a POST request to Ollama, retrying on transient failures."""
    response = requests.post(url, json=payload, timeout=timeout)

    if response.status_code in {429, 500, 502, 503, 504}:
        raise _RetryableOllamaError(
            response.status_code,
            f"Ollama responded with HTTP {response.status_code}.",
        )

    return response


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

    payload: dict[str, Any] = {
        "model": config.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "num_predict": config.max_tokens,
        },
    }

    url = f"{config.ollama_base_url}/api/generate"

    try:
        response = _post_with_retry(url, payload, config.request_timeout)
    except RetryError as retry_exc:
        last_error = retry_exc.last_attempt.exception()
        if isinstance(last_error, requests.exceptions.ConnectionError):
            return "WARNING: Ollama is not running. Please start Ollama first with: ollama serve"
        if isinstance(last_error, requests.exceptions.Timeout):
            return "WARNING: Request timed out. The model might be loading or the query is too complex."
        if isinstance(last_error, _RetryableOllamaError):
            return (
                "WARNING: Ollama API kept returning server errors "
                f"(HTTP {last_error.status_code}). Check the model or server logs."
            )
        if isinstance(last_error, requests.exceptions.RequestException):
            return f"WARNING: Request error: {last_error}"
        return f"WARNING: Unexpected error generating answer: {last_error}"
    except requests.exceptions.ConnectionError:
        return "WARNING: Ollama is not running. Please start Ollama first with: ollama serve"
    except requests.exceptions.Timeout:
        return "WARNING: Request timed out. The model might be loading or the query is too complex."
    except requests.exceptions.RequestException as exc:
        return f"WARNING: Request error: {exc}"

    try:
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "Sorry, I couldn't generate an answer.")

        if response.status_code == 404:
            return (
                "WARNING: Ollama could not find the requested model. "
                f"Install it with `ollama pull {config.ollama_model}`."
            )

        return f"Error calling Ollama API: {response.status_code}"

    except (KeyError, ValueError) as exc:
        return f"WARNING: Configuration or response parsing error: {exc}"
    except Exception as exc:
        return f"WARNING: Unexpected error generating answer: {exc}"


__all__ = ["generate_answer_with_ollama"]
