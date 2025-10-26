"""LLM interaction helpers."""

from __future__ import annotations
from textwrap import dedent

import requests

from types_models import RAGConfig


def generate_answer_with_ollama(
    query: str,
    context_chunks: list[str],
    config: RAGConfig,
) -> str:
    """Call the Ollama HTTP API with contextualized prompt."""
    context = "\n\n".join(
        [f"Document {i + 1}: {chunk}" for i, chunk in enumerate(context_chunks)]
    )

    prompt = dedent(f"""Analyze these documents to answer the question comprehensively. Use ONLY what is written in the documents.

                        DOCUMENTS:
                        {context}

                        QUESTION: {query}

                        Instructions:
                        - Review ALL documents above for relevant information
                        - Synthesize information across multiple documents if available
                        - Provide a comprehensive answer based on patterns you see
                        - Quote specific examples from the documents
                        - If no relevant information exists, respond: "No information found in documents"

                        Response:""")

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
    except Exception as exc:  # pragma: no cover - defensive
        return f"WARNING: Unexpected error generating answer: {exc}"


__all__ = ["generate_answer_with_ollama"]
