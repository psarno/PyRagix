"""
Multi-Query Expansion Module

Generates multiple query variants using LLM to improve retrieval recall.
Inspired by production RAG systems that expand fuzzy user intents into
multiple precise search queries.

Example:
    "climate impact on farms" →
    [
        "global warming agriculture effects",
        "environmental changes farming industry",
        "climate change agricultural impact"
    ]
"""

import requests
import logging

logger = logging.getLogger(__name__)


def expand_query(
    query: str,
    ollama_base_url: str,
    ollama_model: str,
    num_variants: int = 3,
    timeout: int = 30,
) ->list[str]:
    """Generate multiple query variants for improved retrieval recall.

    Args:
        query: Original user query
        ollama_base_url: Ollama API base URL
        ollama_model: Ollama model name
        num_variants: Number of query variants to generate
        timeout: Request timeout in seconds

    Returns:
        List of query variants (includes original query)
    """
    if not query.strip():
        return [query]

    # Craft prompt for query expansion
    prompt = f"""Generate {num_variants} diverse rephrased variants of this search query. Each variant should express the same intent using different words, synonyms, or perspectives.

Original query: {query}

Requirements:
- Keep variants concise (similar length to original)
- Use different terminology and phrasing
- Maintain the original search intent
- Output only the variants, one per line
- No numbering, bullets, or explanations

Variants:"""

    try:
        # Call Ollama API
        response = requests.post(
            f"{ollama_base_url}/api/generate",
            json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,  # Higher temp for diversity
                    "top_p": 0.9,
                    "num_predict": 150,  # Short responses
                },
            },
            timeout=timeout,
        )

        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("response", "").strip()

            # Parse variants from response
            variants: list[str] = []
            for line in generated_text.split("\n"):
                line = line.strip()
                # Clean up common LLM artifacts (numbering, bullets)
                line = line.lstrip("0123456789.-•* ")
                if line and len(line) > 5:  # Ignore very short lines
                    variants.append(line)

            # Always include original query
            all_variants = [query] + variants[:num_variants]

            logger.info(f"Expanded query into {len(all_variants)} variants")
            logger.debug(f"Variants: {all_variants}")

            return all_variants

        else:
            logger.warning(
                f"Query expansion failed (HTTP {response.status_code}), using original query"
            )
            return [query]

    except requests.exceptions.ConnectionError:
        logger.warning(
            "Query expansion failed: Ollama not running, using original query"
        )
        return [query]
    except requests.exceptions.Timeout:
        logger.warning("Query expansion timed out, using original query")
        return [query]
    except Exception as e:
        logger.warning(f"Query expansion error: {e}, using original query")
        return [query]
