"""Utilities for checking Ollama availability before running queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import requests


@dataclass(frozen=True)
class OllamaStatus:
    """Simple container for Ollama model availability."""

    base_url: str
    available_models: frozenset[str]


class OllamaUnavailableError(RuntimeError):
    """Raised when Ollama is unreachable or missing the requested model."""


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _fetch_available_models(base_url: str, timeout: float) -> Iterable[str]:
    response = requests.get(f"{base_url}/api/tags", timeout=timeout)
    if response.status_code != 200:
        raise OllamaUnavailableError(
            f"Ollama at {base_url} responded with HTTP {response.status_code} "
            "when fetching model list."
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise OllamaUnavailableError(
            "Failed to parse Ollama /api/tags response as JSON."
        ) from exc

    models = payload.get("models", [])
    return (entry.get("name", "") for entry in models)


def ensure_ollama_model_available(
    base_url: str, model: str, timeout: float = 5.0
) -> OllamaStatus:
    """Verify Ollama is reachable and the desired model is installed."""
    normalized_url = _normalize_base_url(base_url)
    try:
        model_names = frozenset(
            name for name in _fetch_available_models(normalized_url, timeout) if name
        )
    except requests.exceptions.ConnectionError as exc:
        raise OllamaUnavailableError(
            f"Ollama is not reachable at {normalized_url}. "
            "Start it with `ollama serve`."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise OllamaUnavailableError(
            f"Ollama did not respond within {timeout} seconds. "
            "Ensure it is running and reachable."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise OllamaUnavailableError(
            f"Ollama request failed: {exc}"
        ) from exc

    if model not in model_names:
        raise OllamaUnavailableError(
            f"Ollama model '{model}' is not installed. "
            "Install it with `ollama pull {model}` or update OLLAMA_MODEL in settings."
        )

    return OllamaStatus(base_url=normalized_url, available_models=model_names)


__all__ = [
    "OllamaStatus",
    "OllamaUnavailableError",
    "ensure_ollama_model_available",
]
