# PyRagix Python TODO

## Open Tasks
- Harden ingestion error handling: extend the existing `tenacity` retry/backoff patterns beyond Ollama calls to cover extraction, embedding, and FAISS/BM25 refresh steps so ingestion failures are automatically retried or surfaced with clear diagnostics.
- Expand edge-case test coverage: backfill pytest integration coverage for zero-chunk queries, corrupt or oversized PDFs, simulated Ollama timeouts, and ingestion I/O failures to ensure the recent resilience work stays green.
- Incremental ingestion: persist chunk-level SHA256 hashes (or similar) and compare them before re-embedding so unchanged documents skipped during refreshes without reprocessing.
- Configuration validation: layer stricter schema validation on top of TOML parsing (e.g., Pydantic models for settings) to catch misconfigurations before pipelines spin up.

## Recent Progress
- 2025-11-05 · Config validation at startup now checks FAISS, BM25, metadata DB, and embedding paths before serving (`rag/configuration.py`, `query_rag.py`).
- 2025-11-05 · Ollama query calls wrapped with `tenacity` retries and clearer failure paths (`rag/llm.py`, `pyproject.toml`).
- 2025-11-05 · Hybrid retrieval now tunes alpha dynamically by query length with accompanying tests (`rag/retrieval.py`, `tests/test_retrieval_dynamic_alpha.py`).
- 2025-11-05 · Added pytest coverage around config validation expectations (`tests/test_rag_configuration.py`).
- 2025-11-05 · README now includes query and ingestion pipeline diagrams for quicker onboarding (`README.md`).
