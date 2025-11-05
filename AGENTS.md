# PyRagix – Agent Notes

## Scope
- Local-first RAG engine implemented in Python 3.13+, mirrors the .NET port in `../pyragix-net`.
- Pipelines: document ingestion → hybrid retrieval (FAISS + BM25) → cross-encoder rerank → Ollama answer generation.
- Primary entry points: `ingest_folder.py` (CLI ingestion) and `query_rag.py` (CLI querying).

## Repo Layout
- `ingestion/` – extraction pipeline (PyMuPDF/PaddleOCR), chunking, embedding, FAISS/BM25 management, metadata persistence.
- `rag/` – query-time modules: configuration, retrieval orchestration, reranking, Ollama interaction.
- `utils/` & `typings/` – shared helpers, FAISS/BM25 utilities, strict type stubs for third-party libs.
- `tests/` – pytest suite covering config validation, ingestion helpers, retrieval, and progress reporting.
- `web/` – TypeScript/React client + FastAPI endpoint (generated via `dev.sh` as needed).
- Root scripts: `ingest_folder.py`, `query_rag.py`, `dev.sh`, `settings.toml`.

## Build & Execute
- Install dependencies once with `uv sync` (respects `uv.lock`).
- Run ingestion from repo root to keep relative paths aligned: `uv run python ingest_folder.py ./docs` (append `--fresh` to rebuild indices).
- Query locally with `uv run python query_rag.py "your question here"`.
- Execute tests via `uv run pytest` (or target modules, e.g., `uv run pytest tests/test_file_scanner.py`).
- Static analysis stays green with `uv run pyright --strict`.

## Configuration & Assets
- Working configuration: `settings.toml` (copy/edit from `settings.example.toml`).
- Generated artifacts live alongside config defaults:
  - `local_faiss.index` – FAISS vectors (IVF or flat based on settings).
  - `bm25_index.pkl` – BM25 keyword index (optional, controlled by `ENABLE_HYBRID_SEARCH`).
  - `documents.db` – SQLite metadata backing retrieval.
  - `processed_files.txt` – SHA256 ledger to skip unchanged files.
- OCR assets: PaddleOCR model cache must be reachable; see `ingestion/environment.py` for expected layout.
- Crash diagnostics land in `ingestion.log` and `crash.log`; leave them intact for users.

## Pipeline Highlights
- Ingestion (`ingestion/pipeline.py`): `EnvironmentManager` applies runtime guards → `FileScanner` discovers documents, applies skip rules, and coordinates extraction (`DocumentExtractor`), chunking (`Chunker`), embeddings, FAISS persistence, and BM25 refresh.
- Retry/resilience: tenacity-backed wrappers for extraction, embedding, FAISS saving, and BM25 rebuilds; CUDA OOM retries shrink batch size via config.
- Retrieval (`rag/retrieval.py`): loads metadata, performs FAISS search, optional BM25 fusion with dynamic alpha, reranks via cross-encoder, and forwards top chunks to Ollama (`rag/llm.py`).
- Configuration (`rag/configuration.py`, `types_models.py`): Pydantic models validate TOML with strict typing; hybrid parameters stay in sync across pipelines.

## External Expectations
- Ollama must be reachable at `ollama_base_url` in `settings.toml`; both query expansion and answer generation hit `/api/generate`.
- FAISS GPU support is optional; enable via config and ensure NVIDIA drivers are present before toggling.
- PaddleOCR can leverage GPU when configured; ingestion will fall back gracefully but warn when models/memos are missing.
- Web UI assumes API server runs from repo root; ensure relative paths to indexes/database remain correct.

## Known Gaps / TODO Hooks
- See `todo-python.md` for prioritized tasks (ingestion resilience, edge-case tests, incremental ingestion, config validation hardening).
- Incremental ingestion currently relies on `processed_files.txt`; chunk-level hashing is planned but not yet implemented.
- Web UI environment still evolving; verify API docs when adjusting FastAPI schemas.
- Parity with `../pyragix-net` is ongoing—port shared resilience features in both directions.

## Collaboration Tips
- Keep strict typing: maintain `pyright --strict` with zero suppressions; extend stubs in `typings/` instead of adding `type: ignore`.
- Update documentation (`README.md`, `docs/`) alongside pipeline or config changes; CLI help strings should match README examples.
- For long-running scripts, prefer structured logging via `logging` over bare `print` (ingestion CLI already wires rotating handlers).
- Sync breaking changes with the .NET port when feasible (shared configs, retry semantics, metadata schema).

### Commenting style
- Provide module-, class-, or function-level docstrings summarizing purpose, side effects, and expected inputs/outputs when behavior is non-trivial.
- Add concise inline comments before complex control flow, tricky numerical thresholds, data-shape assumptions, or cross-module invariants (e.g., FAISS/BM25 coordination, retry backoffs).
- Avoid restating obvious code; focus comments on intent, rationale, external system expectations, and temporary workarounds.
- When a bug fix depends on specific config or model behavior, capture that context in a brief comment to guide future maintainers.
