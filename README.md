<img width="500" height="auto" alt="image" src="https://github.com/user-attachments/assets/851dda34-8152-4436-80f9-7d792e676f5b" />

# PyRagix

Local-first RAG system based on modern retrieval research - query expansion, cross-encoder reranking, hybrid search (FAISS + BM25), and semantic chunking. Runs entirely on your machine via Ollama. No cloud APIs, no data leaving your network.

Also available as a .NET port: [pyragix-net](https://github.com/psarno/pyragix-net)

![Python](https://img.shields.io/badge/python-3.13+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Architecture

PyRagix implements a multi-stage retrieval pipeline.

**Query Pipeline:**
```
User Query
  ↓
Multi-Query Expansion (3-5 variants via local LLM)
  ↓
Hybrid Search (FAISS semantic 70% + BM25 keyword 30%)
  ↓
Cross-Encoder Reranking (top-20 → top-7 by relevance)
  ↓
Answer Generation (local Ollama LLM)
```

**Ingestion Pipeline:**
```
Document Input (PDF, HTML, Images)
  ↓
Text Extraction (PyMuPDF, BeautifulSoup, PaddleOCR)
  ↓
Semantic Chunking (sentence-boundary aware)
  ↓
Embedding Generation (local sentence-transformers)
  ↓
Dual Indexing (FAISS vector + BM25 keyword)
```

Query expansion helps with recall on vague or paraphrased questions. Reranking filters out keyword-matched junk. Hybrid search handles structured queries (names, dates, IDs) that pure semantic search misses.

## Features

- **Query expansion** - generates multiple query variants via the local LLM to improve recall
- **Cross-encoder reranking** - re-scores retrieved chunks with a dedicated relevance model
- **Hybrid search** - FAISS semantic search + BM25 keyword matching, weighted and fused
- **Semantic chunking** - splits at sentence boundaries instead of fixed character counts
- **Multi-format ingestion** - PDF, HTML, and images (via PaddleOCR)
- **Incremental updates** - add documents without reprocessing the whole corpus
- **Web UI and console interface** - FastAPI backend with a TypeScript frontend, or use the CLI
- **Runs on Windows, Linux, and macOS**

## Type Safety & Architecture

The entire codebase passes `pyright --strict` with zero errors and zero `# type: ignore` comments. Python 3.13+ syntax throughout (`X | None`, `list[T]`, `dict[K, V]`).

Third-party C++ libraries (FAISS, PyMuPDF, PaddleOCR) are typed via Protocols and custom stubs:

```python
# ingestion/models.py
class PDFPage(Protocol):
    """Protocol for PyMuPDF Page objects."""
    def get_text(self, option: str) -> str: ...
    def get_pixmap(self, dpi: int) -> PDFPixmap: ...
```

Additional stubs for faiss, paddleocr, rank_bm25, sqlite_utils, and others live in `typings/`.

All config and data models use Pydantic v2:

```python
# types_models.py
class MetadataDict(BaseModel):
    model_config = ConfigDict(frozen=True, validate_assignment=True)

    source: str
    chunk_index: int = Field(ge=0)
    total_chunks: int
    file_type: str
```

The codebase is split into three packages with explicit boundaries:

```python
from ingestion import (
    FAISSManager,      # Vector index management
    FileScanner,       # Document discovery
    MetadataStore,     # SQLite operations
    TextProcessor,     # Extraction pipeline
)

from rag import (
    RAGConfig,         # Configuration
    load_models,       # Model initialization
    hybrid_search,     # Multi-stage retrieval
    generate_answer,   # LLM generation
)

from utils import (
    BM25Index,         # Keyword search
    QueryExpander,     # Query rewriting
    Reranker,          # Cross-encoder scoring
)
```

## Quick Start

### Prerequisites

1. **Python 3.13+** with uv package manager (recommended) or pip
2. **Ollama** for local LLM inference - download from [ollama.com](https://ollama.com)
3. **8GB+ RAM** (16GB+ recommended for optimal performance)

### Installation

```bash
# Clone repository
git clone https://github.com/psarno/PyRagix.git
cd PyRagix

# Install dependencies with uv (recommended - fast and reliable)
uv sync

# Or with pip (installs from pyproject.toml)
pip install -e .

# Pull Ollama model for local LLM
ollama pull qwen2.5:7b
ollama serve
```

### Basic Usage

```bash
# Ingest documents (builds FAISS + BM25 indexes)
uv run python ingest_folder.py --fresh ./docs

# Start web interface (compiles TypeScript frontend and starts server)
./dev.sh
# Open http://localhost:8000/web/

# Or use console interface
uv run python query_rag.py
```

## Configuration

PyRagix uses `settings.toml` for all configuration. The file is auto-generated with optimal defaults for your system on first run. A template is available at `settings.example.toml`.

All RAG features are off by default. Turn them on as needed:

```toml
[query_expansion]
ENABLE_QUERY_EXPANSION = true
QUERY_EXPANSION_COUNT = 3

[reranking]
ENABLE_RERANKING = true
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K = 20

[hybrid_search]
ENABLE_HYBRID_SEARCH = true
HYBRID_ALPHA = 0.7          # 70% semantic, 30% keyword

[semantic_chunking]
ENABLE_SEMANTIC_CHUNKING = true
SEMANTIC_CHUNK_MAX_SIZE = 1600
SEMANTIC_CHUNK_OVERLAP = 200
```

- **Query expansion** generates variant phrasings of your query. Helps most with vague or ambiguous questions. `QUERY_EXPANSION_COUNT` controls how many variants (default 3).
- **Reranking** re-scores the top candidates with a cross-encoder. Filters out chunks that matched on keywords but aren't actually relevant. `RERANK_TOP_K` sets the candidate pool (default 20).
- **Hybrid search** fuses FAISS and BM25 results. Mostly useful for structured queries (names, dates, IDs) that pure vector search misses. `HYBRID_ALPHA` controls the weight split.
- **Semantic chunking** splits at sentence boundaries instead of fixed character counts. Better context preservation.

Enabling everything adds a few hundred ms per query, which is small compared to LLM generation time.

### Hardware Tuning

For memory-constrained systems (8-12GB RAM):
```toml
[embeddings]
BATCH_SIZE = 8

[threading]
TORCH_NUM_THREADS = 4

[pdf]
BASE_DPI = 100
```

For high-performance systems (32GB+ RAM):
```toml
[embeddings]
BATCH_SIZE = 32

[threading]
TORCH_NUM_THREADS = 12

[pdf]
BASE_DPI = 200

[faiss]
NLIST = 2048
NPROBE = 32
```

### LLM Configuration

Customize Ollama model and generation parameters:
```toml
[llm]
OLLAMA_MODEL = "qwen2.5:7b"
TEMPERATURE = 0.1
TOP_P = 0.9
MAX_TOKENS = 500
REQUEST_TIMEOUT = 180

[retrieval]
DEFAULT_TOP_K = 7
```
## Advanced Usage

### Incremental Ingestion

Add new documents without reprocessing:
```bash
# Initial ingestion
uv run python ingest_folder.py ./docs

# Later: add more documents (automatically skips processed files)
uv run python ingest_folder.py ./more_docs
```

### Custom Document Filters

Skip specific file types or patterns:
```toml
[pdf]
SKIP_FILES = ["*.tmp", "backup_*", "archive/*"]
```

### FAISS Index Optimization

PyRagix uses IVF (Inverted File) indexing by default for fast search on large corpora:

```toml
[faiss]
INDEX_TYPE = "ivf"
NLIST = 1024
NPROBE = 16
```

- **NLIST**: Number of clusters (default: 1024). Increase for larger datasets (10k+ chunks).
- **NPROBE**: Search clusters (default: 16). Higher values improve recall at the cost of speed.

The system automatically falls back to flat indexing for small collections (< 2048 chunks), then upgrades to IVF as your corpus grows.

### GPU Acceleration

GPU is auto-detected with CPU fallback:

```toml
[gpu]
GPU_ENABLED = true
GPU_DEVICE = 0
GPU_MEMORY_FRACTION = 0.8
```

GPU FAISS requires separate installation. CPU-only FAISS works fine and is the default.

## Project Structure

```
PyRagix/
├── ingest_folder.py        # Document ingestion CLI (thin wrapper)
├── query_rag.py           # Console query CLI (thin wrapper)
├── web_server.py          # FastAPI web server
├── dev.sh                 # Development script (compiles TypeScript + starts server)
├── config.py              # Configuration management
├── settings.toml          # User configuration (auto-generated, TOML format)
├── settings.example.toml  # Configuration template
├── types_models.py        # Shared Pydantic models (MetadataDict, etc.)
│
├── ingestion/             # Document Processing Pipeline (11 modules)
│   ├── __init__.py        # Package exports
│   ├── cli.py             # CLI argument parsing
│   ├── environment.py     # Environment setup (torch, GPU detection)
│   ├── faiss_manager.py   # FAISS index management (IVF, flat)
│   ├── file_filters.py    # File type detection and filtering
│   ├── file_scanner.py    # Recursive document discovery
│   ├── metadata_store.py  # SQLite metadata database
│   ├── models.py          # Protocol definitions (PDFPage, OCRProcessorProtocol, etc.)
│   ├── pipeline.py        # Main ingestion orchestration
│   ├── stale_cleaner.py   # Remove outdated chunks
│   └── text_processing.py # Text extraction (PDF, HTML, OCR)
│
├── rag/                   # Query Pipeline (5 modules)
│   ├── __init__.py        # Package exports
│   ├── configuration.py   # RAGConfig Pydantic model
│   ├── embeddings.py      # Embedding model initialization
│   ├── llm.py             # Ollama LLM client
│   ├── loader.py          # FAISS/BM25 index loading
│   └── retrieval.py       # Multi-stage retrieval (hybrid, rerank)
│
├── utils/                 # RAG Utilities (3 modules)
│   ├── __init__.py        # Package exports
│   ├── bm25_index.py      # BM25 keyword search
│   ├── query_expander.py  # Multi-query expansion via LLM
│   └── reranker.py        # Cross-encoder reranking
│
├── classes/               # Core Processing Classes
│   ├── ProcessingConfig.py # Ingestion configuration dataclass
│   └── OCRProcessor.py     # PaddleOCR wrapper
│
├── typings/               # Type Stubs for Third-Party Libraries
│   ├── faiss/             # FAISS C++ bindings
│   ├── fitz/              # PyMuPDF (fitz)
│   ├── paddleocr/         # PaddleOCR
│   ├── rank_bm25/         # BM25 library
│   ├── sklearn/           # scikit-learn
│   ├── sqlite_utils/      # SQLite utilities
│   ├── lxml/              # XML/HTML parser
│   └── umap/              # UMAP dimensionality reduction
│
├── tests/                 # Pytest Test Suite
│   ├── conftest.py        # Shared fixtures (temp dirs, mocks)
│   ├── test_config.py     # Configuration validation tests
│   ├── test_environment.py # Environment setup tests
│   ├── test_faiss_manager.py # FAISS indexing tests
│   ├── test_file_filters.py # File type detection tests
│   ├── test_file_scanner.py # Document discovery tests
│   └── test_text_processing.py # Text extraction tests
│
├── web/                   # Web Interface (TypeScript)
│   ├── index.html         # Main UI page
│   ├── style.css          # Responsive styling
│   ├── script.ts          # TypeScript source (type-safe API client)
│   └── tsconfig.json      # TypeScript configuration (compile with dev.sh)
│
├── pyrightconfig.json     # Pyright strict type checking config
└── uv.lock               # Dependency lock file
```

## Dependencies

Managed via `pyproject.toml`. Requires Python 3.13+.

**Core ML/AI:**
- **torch** (2.9+): Embedding model backend with CUDA support
- **sentence-transformers**: Dense embeddings and cross-encoder reranking
- **transformers**: HuggingFace model infrastructure
- **faiss-cpu** (1.12+): High-performance vector search with IVF indexing
- **rank-bm25**: BM25 keyword search for hybrid retrieval

**Document Processing:**
- **paddleocr**: OCR for images and scanned documents
- **paddlepaddle** (3.2+): PaddleOCR backend
- **pymupdf**: PDF text extraction
- **beautifulsoup4**: HTML parsing
- **langchain-text-splitters**: Semantic chunking with sentence boundaries
- **pillow**: Image processing

**Data & Infrastructure:**
- **fastapi**: Web API and UI server
- **uvicorn**: ASGI server with WebSockets
- **sqlite-utils**: Metadata database management
- **pydantic**: Data validation and settings management
- **numpy**: Numerical operations

**Utilities:**
- **scikit-learn**: ML utilities (used by reranker)
- **umap-learn**: Dimensionality reduction (visualization)
- **psutil**: System resource monitoring
- **requests**: HTTP client

**Development Tools:**
- **pyright**: Strict static type checking
- **ruff**: Fast Python linter and formatter
- **pytest**: Testing framework

**Installation:**
```bash
# Recommended: Use uv for fast, reliable dependency management
uv sync

# Alternative: Traditional pip installation
pip install -e .

# Development dependencies
uv sync --dev
```

All dependencies are pinned to minimum versions.

## CI/CD

GitHub Actions runs `pyright --strict`, `ruff`, and `pytest` on every push and PR.

## Contributing

Contributions are welcome.

**Development Setup:**
```bash
git clone https://github.com/psarno/PyRagix.git
cd PyRagix
uv sync
```

**Rules:**
- All code must pass `pyright --strict` with zero errors
- No `# type: ignore` - use stubs or `cast()` instead
- No `Any` types except for legitimate sentinel values and validators
- Modern syntax: `X | None`, `list[T]`, `dict[K, V]` (not `Optional`, `List`, `Dict`)
- Pydantic v2 for data models, Protocols for third-party library interfaces
- Tests for new features using fixtures from `tests/conftest.py`

**Workflow:**
```bash
# Type check (must pass before committing)
uv run pyright

# Run tests
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .
```

If you're adding a new third-party library feature, update the type stubs. Look at `ingestion/` and `rag/` for the existing patterns.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgements

Built on FAISS, Sentence Transformers, Ollama, PaddleOCR, and LangChain.
