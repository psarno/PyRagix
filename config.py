# ======================================
# Config for PyRagix
# Loads from settings.toml (read-only, TOML format)
# Default: tuned for 16GB RAM / 6GB VRAM laptop
# ======================================

import tomllib
import warnings
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, field_validator

warnings.filterwarnings(
    "ignore",
    message=r"builtin type (SwigPy|swigvarlink).*__module__ attribute",
    category=DeprecationWarning,
)

# Type definitions
IndexType = Literal["flat", "ivf", "ivf_pq"]


class PyRagixConfig(BaseModel):
    """PyRagix configuration schema with validation.

    Loaded from settings.toml. Pydantic handles type coercion and validation.
    """

    # CPU / Threading
    TORCH_NUM_THREADS: int = 6
    OPENBLAS_NUM_THREADS: int = 6
    MKL_NUM_THREADS: int = 6
    OMP_NUM_THREADS: int = 6
    NUMEXPR_MAX_THREADS: int = 6

    # GPU / CUDA
    CUDA_VISIBLE_DEVICES: str = "0"
    PYTORCH_ALLOC_CONF: str = "max_split_size_mb:1024,garbage_collection_threshold:0.9"
    FAISS_DISABLE_CPU: str = "1"
    CUDA_LAUNCH_BLOCKING: str = "0"

    # FAISS GPU settings (advanced users only)
    GPU_ENABLED: bool = False
    GPU_DEVICE: int = 0
    GPU_MEMORY_FRACTION: float = Field(default=0.8, ge=0.0, le=1.0)

    # Embeddings
    BATCH_SIZE: int = Field(default=16, gt=0)
    EMBED_MODEL: str = "all-MiniLM-L6-v2"

    # FAISS
    INDEX_TYPE: IndexType = "flat"
    NLIST: int = Field(default=1024, gt=0)
    NPROBE: int = Field(default=16, gt=0)

    # PDF Processing
    BASE_DPI: int = Field(default=150, gt=0)
    BATCH_SIZE_RETRY_DIVISOR: int = Field(default=4, gt=0)
    SKIP_FILES: list[str] = Field(default_factory=list)

    # File paths and logging
    INGESTION_LOG_FILE: str = "ingestion.log"
    CRASH_LOG_FILE: str = "crash_log.txt"

    # Ollama / LLM
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b"
    REQUEST_TIMEOUT: int = Field(default=180, gt=0)
    TEMPERATURE: float = Field(default=0.1, ge=0.0, le=2.0)
    TOP_P: float = Field(default=0.9, ge=0.0, le=1.0)
    MAX_TOKENS: int = Field(default=500, gt=0)

    # Retrieval
    DEFAULT_TOP_K: int = Field(default=7, gt=0)

    # Phase 1: Query Expansion
    ENABLE_QUERY_EXPANSION: bool = True
    QUERY_EXPANSION_COUNT: int = Field(default=3, gt=0)

    # Phase 1: Reranking
    ENABLE_RERANKING: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_TOP_K: int = Field(default=20, gt=0)

    # Phase 2: Hybrid Search
    ENABLE_HYBRID_SEARCH: bool = True
    HYBRID_ALPHA: float = Field(default=0.7, ge=0.0, le=1.0)
    BM25_INDEX_PATH: str = "bm25_index.pkl"

    # Phase 3: Semantic Chunking
    ENABLE_SEMANTIC_CHUNKING: bool = True
    SEMANTIC_CHUNK_MAX_SIZE: int = Field(default=1600, ge=100)
    SEMANTIC_CHUNK_OVERLAP: int = Field(default=200, ge=0)

    @field_validator("INDEX_TYPE", mode="before")
    @classmethod
    def _normalize_index_type(cls, value: Any) -> str:
        if isinstance(value, str):
            normalized = value.lower()
            if normalized == "ivf_flat":
                normalized = "ivf"
            if normalized in {"flat", "ivf", "ivf_pq"}:
                return normalized
        raise ValueError("INDEX_TYPE must be one of: flat, ivf, ivf_pq")


# Settings file paths
SETTINGS_FILE = "settings.toml"
SETTINGS_EXAMPLE_FILE = "settings.example.toml"


def _create_settings_from_example() -> None:
    """Create settings.toml from settings.example.toml on first run.

    This copies the example file to the actual settings file so users get
    all the documented defaults and comments without checking it into git.
    """
    example_path = Path(SETTINGS_EXAMPLE_FILE)
    settings_path = Path(SETTINGS_FILE)

    if not example_path.exists():
        print(
            f"Warning: {SETTINGS_EXAMPLE_FILE} not found. Cannot create {SETTINGS_FILE}."
        )
        print("Please ensure settings.example.toml exists in the project directory.")
        return

    try:
        # Copy example file to actual settings file
        with open(example_path, "rb") as src:
            content = src.read()
        with open(settings_path, "wb") as dst:
            _ = dst.write(content)
        print(f"Created {SETTINGS_FILE} from {SETTINGS_EXAMPLE_FILE}")
    except OSError as e:
        print(f"Warning: Could not create {SETTINGS_FILE}: {e}")


def _load_settings() -> PyRagixConfig:
    """Load settings from TOML file, auto-creating from example on first run.

    Uses Pydantic for validation and type coercion. Fails fast on bad data.

    Returns:
        PyRagixConfig: Validated configuration object
    """
    settings_path = Path(SETTINGS_FILE)

    # Create from example if doesn't exist
    if not settings_path.exists():
        _create_settings_from_example()

    # Load and parse TOML (fail fast on errors)
    with open(settings_path, "rb") as f:
        toml_data = tomllib.load(f)

    # Flatten TOML sections into flat dict (settings.toml has [section] structure)
    # Pydantic will handle all type validation and coercion
    flat_settings: dict[str, Any] = {}
    for section_data in toml_data.values():
        # TOML sections are always dicts - cast from tomllib's Any type
        section_dict = cast(dict[str, Any], section_data)
        flat_settings.update(section_dict)

    # Use Pydantic to validate and coerce types (handles defaults + validation)
    return PyRagixConfig.model_validate(flat_settings)


# Load settings and create module-level variables
_config = _load_settings()

# Export all configuration variables at module level (with full type inference!)
TORCH_NUM_THREADS: int = _config.TORCH_NUM_THREADS
OPENBLAS_NUM_THREADS: int = _config.OPENBLAS_NUM_THREADS
MKL_NUM_THREADS: int = _config.MKL_NUM_THREADS
OMP_NUM_THREADS: int = _config.OMP_NUM_THREADS
NUMEXPR_MAX_THREADS: int = _config.NUMEXPR_MAX_THREADS

CUDA_VISIBLE_DEVICES: str = _config.CUDA_VISIBLE_DEVICES
PYTORCH_ALLOC_CONF: str = _config.PYTORCH_ALLOC_CONF
FAISS_DISABLE_CPU: str = _config.FAISS_DISABLE_CPU
CUDA_LAUNCH_BLOCKING: str = _config.CUDA_LAUNCH_BLOCKING

BATCH_SIZE: int = _config.BATCH_SIZE
EMBED_MODEL: str = _config.EMBED_MODEL

INDEX_TYPE: IndexType = _config.INDEX_TYPE
NLIST: int = _config.NLIST
NPROBE: int = _config.NPROBE

# FAISS GPU Settings (advanced users only)
GPU_ENABLED: bool = _config.GPU_ENABLED
GPU_DEVICE: int = _config.GPU_DEVICE
GPU_MEMORY_FRACTION: float = _config.GPU_MEMORY_FRACTION

SKIP_FILES: set[str] = set(_config.SKIP_FILES)

BASE_DPI: int = _config.BASE_DPI
BATCH_SIZE_RETRY_DIVISOR: int = _config.BATCH_SIZE_RETRY_DIVISOR

INGESTION_LOG_FILE: str = _config.INGESTION_LOG_FILE
CRASH_LOG_FILE: str = _config.CRASH_LOG_FILE

OLLAMA_BASE_URL: str = _config.OLLAMA_BASE_URL
OLLAMA_MODEL: str = _config.OLLAMA_MODEL
DEFAULT_TOP_K: int = _config.DEFAULT_TOP_K
REQUEST_TIMEOUT: int = _config.REQUEST_TIMEOUT
TEMPERATURE: float = _config.TEMPERATURE
TOP_P: float = _config.TOP_P
MAX_TOKENS: int = _config.MAX_TOKENS

# Phase 1: Query Expansion (v2)
ENABLE_QUERY_EXPANSION: bool = _config.ENABLE_QUERY_EXPANSION
QUERY_EXPANSION_COUNT: int = _config.QUERY_EXPANSION_COUNT

# Phase 1: Reranking (v2)
ENABLE_RERANKING: bool = _config.ENABLE_RERANKING
RERANKER_MODEL: str = _config.RERANKER_MODEL
RERANK_TOP_K: int = _config.RERANK_TOP_K

# Phase 2: Hybrid Search (v2)
ENABLE_HYBRID_SEARCH: bool = _config.ENABLE_HYBRID_SEARCH
HYBRID_ALPHA: float = _config.HYBRID_ALPHA
BM25_INDEX_PATH: str = _config.BM25_INDEX_PATH

# Phase 3: Semantic Chunking (v2)
ENABLE_SEMANTIC_CHUNKING: bool = _config.ENABLE_SEMANTIC_CHUNKING
SEMANTIC_CHUNK_MAX_SIZE: int = _config.SEMANTIC_CHUNK_MAX_SIZE
SEMANTIC_CHUNK_OVERLAP: int = _config.SEMANTIC_CHUNK_OVERLAP

# Expose the full validated settings object for typed access.
CONFIG: PyRagixConfig = _config
