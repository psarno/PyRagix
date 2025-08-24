# ======================================
# Config for ingestion script
# Default: tuned for 16GB RAM / 6GB VRAM laptop
# ======================================

from typing import Set, Literal

# Type definitions
IndexType = Literal["flat", "ivf_flat", "ivf_pq"]

# CPU / Threading
TORCH_NUM_THREADS: int = 4
OPENBLAS_NUM_THREADS: int = 4
MKL_NUM_THREADS: int = 4
OMP_NUM_THREADS: int = 4
NUMEXPR_MAX_THREADS: int = 4

# GPU / CUDA
CUDA_VISIBLE_DEVICES: str = "0"  # single GPU
PYTORCH_CUDA_ALLOC_CONF: str = "max_split_size_mb:1024,garbage_collection_threshold:0.9"

# Sentence-Transformers
BATCH_SIZE: int = 16

# FAISS

# Under ~50k–100k chunks
INDEX_TYPE: IndexType = "flat"  # "ivf_flat", "ivf_pq", or "flat"
NLIST: int = 1024
NPROBE: int = 16

# If currently beyond 100k chunks, use:
# INDEX_TYPE = "ivf_flat"
# NLIST = 1024
# NPROBE = 16

# Files to skip during processing (by filename)
SKIP_FILES: Set[str] = set()

# PDF Processing settings
BASE_DPI: int = 150  # Base DPI for PDF page rendering
BATCH_SIZE_RETRY_DIVISOR: int = 4  # Divisor for reducing batch size on memory errors

# File paths and logging
INGESTION_LOG_FILE: str = "ingestion.log"
CRASH_LOG_FILE: str = "crash_log.txt"

# RAG/Query settings
OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_MODEL: str = "llama3.1:8b-instruct-q4_0"
DEFAULT_TOP_K: int = 7
REQUEST_TIMEOUT: int = 60
TEMPERATURE: float = 0.1
TOP_P: float = 0.9
MAX_TOKENS: int = 500


def validate_config() -> None:
    """Validate configuration values at startup."""
    # Integer configs that should be positive
    positive_int_configs = [
        ("TORCH_NUM_THREADS", TORCH_NUM_THREADS),
        ("OPENBLAS_NUM_THREADS", OPENBLAS_NUM_THREADS),
        ("MKL_NUM_THREADS", MKL_NUM_THREADS),
        ("OMP_NUM_THREADS", OMP_NUM_THREADS),
        ("NUMEXPR_MAX_THREADS", NUMEXPR_MAX_THREADS),
        ("BATCH_SIZE", BATCH_SIZE),
        ("NLIST", NLIST),
        ("NPROBE", NPROBE),
        ("BASE_DPI", BASE_DPI),
        ("BATCH_SIZE_RETRY_DIVISOR", BATCH_SIZE_RETRY_DIVISOR),
        ("DEFAULT_TOP_K", DEFAULT_TOP_K),
        ("REQUEST_TIMEOUT", REQUEST_TIMEOUT),
        ("MAX_TOKENS", MAX_TOKENS),
    ]

    for config_name, config_val in positive_int_configs:
        if not isinstance(config_val, int) or config_val <= 0:
            raise ValueError(
                f"{config_name} must be a positive integer, got: {config_val}"
            )

    # Float configs with valid ranges
    if not isinstance(TEMPERATURE, (int, float)) or not (0.0 <= TEMPERATURE <= 2.0):
        raise ValueError(
            f"TEMPERATURE must be a float between 0.0 and 2.0, got: {TEMPERATURE}"
        )

    if not isinstance(TOP_P, (int, float)) or not (0.0 <= TOP_P <= 1.0):
        raise ValueError(f"TOP_P must be a float between 0.0 and 1.0, got: {TOP_P}")

    # String configs
    string_configs = [
        ("CUDA_VISIBLE_DEVICES", CUDA_VISIBLE_DEVICES),
        ("INGESTION_LOG_FILE", INGESTION_LOG_FILE),
        ("CRASH_LOG_FILE", CRASH_LOG_FILE),
        ("OLLAMA_BASE_URL", OLLAMA_BASE_URL),
        ("OLLAMA_MODEL", OLLAMA_MODEL),
    ]

    for config_name, config_val in string_configs:
        if not isinstance(config_val, str) or not config_val.strip():
            raise ValueError(
                f"{config_name} must be a non-empty string, got: {config_val}"
            )

    # Index type validation
    valid_index_types = {"flat", "ivf_flat", "ivf_pq"}
    if INDEX_TYPE not in valid_index_types:
        raise ValueError(
            f"INDEX_TYPE must be one of {valid_index_types}, got: {INDEX_TYPE}"
        )

    # Skip files validation
    if not isinstance(SKIP_FILES, set):
        raise ValueError("SKIP_FILES must be a set")


# Validate on import
validate_config()
