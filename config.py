# ======================================
# Config for ingestion script
# Default: tuned for 16GB RAM / 6GB VRAM laptop
# ======================================

from typing import Set, Literal

# Type definitions
IndexType = Literal["flat", "ivf_flat", "ivf_pq"]

# CPU / Threading
TORCH_NUM_THREADS: int = 2
OPENBLAS_NUM_THREADS: int = 2
MKL_NUM_THREADS: int = 2
OMP_NUM_THREADS: int = 2
NUMEXPR_MAX_THREADS: int = 2

# GPU / CUDA
CUDA_VISIBLE_DEVICES: str = "0"  # single GPU
PYTORCH_CUDA_ALLOC_CONF: str = "max_split_size_mb:1024,garbage_collection_threshold:0.9"

# Sentence-Transformers
BATCH_SIZE: int = 16  # safe for 6GB GTX 1660 Ti

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
SKIP_FILES: Set[str] = {
    "UNC - International Biosafety Committee Meeting Minutes 2019 - Redacted.pdf"
}


def validate_config() -> None:
    """Validate configuration values at startup."""
    # Thread counts should be positive
    thread_configs = [
        TORCH_NUM_THREADS,
        OPENBLAS_NUM_THREADS,
        MKL_NUM_THREADS,
        OMP_NUM_THREADS,
        NUMEXPR_MAX_THREADS,
        BATCH_SIZE,
        NLIST,
        NPROBE,
    ]

    for config_val in thread_configs:
        if not isinstance(config_val, int) or config_val <= 0:
            raise ValueError(
                f"Thread/batch config values must be positive integers, got: {config_val}"
            )

    # Index type validation
    valid_index_types = {"flat", "ivf_flat", "ivf_pq"}
    if INDEX_TYPE not in valid_index_types:
        raise ValueError(
            f"INDEX_TYPE must be one of {valid_index_types}, got: {INDEX_TYPE}"
        )

    # CUDA device validation (basic)
    if not isinstance(CUDA_VISIBLE_DEVICES, str):
        raise ValueError("CUDA_VISIBLE_DEVICES must be a string")

    # Skip files validation
    if not isinstance(SKIP_FILES, set):
        raise ValueError("SKIP_FILES must be a set")


# Validate on import
validate_config()
