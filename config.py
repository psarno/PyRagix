# ======================================
# Config for ingestion script
# Default: tuned for 16GB RAM / 6GB VRAM laptop
# ======================================

import json
from pathlib import Path
from typing import Set, Literal, Dict, Any

# Type definitions
IndexType = Literal["flat", "ivf_flat", "ivf_pq"]

# Default configuration values
_DEFAULT_CONFIG = {
    # CPU / Threading
    "TORCH_NUM_THREADS": 6,
    "OPENBLAS_NUM_THREADS": 6,
    "MKL_NUM_THREADS": 6,
    "OMP_NUM_THREADS": 6,
    "NUMEXPR_MAX_THREADS": 6,
    # GPU / CUDA
    "CUDA_VISIBLE_DEVICES": "0",  # single GPU
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:1024,garbage_collection_threshold:0.9",
    # FAISS GPU environment
    "FAISS_DISABLE_CPU": "1",  # Force FAISS to prefer GPU
    "CUDA_LAUNCH_BLOCKING": "0",  # Allow async CUDA operations
    # Sentence-Transformers
    "BATCH_SIZE": 16,
    # FAISS
    "INDEX_TYPE": "flat",  # "ivf_flat", "ivf_pq", or "flat"
    "NLIST": 1024,
    "NPROBE": 16,
    # FAISS GPU settings (advanced users only - requires conda-forge faiss-gpu)
    "GPU_ENABLED": False,  # Enable FAISS GPU acceleration (most users should leave False)
    "GPU_DEVICE": 0,  # GPU device ID
    "GPU_MEMORY_FRACTION": 0.8,  # Fraction of GPU memory to use
    # Files to skip during processing (by filename)
    "SKIP_FILES": [],
    # PDF Processing settings
    "BASE_DPI": 150,  # Base DPI for PDF page rendering
    "BATCH_SIZE_RETRY_DIVISOR": 4,  # Divisor for reducing batch size on memory errors
    # File paths and logging
    "INGESTION_LOG_FILE": "ingestion.log",
    "CRASH_LOG_FILE": "crash_log.txt",
    # RAG/Query settings
    "EMBED_MODEL": "all-MiniLM-L6-v2",
    "OLLAMA_BASE_URL": "http://localhost:11434",
    "OLLAMA_MODEL": "qwen2.5:7b",
    "DEFAULT_TOP_K": 3,
    "REQUEST_TIMEOUT": 90,
    "TEMPERATURE": 0.1,
    "TOP_P": 0.9,
    "MAX_TOKENS": 500,
}

# Settings file path
SETTINGS_FILE = "settings.json"


def _load_settings() -> Dict[str, Any]:
    """Load settings from JSON file, creating it with defaults if it doesn't exist.

    Returns:
        Dict[str, Any]: Configuration settings
    """
    settings_path = Path(SETTINGS_FILE)

    if settings_path.exists():
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                user_settings = json.load(f)

            # Convert SKIP_FILES list to set if present
            if "SKIP_FILES" in user_settings:
                user_settings["SKIP_FILES"] = set(user_settings["SKIP_FILES"])

            # Merge defaults with user settings (user settings take precedence)
            merged_settings = _DEFAULT_CONFIG.copy()
            merged_settings.update(user_settings)
            return merged_settings

        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Could not load {SETTINGS_FILE}: {e}")
            print("Using default settings. Please check your settings file.")
            # Return defaults but don't overwrite the existing file
            defaults = _DEFAULT_CONFIG.copy()
            defaults["SKIP_FILES"] = set(defaults["SKIP_FILES"])
            return defaults
    else:
        # File doesn't exist - create it with defaults
        _create_default_settings()
        defaults = _DEFAULT_CONFIG.copy()
        defaults["SKIP_FILES"] = set(defaults["SKIP_FILES"])
        return defaults


def _create_default_settings() -> None:
    """Create settings.json file with default values."""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(_DEFAULT_CONFIG, f, indent=4)
        print(f"Created {SETTINGS_FILE} with default settings.")
        print("You can edit this file to customize your configuration.")
    except OSError as e:
        print(f"Warning: Could not create {SETTINGS_FILE}: {e}")


# Load settings and create module-level variables
_settings = _load_settings()

# Export all configuration variables at module level
TORCH_NUM_THREADS: int = _settings["TORCH_NUM_THREADS"]
OPENBLAS_NUM_THREADS: int = _settings["OPENBLAS_NUM_THREADS"]
MKL_NUM_THREADS: int = _settings["MKL_NUM_THREADS"]
OMP_NUM_THREADS: int = _settings["OMP_NUM_THREADS"]
NUMEXPR_MAX_THREADS: int = _settings["NUMEXPR_MAX_THREADS"]

CUDA_VISIBLE_DEVICES: str = _settings["CUDA_VISIBLE_DEVICES"]
PYTORCH_CUDA_ALLOC_CONF: str = _settings["PYTORCH_CUDA_ALLOC_CONF"]
FAISS_DISABLE_CPU: str = _settings.get("FAISS_DISABLE_CPU", "0")
CUDA_LAUNCH_BLOCKING: str = _settings.get("CUDA_LAUNCH_BLOCKING", "0")

BATCH_SIZE: int = _settings["BATCH_SIZE"]

INDEX_TYPE: IndexType = _settings["INDEX_TYPE"]
NLIST: int = _settings["NLIST"]
NPROBE: int = _settings["NPROBE"]

# FAISS GPU Settings (advanced users only)
GPU_ENABLED: bool = _settings.get("GPU_ENABLED", False)
GPU_DEVICE: int = _settings.get("GPU_DEVICE", 0)
GPU_MEMORY_FRACTION: float = _settings.get("GPU_MEMORY_FRACTION", 0.8)


SKIP_FILES: Set[str] = _settings["SKIP_FILES"]

BASE_DPI: int = _settings["BASE_DPI"]
BATCH_SIZE_RETRY_DIVISOR: int = _settings["BATCH_SIZE_RETRY_DIVISOR"]

INGESTION_LOG_FILE: str = _settings["INGESTION_LOG_FILE"]
CRASH_LOG_FILE: str = _settings["CRASH_LOG_FILE"]

EMBED_MODEL: str = _settings["EMBED_MODEL"]
OLLAMA_BASE_URL: str = _settings["OLLAMA_BASE_URL"]
OLLAMA_MODEL: str = _settings["OLLAMA_MODEL"]
DEFAULT_TOP_K: int = _settings["DEFAULT_TOP_K"]
REQUEST_TIMEOUT: int = _settings["REQUEST_TIMEOUT"]
TEMPERATURE: float = _settings["TEMPERATURE"]
TOP_P: float = _settings["TOP_P"]
MAX_TOKENS: int = _settings["MAX_TOKENS"]


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
    valid_index_types = {"flat", "ivf", "ivf_flat", "ivf_pq"}
    if INDEX_TYPE not in valid_index_types:
        raise ValueError(
            f"INDEX_TYPE must be one of {valid_index_types}, got: {INDEX_TYPE}"
        )

    # Skip files validation
    if not isinstance(SKIP_FILES, set):
        raise ValueError("SKIP_FILES must be a set")


# Validate on import
validate_config()
