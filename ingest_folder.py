from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any, Callable

# Suppress misleading PaddlePaddle ccache warning BEFORE any imports
# (only relevant when building from source, not using pre-built wheels)
warnings.filterwarnings("ignore", message=".*ccache.*", category=UserWarning)

# Suppress C++ library logging BEFORE importing torch/paddle/faiss
# Must be set before any imports that trigger these libraries
_ = os.environ.setdefault("GLOG_minloglevel", "2")  # Google logging (PaddleOCR)
_ = os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # TensorFlow/oneDNN
_ = os.environ.setdefault("ONEDNN_VERBOSE", "0")  # oneDNN verbose output

if TYPE_CHECKING:  # pragma: no cover - import hints only
    from ingestion.environment import EnvironmentManager
    from ingestion.models import IngestionContext


_env_manager: EnvironmentManager | None = None


def _get_env_manager() -> EnvironmentManager:
    """Instantiate EnvironmentManager lazily to avoid eager heavy imports."""
    global _env_manager
    if _env_manager is None:
        from ingestion.environment import EnvironmentManager

        _env_manager = EnvironmentManager()
    return _env_manager


def _get_cli_main() -> Callable[[], None]:
    from ingestion.cli import main as _cli_main

    return _cli_main


def main() -> None:
    """Compatibility wrapper that delegates to `ingestion.cli.main`."""
    _get_cli_main()()


def apply_user_configuration() -> None:
    """Backwards-compatible helper to apply configuration settings."""
    _get_env_manager().apply()


def create_context() -> IngestionContext:
    """Backwards-compatible helper to retrieve an ingestion context."""
    return _get_env_manager().initialize()


def build_index(*args: Any, **kwargs: Any) -> Any:
    """Lazy proxy to the ingestion pipeline build_index function."""
    from ingestion.pipeline import build_index as _build_index

    return _build_index(*args, **kwargs)


__all__ = ["apply_user_configuration", "build_index", "create_context", "main"]


if __name__ == "__main__":
    main()
