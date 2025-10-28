import os
import warnings

# Suppress misleading PaddlePaddle ccache warning BEFORE any imports
# (only relevant when building from source, not using pre-built wheels)
warnings.filterwarnings("ignore", message=".*ccache.*", category=UserWarning)

# Suppress C++ library logging BEFORE importing torch/paddle/faiss
# Must be set before any imports that trigger these libraries
_ = os.environ.setdefault("GLOG_minloglevel", "2")  # Google logging (PaddleOCR)
_ = os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # TensorFlow/oneDNN
_ = os.environ.setdefault("ONEDNN_VERBOSE", "0")  # oneDNN verbose output

from ingestion.cli import main as _cli_main  # noqa: E402
from ingestion.environment import EnvironmentManager  # noqa: E402
from ingestion.pipeline import build_index  # noqa: E402

_ENV_MANAGER = EnvironmentManager()

__all__ = ["apply_user_configuration", "build_index", "create_context", "main"]


def main() -> None:
    """Compatibility wrapper that delegates to `ingestion.cli.main`."""
    _cli_main()


def apply_user_configuration() -> None:
    """Backwards-compatible helper to apply configuration settings."""
    _ENV_MANAGER.apply()


def create_context():
    """Backwards-compatible helper to retrieve an ingestion context."""
    return _ENV_MANAGER.initialize()


if __name__ == "__main__":
    main()
