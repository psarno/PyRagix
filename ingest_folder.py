from __future__ import annotations

from ingestion.cli import main as _cli_main
from ingestion.environment import EnvironmentManager
from ingestion.pipeline import build_index

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
