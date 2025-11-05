from __future__ import annotations

from pathlib import Path

import pytest

from rag.configuration import DEFAULT_CONFIG, validate_config


def _base_config(tmp_path: Path):
    config = DEFAULT_CONFIG.model_copy()
    config.index_path = tmp_path / "local_faiss.index"
    config.db_path = tmp_path / "documents.db"
    config.enable_hybrid_search = False
    return config


def test_validate_config_requires_existing_index(tmp_path: Path) -> None:
    config = _base_config(tmp_path)

    with pytest.raises(FileNotFoundError) as exc:
        validate_config(config)

    assert "FAISS index not found" in str(exc.value)


def test_validate_config_passes_with_existing_artifacts(tmp_path: Path) -> None:
    config = _base_config(tmp_path)

    index_path = Path(config.index_path)
    db_path = Path(config.db_path)

    _ = index_path.write_bytes(b"\x00")
    _ = db_path.write_text("test", encoding="utf-8")

    # Should not raise
    validate_config(config)
