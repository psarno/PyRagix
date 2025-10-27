from pathlib import Path

from classes.ProcessingConfig import ProcessingConfig
from ingestion.file_filters import (
    calculate_file_hash,
    should_skip_file,
)


def _make_config() -> ProcessingConfig:
    cfg = ProcessingConfig()
    cfg.doc_extensions.add(".txt")
    cfg.skip_files = {"skipme.txt"}
    return cfg


def test_should_skip_file_honors_skip_list(tmp_path: Path) -> None:
    cfg = _make_config()
    sample = tmp_path / "skipme.txt"
    _ = sample.write_text("data", encoding="utf-8")

    skip, reason = should_skip_file(str(sample), ".txt", set(), cfg)

    assert skip is True
    assert "skip" in reason


def test_should_skip_file_detects_processed_hash(tmp_path: Path) -> None:
    cfg = _make_config()
    sample = tmp_path / "doc.txt"
    _ = sample.write_text("docs", encoding="utf-8")

    processed = {calculate_file_hash(str(sample))}

    skip, reason = should_skip_file(str(sample), ".txt", processed, cfg)

    assert skip is True
    assert reason == "already processed"


def test_should_skip_file_respects_extension_filters(tmp_path: Path) -> None:
    cfg = _make_config()
    cfg.allowed_extensions = {".txt"}

    sample = tmp_path / "image.png"
    _ = sample.write_bytes(b"\x00\x01")

    skip, reason = should_skip_file(str(sample), ".png", set(), cfg)

    assert skip is True
    assert "file type not in filter" in reason
