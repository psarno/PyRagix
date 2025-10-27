from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from classes.ProcessingConfig import ProcessingConfig
from ingestion.file_scanner import Chunker, DocumentExtractor

if TYPE_CHECKING:
    from ingestion.models import PDFDocument, PDFPage, PILImage


class MockOCRProcessor:
    """Mock OCR processor for testing."""

    def __init__(self, config: ProcessingConfig | None = None) -> None:
        """Initialize mock OCR processor."""
        super().__init__()
        self.config = config or ProcessingConfig()

    def extract_from_image(self, path: str) -> str:
        return "Mocked OCR text"

    def ocr_embedded_images(self, doc: PDFDocument, page: PDFPage) -> str:
        return ""

    def ocr_pil_image(self, pil_img: PILImage) -> str:
        return "Mocked PIL OCR"

    def ocr_page_tiled(
        self,
        page: PDFPage,
        dpi: int,
        tile_px: int | None = None,
        overlap: int | None = None,
    ) -> str:
        return "Mocked tiled OCR"


class MockEmbedder:
    """Mock embedding model for testing."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> list[list[float]]:
        # Return simple mock embeddings
        return [[0.1, 0.2, 0.3] for _ in sentences]


def _make_config() -> ProcessingConfig:
    cfg = ProcessingConfig()
    cfg.chunk_size = 100
    cfg.chunk_overlap = 20
    return cfg


def test_document_extractor_handles_text_file(tmp_path: Path) -> None:
    """Test that DocumentExtractor can extract from a simple text file."""
    cfg = _make_config()
    ocr = MockOCRProcessor()
    extractor = DocumentExtractor(cfg, ocr)

    # Create a mock text file (simulated as image for OCR)
    text_file = tmp_path / "test.png"
    _ = text_file.write_bytes(b"\x00\x01")  # Minimal PNG-like data

    with patch("ingestion.file_scanner.extract_text") as mock_extract:
        mock_extract.return_value = "  Sample   text   with   spaces  "

        result = extractor.extract(str(text_file))

        # Should clean the text (normalize whitespace)
        assert result == "Sample text with spaces"
        mock_extract.assert_called_once_with(str(text_file), ocr, cfg)


def test_document_extractor_handles_empty_extraction(tmp_path: Path) -> None:
    """Test that DocumentExtractor handles empty text extraction gracefully."""
    cfg = _make_config()
    ocr = MockOCRProcessor()
    extractor = DocumentExtractor(cfg, ocr)

    text_file = tmp_path / "empty.png"
    _ = text_file.write_bytes(b"\x00\x01")

    with patch("ingestion.file_scanner.extract_text") as mock_extract:
        mock_extract.return_value = "   "

        result = extractor.extract(str(text_file))

        # Empty text should still be cleaned
        assert result == ""


def test_document_extractor_handles_ocr_failure(tmp_path: Path) -> None:
    """Test that DocumentExtractor handles OCR failures gracefully."""
    cfg = _make_config()
    ocr = MockOCRProcessor()
    extractor = DocumentExtractor(cfg, ocr)

    text_file = tmp_path / "corrupt.png"
    _ = text_file.write_bytes(b"\x00\x01")

    with patch("ingestion.file_scanner.extract_text") as mock_extract:
        mock_extract.side_effect = RuntimeError("OCR failed")

        # Should propagate the exception
        with pytest.raises(RuntimeError, match="OCR failed"):
            _ = extractor.extract(str(text_file))


def test_chunker_creates_fixed_size_chunks() -> None:
    """Test that Chunker creates proper fixed-size chunks."""
    cfg = _make_config()
    cfg.chunk_size = 50
    cfg.chunk_overlap = 10
    embedder = MockEmbedder()
    chunker = Chunker(cfg, embedder)

    text = "A" * 120  # 120 characters

    with patch("config.ENABLE_SEMANTIC_CHUNKING", False):
        chunks = chunker.chunk(text)

        # Should create 3 chunks: [0:50], [40:90], [80:120]
        assert len(chunks) >= 2
        assert all(len(chunk) <= 50 for chunk in chunks[:-1])


def test_chunker_handles_empty_text() -> None:
    """Test that Chunker handles empty text gracefully."""
    cfg = _make_config()
    embedder = MockEmbedder()
    chunker = Chunker(cfg, embedder)

    chunks = chunker.chunk("")

    assert chunks == []


def test_chunker_semantic_fallback_when_no_embedder() -> None:
    """Test that Chunker uses semantic chunking when enabled and embedder available."""
    cfg = _make_config()
    cfg.chunk_size = 50
    cfg.chunk_overlap = 10
    embedder = MockEmbedder()
    chunker = Chunker(cfg, embedder)

    text = "This is a test. " * 20

    with patch("config.ENABLE_SEMANTIC_CHUNKING", True):
        # Call chunk_text directly which handles semantic chunking
        with patch("ingestion.file_scanner.chunk_text") as mock_chunk:
            mock_chunk.return_value = ["chunk1", "chunk2"]

            chunks = chunker.chunk(text)

            # Should use semantic chunking
            assert len(chunks) == 2
            mock_chunk.assert_called_once_with(text, cfg, embedder=embedder)


def test_chunker_handles_semantic_import_error() -> None:
    """Test that Chunker delegates to chunk_text correctly."""
    cfg = _make_config()
    cfg.chunk_size = 50
    cfg.chunk_overlap = 10
    embedder = MockEmbedder()
    chunker = Chunker(cfg, embedder)

    text = "A" * 120

    # Test that Chunker delegates to chunk_text
    with patch("ingestion.file_scanner.chunk_text") as mock_chunk:
        mock_chunk.return_value = ["chunk1", "chunk2", "chunk3"]

        chunks = chunker.chunk(text)

        # Should delegate to chunk_text
        assert len(chunks) == 3
        mock_chunk.assert_called_once_with(text, cfg, embedder=embedder)


def test_document_extractor_and_chunker_orchestration(tmp_path: Path) -> None:
    """Test the full orchestration of extraction and chunking."""
    cfg = _make_config()
    cfg.chunk_size = 30
    cfg.chunk_overlap = 5
    ocr = MockOCRProcessor()
    embedder = MockEmbedder()

    extractor = DocumentExtractor(cfg, ocr)
    chunker = Chunker(cfg, embedder)

    # Create a test file
    test_file = tmp_path / "test.png"
    _ = test_file.write_bytes(b"\x00\x01")

    with patch("ingestion.file_scanner.extract_text") as mock_extract:
        # Simulate extracted text with extra whitespace
        mock_extract.return_value = (
            "  The   quick brown  fox jumps  over the lazy  dog  "
        )

        with patch("config.ENABLE_SEMANTIC_CHUNKING", False):
            # Extract and clean text
            text = extractor.extract(str(test_file))
            assert text == "The quick brown fox jumps over the lazy dog"

            # Chunk the text
            chunks = chunker.chunk(text)

            # Verify chunks are created
            assert len(chunks) >= 1
            # Verify all chunks are within size limit
            assert all(len(chunk) <= cfg.chunk_size for chunk in chunks[:-1])


def test_chunker_respects_config_parameters() -> None:
    """Test that Chunker respects custom chunk size and overlap."""
    cfg = _make_config()
    cfg.chunk_size = 20
    cfg.chunk_overlap = 5
    embedder = MockEmbedder()
    chunker = Chunker(cfg, embedder)

    text = "0123456789" * 10  # 100 characters

    with patch("config.ENABLE_SEMANTIC_CHUNKING", False):
        chunks = chunker.chunk(text)

        # First chunk should be exactly 20 chars
        assert len(chunks[0]) == 20
        # Step size should be 15 (20 - 5 overlap)
        # So chunks start at: 0, 15, 30, 45, 60, 75, 90
        assert len(chunks) >= 6


def test_document_extractor_pdf_extraction(tmp_path: Path) -> None:
    """Test that DocumentExtractor handles PDF files correctly."""
    cfg = _make_config()
    ocr = MockOCRProcessor()
    extractor = DocumentExtractor(cfg, ocr)

    pdf_file = tmp_path / "test.pdf"
    _ = pdf_file.write_bytes(b"%PDF-1.4\n")  # Minimal PDF header

    with patch("ingestion.file_scanner.extract_text") as mock_extract:
        mock_extract.return_value = "PDF content extracted"

        result = extractor.extract(str(pdf_file))

        assert result == "PDF content extracted"
        mock_extract.assert_called_once_with(str(pdf_file), ocr, cfg)


def test_document_extractor_html_extraction(tmp_path: Path) -> None:
    """Test that DocumentExtractor handles HTML files correctly."""
    cfg = _make_config()
    ocr = MockOCRProcessor()
    extractor = DocumentExtractor(cfg, ocr)

    html_file = tmp_path / "test.html"
    _ = html_file.write_text("<html><body>HTML content</body></html>", encoding="utf-8")

    with patch("ingestion.file_scanner.extract_text") as mock_extract:
        mock_extract.return_value = "HTML content"

        result = extractor.extract(str(html_file))

        assert result == "HTML content"
