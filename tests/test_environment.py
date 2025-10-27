from __future__ import annotations

from typing import Any

import pytest

from ingestion.environment import EnvironmentManager


def test_environment_manager_initialize_uses_mocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = EnvironmentManager()

    ocr_calls: list[Any] = []
    embed_calls: list[Any] = []
    faiss_calls = {"count": 0}

    class OCRStub:
        def __init__(self, cfg: Any) -> None:
            super().__init__()
            ocr_calls.append(cfg)

        def extract_from_image(self, path: str) -> str:
            return "stub"

        def ocr_embedded_images(self, doc: Any, page: Any) -> str:
            return ""

        def ocr_pil_image(self, pil_img: Any) -> str:
            return ""

        def ocr_page_tiled(
            self,
            page: Any,
            dpi: int,
            tile_px: int | None = None,
            overlap: int | None = None,
        ) -> str:
            return ""

    class EmbedderStub:
        def __init__(self, model_name: str, **_: Any) -> None:
            super().__init__()
            embed_calls.append(model_name)

        def encode(
            self,
            sentences: list[str],
            *,
            batch_size: int,
            convert_to_numpy: bool,
            normalize_embeddings: bool,
        ) -> list[list[float]]:
            assert batch_size > 0
            assert convert_to_numpy is True
            assert normalize_embeddings is True
            return [[0.0] * 3 for _ in sentences]

    class FaissStub:
        def __init__(self) -> None:
            super().__init__()
            faiss_calls["count"] += 1

    monkeypatch.setattr("ingestion.environment.OCRProcessor", OCRStub)
    monkeypatch.setattr("ingestion.environment.SentenceTransformer", EmbedderStub)
    monkeypatch.setattr("ingestion.environment.FaissManager", FaissStub)

    context = manager.initialize()

    assert ocr_calls == [context.config]
    assert embed_calls == [context.config.embed_model]
    assert faiss_calls["count"] == 1

    assert isinstance(context.ocr, OCRStub)
    assert isinstance(context.embedder, EmbedderStub)
    assert isinstance(context.faiss_manager, FaissStub)
    assert context.metadata == []
    assert context.processed_hashes == set()

    again = manager.initialize()
    assert again is context
    assert faiss_calls["count"] == 1
    assert len(ocr_calls) == 1
    assert len(embed_calls) == 1
