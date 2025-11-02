from __future__ import annotations

from types import SimpleNamespace

import pytest

from ingestion.faiss_manager import FaissManager
from utils.faiss_types import SupportsNList, SupportsQuantizer


def _install_faiss_stub(monkeypatch: pytest.MonkeyPatch, *, ivf_raises: bool) -> None:
    class FlatIndex:
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.dim = dim
            self.ntotal = 0

    class IvfIndex:
        def __init__(
            self, quantizer: FlatIndex, dim: int, nlist: int, metric: int
        ) -> None:  # noqa: ARG002
            super().__init__()
            if ivf_raises:
                raise RuntimeError("ivf not supported")
            self.dim = dim
            self.nlist = nlist
            self.quantizer = quantizer
            self.is_trained = False

    stub = SimpleNamespace(
        IndexFlatIP=FlatIndex,
        IndexIVFFlat=IvfIndex,
        METRIC_INNER_PRODUCT=0,
    )

    monkeypatch.setattr("ingestion.faiss_manager.faiss", stub)


def _apply_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ingestion.faiss_manager.config.GPU_ENABLED", False)
    monkeypatch.setattr("ingestion.faiss_manager.config.INDEX_TYPE", "ivf")
    monkeypatch.setattr("ingestion.faiss_manager.config.NLIST", 8)


def test_create_ivf_uses_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    _apply_config_defaults(monkeypatch)
    _install_faiss_stub(monkeypatch, ivf_raises=False)

    manager = FaissManager()
    index, actual_type = manager.create(dim=384)

    assert actual_type == "ivf"
    assert type(index).__name__ == "IvfIndex"
    assert isinstance(index, SupportsNList)
    assert isinstance(index, SupportsQuantizer)
    assert index.nlist == 8
    assert index.quantizer.__class__.__name__ == "FlatIndex"


def test_create_ivf_falls_back_to_flat(monkeypatch: pytest.MonkeyPatch) -> None:
    _apply_config_defaults(monkeypatch)
    _install_faiss_stub(monkeypatch, ivf_raises=True)

    manager = FaissManager()
    index, actual_type = manager.create(dim=128)

    assert actual_type == "flat"
    assert type(index).__name__ == "FlatIndex"
