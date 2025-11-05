"""Process-wide environment configuration for the ingestion pipeline.

This module centralises knobs that influence native dependencies (PyTorch,
FAISS, PaddleOCR).  Configuration values originating from `settings.toml` are
mirrored into environment variables before those libraries are imported so that
thread counts, CUDA visibility, and allocator behaviour stay predictable across
ingestion runs. Expensive libraries are imported lazily to keep CLI start-up
fast in workflows that only manipulate metadata.
"""

from __future__ import annotations

import gc
import logging
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator, cast

import config
from classes.ProcessingConfig import ProcessingConfig
from ingestion.models import EmbeddingModel, IngestionContext

if TYPE_CHECKING:
    from classes.OCRProcessor import OCRProcessor as OCRProcessorType
    from ingestion.faiss_manager import FaissManager as FaissManagerType
    from sentence_transformers import SentenceTransformer as SentenceTransformerType
else:
    OCRProcessorType = Any
    FaissManagerType = Any
    SentenceTransformerType = Any

# Lazily populated globals so tests can monkeypatch the constructors without importing heavy deps up front.
OCRProcessor: type[OCRProcessorType] | None = None
SentenceTransformer: type[SentenceTransformerType] | None = None
FaissManager: type[FaissManagerType] | None = None


def _get_torch():
    """Import torch lazily to avoid heavy startup cost."""
    import torch

    return torch


def _ensure_factories() -> tuple[
    type[OCRProcessorType], type[SentenceTransformerType], type[FaissManagerType]
]:
    """Ensure OCR, embedder, and FAISS constructors are available (lazy import)."""
    global OCRProcessor, SentenceTransformer, FaissManager

    if OCRProcessor is None:
        from classes.OCRProcessor import OCRProcessor as _OCRProcessor

        OCRProcessor = _OCRProcessor

    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as _SentenceTransformer

        SentenceTransformer = _SentenceTransformer

    if FaissManager is None:
        from ingestion.faiss_manager import FaissManager as _FaissManager

        FaissManager = _FaissManager

    return OCRProcessor, SentenceTransformer, FaissManager


logger = logging.getLogger(__name__)


class EnvironmentManager:
    """Apply ingestion-specific environment tuning and shared context creation."""

    def __init__(self) -> None:
        super().__init__()
        self._context: IngestionContext | None = None

    def apply(self) -> None:
        """Materialise config-driven env vars and silence verbose dependencies."""
        env_vars = [
            "TORCH_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OMP_NUM_THREADS",
            "NUMEXPR_MAX_THREADS",
            "CUDA_VISIBLE_DEVICES",
            "PYTORCH_ALLOC_CONF",
            "FAISS_DISABLE_CPU",
            "CUDA_LAUNCH_BLOCKING",
        ]

        settings = config.CONFIG
        env_values = settings.model_dump()

        for var in env_vars:
            if var in env_values:
                os.environ[var] = str(env_values[var])

        # glog must be constrained before Paddle's native libraries initialise.
        os.environ["GLOG_minloglevel"] = "2"

        import paddle

        torch = _get_torch()

        for logger_name in [
            "faiss",
            "sentence_transformers",
            "torch",
            "paddle",
            "paddleocr",
        ]:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

        torch.set_num_threads(config.TORCH_NUM_THREADS)

        logger.info(
            "Torch loaded: %s, CUDA available: %s",
            torch.__version__,
            torch.cuda.is_available(),
        )
        logger.info(
            "Paddle compiled with CUDA: %s", paddle.device.is_compiled_with_cuda()
        )

    def initialize(self) -> IngestionContext:
        """Instantiate the shared ingestion context, caching the result.

        All heavyweight dependencies (OCR, embeddings, FAISS) are constructed
        once here so downstream ingestion stages can share them without re-doing
        GPU/CPU initialisation work mid-pipeline.
        """
        if self._context is not None:
            return self._context

        cfg = ProcessingConfig()
        ocr_cls, embed_cls, faiss_cls = _ensure_factories()
        ocr_processor = ocr_cls(cfg)
        embedder = cast(EmbeddingModel, embed_cls(cfg.embed_model))
        faiss_manager = faiss_cls()

        self._context = IngestionContext(
            config=cfg,
            ocr=ocr_processor,
            embedder=embedder,
            faiss_manager=faiss_manager,
            metadata=[],
            processed_hashes=set(),
        )
        return self._context

    @contextmanager
    def memory_guard(self) -> Iterator[None]:
        """Context manager for automatic memory cleanup after processing."""
        try:
            yield
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Force garbage collection and CUDA memory cleanup."""
        _ = gc.collect()
        torch = _get_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["EnvironmentManager"]
