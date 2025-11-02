import gc
import logging
import os
from contextlib import contextmanager
from typing import Iterator, cast

import torch
from sentence_transformers import SentenceTransformer

import config
from classes.OCRProcessor import OCRProcessor
from classes.ProcessingConfig import ProcessingConfig
from ingestion.faiss_manager import FaissManager
from ingestion.models import EmbeddingModel, IngestionContext

logger = logging.getLogger(__name__)


class EnvironmentManager:
    """Owns environment/application state setup for ingestion."""

    def __init__(self) -> None:
        super().__init__()
        self._context: IngestionContext | None = None

    def apply(self) -> None:
        """Apply configuration-driven environment tuning."""
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

        os.environ["GLOG_minloglevel"] = "2"

        import paddle

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
        """Instantiate the shared ingestion context, caching the result."""
        if self._context is not None:
            return self._context

        cfg = ProcessingConfig()
        ocr_processor = OCRProcessor(cfg)
        embedder = cast(EmbeddingModel, SentenceTransformer(cfg.embed_model))
        faiss_manager = FaissManager()

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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["EnvironmentManager"]
