import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np

import config

logger = logging.getLogger(__name__)


class FaissManager:
    """Encapsulates FAISS index + GPU resource management."""

    def __init__(self) -> None:
        super().__init__()
        self._gpu_resources: Any | None = None
        self._gpu_functions_available = False
        self._gpu_status = "GPU disabled"

        if config.GPU_ENABLED:
            available, status = self._detect_gpu_faiss()
            self._gpu_status = status
            logger.info(f"🎮 GPU detection: {status}")
            if available:
                self._gpu_resources = self._create_gpu_resources()
                if self._gpu_resources is None:
                    logger.warning(
                        "⚠️  GPU requested but initialization failed, continuing on CPU"
                    )
            else:
                logger.info("💻 GPU FAISS not available, using CPU")
        else:
            logger.info("💻 GPU disabled, using CPU FAISS")

    @property
    def gpu_ready(self) -> bool:
        return self._gpu_resources is not None

    @property
    def status(self) -> str:
        return self._gpu_status

    def create(
        self,
        dim: int,
        *,
        index_type: str | None = None,
        nlist: int | None = None,
    ) -> tuple[faiss.Index, str]:
        """Create an index and move it to GPU if available."""
        if index_type is None:
            index_type = config.INDEX_TYPE
        if nlist is None:
            nlist = config.NLIST

        cpu_index: faiss.Index | None = None
        actual_type = "flat"
        normalized_type = index_type.lower()

        if normalized_type == "ivf":
            try:
                quantizer = faiss.IndexFlatIP(dim)
                cpu_index = faiss.IndexIVFFlat(
                    quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
                )
                actual_type = "ivf"
                logger.info(f"🔧 Created IVF index: dim={dim}, nlist={nlist}")
            except Exception as exc:
                logger.error(f"⚠️  Failed to create IVF index: {exc}")
                logger.info("Falling back to flat index...")

        if cpu_index is None:
            cpu_index = faiss.IndexFlatIP(dim)
            actual_type = "flat"
            logger.info(f"🔧 Created flat index: dim={dim}")

        if self._gpu_resources is not None and config.GPU_ENABLED:
            gpu_index = self._move_index_to_gpu(cpu_index)
            if gpu_index is not None:
                return gpu_index, actual_type
            logger.info("💻 Continuing with CPU index")

        return cpu_index, actual_type

    def train(self, index: faiss.Index, training_data: np.ndarray) -> bool:
        """Train IVF index with provided vectors."""
        if not hasattr(index, "is_trained") or index.is_trained:
            logger.debug("Index already trained or training not required")
            return True

        num_vectors = len(training_data)
        nlist = getattr(index, "nlist", config.NLIST)
        min_vectors_needed = nlist * 2

        if num_vectors < min_vectors_needed:
            logger.info(
                f"ℹ️  Not enough vectors for IVF training: {num_vectors} < {min_vectors_needed}"
            )
            logger.info("Falling back to flat index for now...")
            return False

        try:
            logger.info(
                f"🎯 Training IVF index with {num_vectors} vectors, {nlist} clusters..."
            )
            _ = index.train(training_data)
            logger.info("✅ IVF index training completed")
            return True
        except (RuntimeError, ValueError) as exc:
            logger.error(f"⚠️  IVF training failed: {exc}")
            logger.info("Will retry with accumulated vectors later...")
            return False

    def maybe_retrain(self, index: faiss.Index, new_vectors: np.ndarray) -> faiss.Index:
        """Add vectors, retraining IVF indexes when the growth ratio is high."""
        if not self._should_retrain(index, len(new_vectors)):
            index.add(new_vectors)
            return index

        logger.info("🔄 Significant data growth detected, retraining IVF index...")
        dim = new_vectors.shape[1]
        existing_vectors = np.zeros((index.ntotal, dim), dtype=np.float32)
        if index.ntotal > 0:
            index.reconstruct_n(0, index.ntotal, existing_vectors)

        all_vectors = (
            np.vstack([existing_vectors, new_vectors])
            if index.ntotal > 0
            else new_vectors
        )

        new_index, actual_type = self.create(dim)
        if actual_type == "ivf":
            training_success = self.train(new_index, all_vectors)
            if training_success:
                if hasattr(new_index, "nprobe"):
                    new_index.nprobe = config.NPROBE
                new_index.add(all_vectors)
                logger.info("✅ IVF index retraining completed")
                return new_index

            logger.info("IVF retraining failed, continuing with existing index")
            index.add(new_vectors)
            return index

        new_index.add(all_vectors)
        logger.info("✅ Switched to flat index with all vectors")
        return new_index

    def prepare_loaded_index(self, index: faiss.Index) -> faiss.Index:
        """Move an index to GPU (if available) and apply config tuning."""
        if self._gpu_resources is not None and config.GPU_ENABLED:
            gpu_index = self._move_index_to_gpu(index)
            if gpu_index is not None:
                index = gpu_index

        if hasattr(index, "nprobe"):
            index.nprobe = config.NPROBE
            logger.info(f"🎯 Set IVF nprobe to {config.NPROBE}")

        return index

    def save(self, index: faiss.Index, path: Path) -> None:
        """Persist an index, moving it to CPU first if needed."""
        save_index = index
        if self._gpu_functions_available and hasattr(index, "device"):
            try:
                device_val = getattr(index, "device")
                is_gpu_index = isinstance(device_val, int) and device_val >= 0
            except Exception:
                is_gpu_index = False

            if is_gpu_index:
                try:
                    save_index = faiss.index_gpu_to_cpu(index)
                    logger.debug("Moved GPU index to CPU for saving")
                except Exception as exc:
                    logger.warning(f"Failed to move GPU index to CPU for saving: {exc}")

        faiss.write_index(save_index, str(path))

    def _detect_gpu_faiss(self) -> tuple[bool, str]:
        required_attrs = [
            "StandardGpuResources",
            "index_cpu_to_gpu",
            "index_gpu_to_cpu",
        ]
        missing_attrs = [attr for attr in required_attrs if not hasattr(faiss, attr)]

        if missing_attrs:
            self._gpu_functions_available = False
            return (
                False,
                f"GPU functions not available in faiss module (missing: {missing_attrs})",
            )

        try:
            gpu_res = faiss.StandardGpuResources()
            test_index = faiss.IndexFlatIP(384)
            _ = faiss.index_cpu_to_gpu(gpu_res, config.GPU_DEVICE, test_index)
            del gpu_res, test_index
            self._gpu_functions_available = True
            return True, f"GPU {config.GPU_DEVICE} available and working"
        except Exception as exc:
            self._gpu_functions_available = True
            return False, f"GPU functions available but GPU failed: {str(exc)[:100]}"

    def _create_gpu_resources(self) -> Any | None:
        if not self._gpu_functions_available:
            return None

        try:
            gpu_res = faiss.StandardGpuResources()
            gpu_res.setTempMemoryFraction(config.GPU_MEMORY_FRACTION)

            logger.info(
                "🎮 GPU resources initialized "
                + f"(device {config.GPU_DEVICE}, memory fraction: {config.GPU_MEMORY_FRACTION})"
            )
            return gpu_res
        except Exception as exc:
            logger.error(f"⚠️  Failed to create GPU resources: {exc}")
            return None

    def _move_index_to_gpu(self, index: faiss.Index) -> faiss.Index | None:
        if not self._gpu_functions_available or self._gpu_resources is None:
            return None

        try:
            gpu_index = faiss.index_cpu_to_gpu(
                self._gpu_resources, config.GPU_DEVICE, index
            )
            logger.info("🎮 Index moved to GPU")
            return gpu_index
        except Exception as exc:
            logger.error(f"⚠️  Failed to move index to GPU: {exc}")
            return None

    @staticmethod
    def _should_retrain(index: faiss.Index, new_vector_count: int) -> bool:
        if not hasattr(index, "is_trained") or not hasattr(index, "ntotal"):
            return False
        if not getattr(index, "is_trained", False):
            return False

        if index.ntotal > 0:
            growth_ratio = new_vector_count / index.ntotal
            return growth_ratio > 0.2
        return False


__all__ = ["FaissManager"]
