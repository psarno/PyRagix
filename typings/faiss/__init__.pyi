"""Minimal type stubs for FAISS methods used in PyRagix."""

from typing import Tuple
import numpy as np
import numpy.typing as npt

class Index:
    """FAISS Index base class."""

    ntotal: int  # Total number of vectors in the index
    d: int  # Dimension of the vectors
    xb: npt.NDArray[np.float32] | None  # Direct access to vectors (for Flat indices)
    is_trained: bool  # Whether the index has been trained

    def search(
        self,
        x: npt.NDArray[np.float32],
        k: int,
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """Search the index for k nearest neighbors.

        Args:
            x: Query vectors of shape (n_queries, dimension)
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (distances, labels) where:
                - distances: array of shape (n_queries, k) with distances to neighbors
                - labels: array of shape (n_queries, k) with indices of neighbors
        """
        ...

    def add(self, x: npt.NDArray[np.float32]) -> None:
        """Add vectors to the index.

        Args:
            x: Vectors to add of shape (n, dimension)
        """
        ...

    def train(self, x: npt.NDArray[np.float32]) -> None:
        """Train the index (required for IVF indices).

        Args:
            x: Training vectors of shape (n, dimension)
        """
        ...

    def reconstruct(self, key: int, recons: npt.NDArray[np.float32]) -> None:
        """Reconstruct a stored vector by index.

        Args:
            key: Index of the vector to reconstruct
            recons: Pre-allocated array to store the reconstructed vector
        """
        ...

    def reconstruct_n(self, n0: int, ni: int, recons: npt.NDArray[np.float32]) -> None:
        """Reconstruct multiple stored vectors.

        Args:
            n0: Starting index
            ni: Number of vectors to reconstruct
            recons: Pre-allocated array to store reconstructed vectors
        """
        ...

    # IVF index specific attribute
    nprobe: int

class IndexFlatIP(Index):
    """FAISS Flat index with inner product similarity."""

    def __init__(self, d: int) -> None:
        """Initialize flat IP index.

        Args:
            d: Dimension of the vectors
        """
        ...

class IndexIVFFlat(Index):
    """FAISS IVF (Inverted File) index with flat quantizer."""

    def __init__(self, quantizer: Index, d: int, nlist: int, metric: int = ...) -> None:
        """Initialize IVF flat index.

        Args:
            quantizer: Quantizer index (typically IndexFlatIP)
            d: Dimension of the vectors
            nlist: Number of inverted lists (clusters)
            metric: Distance metric (e.g., METRIC_INNER_PRODUCT)
        """
        ...

# Distance metrics
METRIC_INNER_PRODUCT: int

def read_index(fname: str) -> Index:
    """Read an index from a file.

    Args:
        fname: Path to the index file

    Returns:
        The loaded FAISS index
    """
    ...

def write_index(index: Index, fname: str) -> None:
    """Write an index to a file.

    Args:
        index: The FAISS index to save
        fname: Path to save the index file
    """
    ...

def vector_to_array(vector: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Convert a FAISS vector to a numpy array.

    Args:
        vector: FAISS vector (typically from index.xb attribute)

    Returns:
        Numpy array representation of the vector
    """
    ...

# GPU support (only available with faiss-gpu)
class StandardGpuResources:
    """GPU resources manager for FAISS GPU indices."""

    def __init__(self) -> None:
        """Initialize GPU resources."""
        ...

    def setTempMemoryFraction(self, fraction: float) -> None:
        """Set fraction of GPU memory to use for temporary storage.

        Args:
            fraction: Memory fraction (0.0 to 1.0)
        """
        ...

def index_cpu_to_gpu(
    resources: StandardGpuResources,
    device: int,
    index: Index,
) -> Index:
    """Move a CPU index to GPU.

    Args:
        resources: GPU resources manager
        device: GPU device ID
        index: CPU index to move

    Returns:
        GPU version of the index
    """
    ...

def index_gpu_to_cpu(index: Index) -> Index:
    """Move a GPU index to CPU.

    Args:
        index: GPU index to move

    Returns:
        CPU version of the index
    """
    ...
