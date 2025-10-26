"""Minimal type stubs for UMAP-learn library used in PyRagix."""

from typing import Literal
import numpy as np
import numpy.typing as npt

class UMAP:
    """UMAP dimensionality reduction algorithm."""

    def __init__(
        self,
        n_components: int = 2,
        random_state: int | None = None,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str | Literal["cosine", "euclidean", "manhattan"] = "euclidean",
    ) -> None:
        """Initialize UMAP reducer.

        Args:
            n_components: Number of dimensions to reduce to
            random_state: Random seed for reproducibility
            n_neighbors: Number of neighbors to consider for manifold approximation
            min_dist: Minimum distance between points in low-dimensional representation
            metric: Distance metric to use
        """
        ...

    def fit_transform(
        self,
        X: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Fit the model and transform the data.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            Transformed data of shape (n_samples, n_components)
        """
        ...
