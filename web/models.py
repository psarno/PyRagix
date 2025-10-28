# ======================================
# Web Visualization Models
# - Pydantic v2 models for visualization data
# - Type-safe data structures for frontend/backend
# ======================================

from typing import Literal
from pydantic import BaseModel, Field

DimensionalityMethod = Literal["umap", "tsne"]


class VisualizationPoint(BaseModel):
    """Single point in embedding visualization space."""

    id: int = Field(description="Unique point identifier")
    x: float = Field(description="X coordinate in reduced space")
    y: float = Field(description="Y coordinate in reduced space")
    z: float | None = Field(None, description="Z coordinate (3D only)")
    source: str = Field(description="Source document path or [QUERY]")
    chunk_idx: int = Field(description="Chunk index within source")
    score: float = Field(ge=0.0, le=1.0, description="Similarity score")
    text: str = Field(description="Preview text (truncated)")
    is_query: bool = Field(description="Whether this is the query point")


class VisualizationData(BaseModel):
    """Complete visualization dataset for frontend."""

    points: list[VisualizationPoint] = Field(description="All visualization points")
    query: str = Field(description="Original query text")
    method: DimensionalityMethod = Field(description="Reduction method used")
    dimensions: Literal[2, 3] = Field(description="Output dimensionality")
    total_points: int = Field(ge=0, description="Total points in visualization")
    retrieved_count: int = Field(ge=0, description="Number of retrieved documents")
    error: str | None = Field(None, description="Error message if failed")
