"""Post-hoc features for region tables (e.g. segment CSV / DataFrame output)."""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from pydantic import Field, field_validator
from prefect import task
from sklearn.neighbors import NearestNeighbors

from vistiq.core import Configurable, Configuration

logger = logging.getLogger(__name__)


def add_neighbor_features(
    df: pd.DataFrame,
    centroid_cols: Tuple[str, ...] = ("centroid-0", "centroid-1", "centroid-2"),
    k: int = 5,
    radius: float = 75,
    cluster_quantile: float = 0.75,
) -> pd.DataFrame:
    df = df.copy()

    coords = df[list(centroid_cols)].to_numpy(dtype=float)
    labels = df["label"].to_numpy()

    n_neighbors = min(k + 1, len(df))

    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(coords)

    distances, indices = nn.kneighbors(coords)

    # remove self neighbor
    neighbor_distances = distances[:, 1:]
    neighbor_indices = indices[:, 1:]

    df["nearest_neighbor_label"] = labels[neighbor_indices[:, 0]]
    df["nearest_neighbor_distance"] = neighbor_distances[:, 0]
    df["knn_mean_distance"] = neighbor_distances.mean(axis=1)
    df["knn_density"] = k / (df["knn_mean_distance"] + 1e-9)

    radius_neighbors = nn.radius_neighbors(coords, radius=radius, return_distance=False)
    df["neighbors_within_radius"] = [len(neigh) - 1 for neigh in radius_neighbors]

    threshold = df["knn_density"].quantile(cluster_quantile)
    df["cluster_flag"] = df["knn_density"] >= threshold
    df["density_threshold_used"] = threshold
    df["density_quantile_used"] = cluster_quantile

    return df


def add_grid_density_features(
    df: pd.DataFrame,
    grid_shape: Tuple[int, int, int] = (5, 10, 10),
) -> pd.DataFrame:
    df = df.copy()

    coords = df[["centroid-0", "centroid-1", "centroid-2"]].to_numpy(dtype=float)

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    ranges = np.where(maxs - mins == 0, 1, maxs - mins)

    normalized = (coords - mins) / ranges

    grid_indices = np.floor(normalized * np.array(grid_shape)).astype(int)

    grid_indices = np.minimum(grid_indices, np.array(grid_shape) - 1)

    df["grid_z"] = grid_indices[:, 0]
    df["grid_y"] = grid_indices[:, 1]
    df["grid_x"] = grid_indices[:, 2]

    grid_counts = (
        df.groupby(["grid_z", "grid_y", "grid_x"])
        .size()
        .rename("grid_cell_count")
        .reset_index()
    )

    df = df.merge(grid_counts, on=["grid_z", "grid_y", "grid_x"], how="left")

    total_cells = np.prod(grid_shape)

    avg_objects_per_cell = len(df) / total_cells

    df["grid_density_ratio"] = df["grid_cell_count"] / avg_objects_per_cell

    return df


class NeighborFeaturesConfig(Configuration):
    """Settings for :func:`add_neighbor_features`."""

    centroid_cols: Tuple[str, ...] = ("centroid-0", "centroid-1", "centroid-2")
    k: int = Field(default=5, ge=1)
    radius: float = Field(default=75.0, gt=0)
    cluster_quantile: float = Field(default=0.75, ge=0.0, le=1.0)


class GridDensityFeaturesConfig(Configuration):
    """Settings for :func:`add_grid_density_features`."""

    grid_shape: Tuple[int, int, int] = (5, 10, 10)

    @field_validator("grid_shape")
    @classmethod
    def validate_grid_shape(cls, v: Tuple[int, int, int]) -> Tuple[int, int, int]:
        if len(v) != 3 or any(x < 1 for x in v):
            raise ValueError("grid_shape must be three positive integers")
        return v


class RegionDataFrameEnrichmentConfig(Configuration):
    """Toggle and parameterize neighbor- and grid-based region table features."""

    enable_neighbor_features: bool = True
    neighbor: NeighborFeaturesConfig = Field(default_factory=NeighborFeaturesConfig)
    enable_grid_density_features: bool = True
    grid: GridDensityFeaturesConfig = Field(default_factory=GridDensityFeaturesConfig)


class RegionDataFrameEnricher(Configurable[RegionDataFrameEnrichmentConfig]):
    """Applies configured enrichment steps to a regions DataFrame (e.g. before CSV export)."""

    @classmethod
    def from_config(cls, config: RegionDataFrameEnrichmentConfig) -> RegionDataFrameEnricher:
        return cls(config)

    @task(name="RegionDataFrameEnricher.run")
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` with optional neighbor and grid density columns added."""
        out = df
        if self.config.enable_neighbor_features:
            nc = self.config.neighbor
            out = add_neighbor_features(
                out,
                centroid_cols=nc.centroid_cols,
                k=nc.k,
                radius=nc.radius,
                cluster_quantile=nc.cluster_quantile,
            )
        if self.config.enable_grid_density_features:
            out = add_grid_density_features(
                out, grid_shape=self.config.grid.grid_shape
            )
        logger.info(
            "RegionDataFrameEnricher: columns=%s rows=%s",
            list(out.columns),
            len(out),
        )
        return out
