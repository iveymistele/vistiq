"""Tests for vistiq.analysis enrichment (region DataFrame features)."""

import numpy as np
import pandas as pd
import pytest

from vistiq.analysis import (
    GridDensityFeaturesConfig,
    NeighborFeaturesConfig,
    RegionDataFrameEnricher,
    RegionDataFrameEnrichmentConfig,
    add_grid_density_features,
    add_neighbor_features,
)


def _sample_regions_df(n: int = 6) -> pd.DataFrame:
    """Minimal table matching segment-style centroid columns."""
    return pd.DataFrame(
        {
            "label": np.arange(1, n + 1, dtype=int),
            "centroid-0": np.linspace(0.0, 50.0, n),
            "centroid-1": np.zeros(n, dtype=float),
            "centroid-2": np.zeros(n, dtype=float),
        }
    )


class TestNeighborFeaturesConfig:
    """Tests for NeighborFeaturesConfig."""

    def test_default_config(self):
        """Test default NeighborFeaturesConfig."""
        config = NeighborFeaturesConfig()
        assert config.centroid_cols == ("centroid-0", "centroid-1", "centroid-2")
        assert config.k == 5
        assert config.radius == 75.0
        assert config.cluster_quantile == 0.75

    def test_custom_config(self):
        """Test custom NeighborFeaturesConfig."""
        config = NeighborFeaturesConfig(
            centroid_cols=("centroid-0", "centroid-1"),
            k=3,
            radius=10.0,
            cluster_quantile=0.5,
        )
        assert config.centroid_cols == ("centroid-0", "centroid-1")
        assert config.k == 3
        assert config.radius == 10.0
        assert config.cluster_quantile == 0.5

    def test_k_validation(self):
        """Test k must be >= 1."""
        NeighborFeaturesConfig(k=1)
        with pytest.raises(Exception):
            NeighborFeaturesConfig(k=0)

    def test_radius_validation(self):
        """Test radius must be > 0."""
        NeighborFeaturesConfig(radius=0.001)
        with pytest.raises(Exception):
            NeighborFeaturesConfig(radius=0.0)

    def test_cluster_quantile_validation(self):
        """Test cluster_quantile in [0, 1]."""
        NeighborFeaturesConfig(cluster_quantile=0.0)
        NeighborFeaturesConfig(cluster_quantile=1.0)
        with pytest.raises(Exception):
            NeighborFeaturesConfig(cluster_quantile=-0.01)
        with pytest.raises(Exception):
            NeighborFeaturesConfig(cluster_quantile=1.01)


class TestGridDensityFeaturesConfig:
    """Tests for GridDensityFeaturesConfig."""

    def test_default_config(self):
        """Test default GridDensityFeaturesConfig."""
        config = GridDensityFeaturesConfig()
        assert config.grid_shape == (5, 10, 10)

    def test_custom_config(self):
        """Test custom GridDensityFeaturesConfig."""
        config = GridDensityFeaturesConfig(grid_shape=(2, 4, 8))
        assert config.grid_shape == (2, 4, 8)

    def test_grid_shape_validation(self):
        """Test grid_shape must be three positive integers."""
        with pytest.raises(Exception):
            GridDensityFeaturesConfig(grid_shape=(0, 10, 10))
        with pytest.raises(Exception):
            GridDensityFeaturesConfig(grid_shape=(5, 10))


class TestRegionDataFrameEnrichmentConfig:
    """Tests for RegionDataFrameEnrichmentConfig."""

    def test_default_config(self):
        """Test default RegionDataFrameEnrichmentConfig."""
        config = RegionDataFrameEnrichmentConfig()
        assert config.enable_neighbor_features is True
        assert config.enable_grid_density_features is True
        assert isinstance(config.neighbor, NeighborFeaturesConfig)
        assert isinstance(config.grid, GridDensityFeaturesConfig)

    def test_disable_steps(self):
        """Test toggling individual enrichment steps."""
        config = RegionDataFrameEnrichmentConfig(
            enable_neighbor_features=False,
            enable_grid_density_features=False,
        )
        assert config.enable_neighbor_features is False
        assert config.enable_grid_density_features is False


class TestAddNeighborFeatures:
    """Tests for add_neighbor_features."""

    def test_adds_expected_columns(self):
        """Test that neighbor feature columns are added."""
        df = _sample_regions_df()
        out = add_neighbor_features(df, k=3, radius=20.0)
        for col in (
            "nearest_neighbor_label",
            "nearest_neighbor_distance",
            "knn_mean_distance",
            "knn_density",
            "neighbors_within_radius",
            "cluster_flag",
            "density_threshold_used",
            "density_quantile_used",
        ):
            assert col in out.columns
        assert len(out) == len(df)
        assert (out["label"].to_numpy() == df["label"].to_numpy()).all()

    def test_does_not_mutate_input(self):
        """Test input DataFrame is unchanged."""
        df = _sample_regions_df()
        before = df.copy()
        add_neighbor_features(df)
        pd.testing.assert_frame_equal(df, before)


class TestAddGridDensityFeatures:
    """Tests for add_grid_density_features."""

    def test_adds_expected_columns(self):
        """Test grid index and density columns."""
        df = _sample_regions_df()
        out = add_grid_density_features(df, grid_shape=(2, 2, 2))
        for col in ("grid_z", "grid_y", "grid_x", "grid_cell_count", "grid_density_ratio"):
            assert col in out.columns
        assert len(out) == len(df)

    def test_does_not_mutate_input(self):
        """Test input DataFrame is unchanged."""
        df = _sample_regions_df()
        before = df.copy()
        add_grid_density_features(df)
        pd.testing.assert_frame_equal(df, before)


class TestRegionDataFrameEnricher:
    """Tests for RegionDataFrameEnricher."""

    def test_initialization(self):
        """Test RegionDataFrameEnricher initialization."""
        config = RegionDataFrameEnrichmentConfig()
        enricher = RegionDataFrameEnricher(config)
        assert isinstance(enricher, RegionDataFrameEnricher)
        assert isinstance(enricher.config, RegionDataFrameEnrichmentConfig)
        assert enricher.config.enable_neighbor_features == config.enable_neighbor_features

    def test_from_config(self):
        """Test from_config factory."""
        config = RegionDataFrameEnrichmentConfig()
        enricher = RegionDataFrameEnricher.from_config(config)
        assert isinstance(enricher, RegionDataFrameEnricher)

    def test_run_full_pipeline(self):
        """Test run applies neighbor then grid features."""
        df = _sample_regions_df()
        config = RegionDataFrameEnrichmentConfig(
            neighbor=NeighborFeaturesConfig(k=3, radius=25.0),
            grid=GridDensityFeaturesConfig(grid_shape=(2, 2, 2)),
        )
        enricher = RegionDataFrameEnricher(config)
        out = enricher.run(df)
        assert "nearest_neighbor_label" in out.columns
        assert "grid_density_ratio" in out.columns

    def test_run_neighbor_only(self):
        """Test run with grid density disabled."""
        df = _sample_regions_df()
        config = RegionDataFrameEnrichmentConfig(
            enable_grid_density_features=False,
            neighbor=NeighborFeaturesConfig(k=2, radius=100.0),
        )
        enricher = RegionDataFrameEnricher(config)
        out = enricher.run(df)
        assert "knn_mean_distance" in out.columns
        assert "grid_z" not in out.columns

    def test_run_grid_only(self):
        """Test run with neighbor features disabled."""
        df = _sample_regions_df()
        config = RegionDataFrameEnrichmentConfig(
            enable_neighbor_features=False,
            grid=GridDensityFeaturesConfig(grid_shape=(2, 2, 2)),
        )
        enricher = RegionDataFrameEnricher(config)
        out = enricher.run(df)
        assert "grid_cell_count" in out.columns
        assert "nearest_neighbor_label" not in out.columns
