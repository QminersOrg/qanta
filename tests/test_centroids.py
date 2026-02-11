"""Tests for centroid computation."""

import numpy as np
import pandas as pd
import pytest

from qanta import compute_centroids, weighted_percentile


class TestWeightedPercentile:
    """Tests for weighted_percentile function."""

    def test_unweighted_median(self):
        """Equal weights should give standard percentiles."""
        values = [1, 2, 3, 4, 5]
        weights = [1, 1, 1, 1, 1]
        result = weighted_percentile(values, weights, [0.5])
        assert result[0] == pytest.approx(3.0)

    def test_weighted_median(self):
        """Weights should shift the percentile."""
        values = [1, 2, 3]
        weights = [1, 1, 10]  # Heavy weight on 3
        result = weighted_percentile(values, weights, [0.5])
        assert result[0] > 2.5  # Median should be pulled toward 3

    def test_multiple_quantiles(self):
        """Should return multiple percentiles."""
        values = [1, 2, 3, 4, 5]
        weights = [1, 1, 1, 1, 1]
        result = weighted_percentile(values, weights, [0.0, 0.5, 1.0])
        assert len(result) == 3
        assert result[0] == pytest.approx(1.0)
        assert result[2] == pytest.approx(5.0)

    def test_empty_input(self):
        """Empty input should return NaN."""
        result = weighted_percentile([], [], [0.5])
        assert np.isnan(result[0])

    def test_nan_values_filtered(self):
        """NaN values should be ignored."""
        values = [1, np.nan, 3, np.nan, 5]
        weights = [1, 1, 1, 1, 1]
        result = weighted_percentile(values, weights, [0.5])
        assert result[0] == pytest.approx(3.0)

    def test_float_weights_normalized(self):
        """Float weights should be normalized to integers internally."""
        values = [1, 2, 3, 4, 5]
        weights = [0.1, 0.2, 0.3, 0.2, 0.1]  # Non-integer weights
        result = weighted_percentile(values, weights, [0.5])
        # Should work and return a valid result (median pulled toward 3)
        assert result[0] == pytest.approx(3.0)


class TestComputeCentroids:
    """Tests for compute_centroids function."""

    def test_simple_1d(self):
        """Simple 1D binning."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6]})
        result = compute_centroids(df, bins={'x': 2}, quantile=False)

        assert 'x' in result.columns
        assert len(result) == 2
        # First bin: 1, 2, 3 -> mean 2
        # Second bin: 4, 5, 6 -> mean 5
        assert result['x'].iloc[0] == pytest.approx(2.0)
        assert result['x'].iloc[1] == pytest.approx(5.0)

    def test_2d_binning(self):
        """2D binning should create multiple centroids."""
        df = pd.DataFrame(
            {
                'x': [0, 1, 2, 3, 4, 5],
                'y': [0, 0, 0, 1, 1, 1],
            }
        )
        result = compute_centroids(df, bins={'x': 2, 'y': 2}, quantile=False)

        assert 'x' in result.columns
        assert 'y' in result.columns
        # Should have bins for combinations

    def test_with_weights(self):
        """Weighted centroids should be pulled toward heavy observations."""
        df = pd.DataFrame({'x': [1, 2, 3], 'w': [1, 1, 100]})
        result = compute_centroids(df, bins={'x': 1}, weights='w')

        # Centroid should be close to 3 due to heavy weight
        assert result['x'].iloc[0] > 2.9

    def test_output_weights(self):
        """Should output aggregated weights when requested."""
        df = pd.DataFrame({'x': [1, 2, 3, 4], 'w': [1, 2, 3, 4]})
        result = compute_centroids(df, bins={'x': 2}, weights='w', output_weights='w')

        assert 'w' in result.columns
        assert len(result) == 2
        # Total weights should sum to 10
        assert result['w'].sum() == pytest.approx(10.0)
        # Each bin should have positive weight
        assert (result['w'] > 0).all()

    def test_compute_std_errors(self):
        """Should compute standard errors when requested."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6]})
        result = compute_centroids(df, bins={'x': 2}, compute_std_errors=True)

        assert 'x_se' in result.columns
        assert len(result) == 2
        # SE should be positive
        assert (result['x_se'] > 0).all()

    def test_empty_input(self):
        """Empty input should return empty DataFrame."""
        df = pd.DataFrame({'x': [], 'y': []})
        result = compute_centroids(df, bins={'x': 2, 'y': 2})

        assert len(result) == 0
        assert 'x' in result.columns
        assert 'y' in result.columns

    def test_empty_input_with_options(self):
        """Empty input with all options should return proper structure."""
        df = pd.DataFrame({'x': pd.Series([], dtype=float)})
        result = compute_centroids(
            df, bins={'x': 2}, output_weights='w', compute_std_errors=True
        )

        assert len(result) == 0
        assert 'x' in result.columns
        assert 'w' in result.columns
        assert 'x_se' in result.columns

    def test_quantile_vs_equal_width(self):
        """Quantile binning should differ from equal-width for skewed data."""
        # Skewed data: mostly small values, few large
        df = pd.DataFrame({'x': [1, 1, 1, 1, 1, 1, 1, 1, 100, 100]})

        result_quantile = compute_centroids(df, bins={'x': 2}, quantile=True)
        result_equal = compute_centroids(df, bins={'x': 2}, quantile=False)

        # Results should differ for skewed data
        assert not np.allclose(result_quantile['x'].values, result_equal['x'].values)

    def test_custom_bin_edges(self):
        """Should accept custom bin edges."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        result = compute_centroids(df, bins={'x': [0, 3, 7, 11]}, quantile=False)

        # Bins: [0-3), [3-7), [7-11)
        # Values: [1,2], [3,4,5,6], [7,8,9,10]
        assert len(result) == 3

    def test_single_bin_edge_equal_width(self):
        """Single bin edge should fallback to all-encompassing bin."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        result = compute_centroids(df, bins={'x': [5]}, quantile=False)

        # Should put all values in one bin
        assert len(result) == 1
        assert result['x'].iloc[0] == pytest.approx(3.0)  # mean of [1,2,3,4,5]

    def test_custom_quantile_points(self):
        """Should accept explicit quantile points as bin spec."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        # Use quantile points [0.0, 0.5, 1.0] to create 2 bins at median
        result = compute_centroids(df, bins={'x': [0.0, 0.5, 1.0]}, quantile=True)

        assert len(result) == 2

    def test_degenerate_bins_identical_values(self):
        """All identical values should result in single bin."""
        df = pd.DataFrame({'x': [5, 5, 5, 5, 5]})
        result = compute_centroids(df, bins={'x': 3}, quantile=True)

        # All values are the same, bins collapse
        assert len(result) == 1
        assert result['x'].iloc[0] == pytest.approx(5.0)

    def test_deterministic_column_order(self):
        """Output should be deterministic regardless of input order."""
        df = pd.DataFrame({'b': [1, 2, 3], 'a': [4, 5, 6]})
        result1 = compute_centroids(df, bins={'b': 1, 'a': 1})
        result2 = compute_centroids(df, bins={'a': 1, 'b': 1})

        pd.testing.assert_frame_equal(result1, result2)

    def test_returns_dataframe(self):
        """Should return a pandas DataFrame."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        result = compute_centroids(df, bins={'x': 1})

        assert isinstance(result, pd.DataFrame)

    def test_weighted_2d_with_std_errors(self):
        """Test weighted 2D binning with standard errors."""
        df = pd.DataFrame(
            {
                'x': [0, 1, 7, 8, 9, 10],
                'y': [0, 1, 7, 1, 3, 9],
                'w': [1, 3, 1, 1, 2, 1],
            }
        )

        result = compute_centroids(
            df,
            bins={'x': 2, 'y': 1},
            weights='w',
            output_weights='w',
            compute_std_errors=True,
        )

        # Should have all expected columns
        assert 'x' in result.columns
        assert 'y' in result.columns
        assert 'w' in result.columns
        assert 'x_se' in result.columns
        assert 'y_se' in result.columns

        # With 2 bins for x and 1 bin for y, expect 2 centroids
        assert len(result) == 2
