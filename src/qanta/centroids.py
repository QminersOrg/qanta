"""N-dimensional centroid computation for binned data."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from functools import reduce
from typing import TYPE_CHECKING

import numba  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray

    FloatArray = NDArray[np.floating]


def compute_centroids(
    df: pd.DataFrame,
    bins: Mapping[str, int | Collection[float]],
    *,
    quantile: bool = True,
    weights: str | None = None,
    output_weights: str | None = None,
    compute_std_errors: bool = False,
) -> pd.DataFrame:
    """Compute n-dimensional centroids for binned data.

    Splits observations (rows) into n-dimensional hyper-rectangles using the
    specified binning strategy, then computes the centroid (weighted mean)
    for each bin.

    Args:
        df: Input DataFrame containing the data.
        bins: Dictionary mapping column names to either:
            - An integer specifying the number of bins
            - A sequence of bin edges (for custom binning)
        quantile: If True, use quantile-based binning (equal-count bins).
            If False, use equal-width binning.
        weights: Column name to use as observation weights. If None, all
            observations are weighted equally.
        output_weights: If provided, include aggregated weights in the output
            under this column name.
        compute_std_errors: If True, include standard errors of the
            centroids as additional columns with '_se' suffix.

    Returns:
        DataFrame with one row per bin, containing:
            - Centroid values for each column specified in `bins`
            - Aggregated weights (if output_weights is provided)
            - Standard errors (if compute_std_errors=True)

    Examples:
        ```python
        import pandas as pd

        df = pd.DataFrame({
            'x': [0, 1, 7, 8, 9, 10],
            'y': [0, 1, 7, 1, 3, 9],
            'w': [1, 3, 1, 1, 2, 1],
        })

        result = compute_centroids(
            df,
            bins={'x': 2, 'y': 1},
            weights='w',
            output_weights='w',
        )
        ```
    """
    # Sort columns for deterministic output
    column_bins = sorted(bins.items())
    columns = [col for col, _ in column_bins]

    # Handle empty input
    if df.empty:
        result_columns = (
            columns
            + ([output_weights] if output_weights else [])
            + ([f'{col}_se' for col in columns] if compute_std_errors else [])
        )
        return pd.DataFrame(columns=result_columns)

    # Extract data as numpy arrays
    values = df[columns].to_numpy(dtype=np.float64)
    weight_values = (
        df[weights].to_numpy(dtype=np.float64) if weights is not None else None
    )

    # Compute bin assignments for each column
    categories = []
    for i, (_, bin_spec) in enumerate(column_bins):
        column_values = values[:, i]
        if quantile:
            category = _create_quantile_bins(column_values, bin_spec, weight_values)
        else:
            category = _create_equal_width_bins(column_values, bin_spec)
        categories.append(category)

    # Group observations into bins
    filtered_values, filtered_weights, bin_codes = _group_by(
        values, weight_values, categories
    )

    # Compute centroids (weighted means) for each bin
    means, standard_errors, agg_weights = _compute_means(
        filtered_values, filtered_weights, bin_codes, compute_std_errors
    )

    # Build result DataFrame
    result = pd.DataFrame(means, columns=columns)

    if output_weights is not None:
        result[output_weights] = agg_weights

    if compute_std_errors:
        standard_errors_df = pd.DataFrame(
            standard_errors, columns=[f'{col}_se' for col in columns]
        )
        result = pd.concat([result, standard_errors_df], axis=1)

    return result


def weighted_percentile(
    values: Collection[float],
    weights: Collection[float],
    quantiles: Collection[float],
) -> list[float]:
    """Compute weighted percentiles.

    Args:
        values: Data values.
        weights: Weights for each value (must be positive).
        quantiles: Quantiles to compute (values between 0 and 1).

    Returns:
        List of percentile values corresponding to the requested quantiles.
    """
    values_array = np.asarray(values, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    quantile_points = np.asarray(quantiles, dtype=np.float64)

    # Filter out non-finite values
    valid_mask = np.isfinite(values_array)
    values_array = values_array[valid_mask]
    weight_array = weight_array[valid_mask]

    if len(values_array) == 0:
        return [np.nan] * len(quantile_points)

    # Sort by values
    sort_indices = np.argsort(values_array)
    values_array = values_array[sort_indices]
    weight_array = weight_array[sort_indices]

    # Normalize weights if they're not integers
    if np.any(weight_array.astype(np.int64) != weight_array):
        weight_array = weight_array / weight_array.max()
        weight_array = np.clip(weight_array, 1e-6, 1.0)
        weight_array = weight_array / weight_array.min()
        weight_array = (weight_array * 1000).astype(np.int64)

    # Build cumulative distribution for interpolation
    cumulative_weights = np.cumsum(weight_array)
    x_coords = (
        np.concatenate([[0], cumulative_weights[:-1], cumulative_weights - 1])
        .reshape(2, -1)
        .T.ravel()
    )
    y_coords = np.repeat(values_array, 2)

    # Interpolate to find percentile values
    interpolated: FloatArray = np.interp(
        quantile_points * x_coords[-1], x_coords, y_coords
    )
    return [float(val) for val in interpolated]


def _create_equal_width_bins(
    values: FloatArray,
    bin_spec: int | Collection[float],
) -> tuple[FloatArray, NDArray[np.intp], NDArray[np.bool_]]:
    """Create equal-width bins."""
    if isinstance(bin_spec, Collection) and not isinstance(bin_spec, int | np.integer):
        bin_edges = np.asarray(bin_spec, dtype=np.float64)
        if len(bin_edges) < 2:
            bin_edges = np.array([-np.inf, np.inf])
    else:
        n_bins = int(bin_spec)
        min_value = float(np.nanmin(values))
        max_value = float(np.nanmax(values))
        bin_edges = np.linspace(min_value, max_value, n_bins + 1)

    return _assign_values_to_bins(bin_edges, values)


def _create_quantile_bins(
    values: FloatArray,
    bin_spec: int | Collection[float],
    weights: FloatArray | None,
) -> tuple[FloatArray, NDArray[np.intp], NDArray[np.bool_]]:
    """Create quantile-based bins (equal count per bin)."""
    if isinstance(bin_spec, Collection) and not isinstance(bin_spec, int | np.integer):
        quantile_points = np.asarray(bin_spec, dtype=np.float64)
    else:
        n_bins = int(bin_spec)
        quantile_points = np.linspace(0, 1, n_bins + 1)

    # Compute bin edges from quantiles
    if weights is None:
        bin_edges = np.nanpercentile(values, quantile_points * 100)
    else:
        bin_edges = np.array(weighted_percentile(values, weights, quantile_points))

    # Handle duplicate edges (can happen with discrete data)
    bin_edges = np.sort(np.unique(bin_edges))

    # If bins degenerate into single point, create one all-encompassing bin
    if len(bin_edges) <= 1:
        bin_edges = np.array([-np.inf, np.inf])

    return _assign_values_to_bins(bin_edges, values)


def _assign_values_to_bins(
    bin_edges: FloatArray,
    values: FloatArray,
) -> tuple[FloatArray, NDArray[np.intp], NDArray[np.bool_]]:
    """Assign each value to a bin based on bin edges."""
    bin_indices = np.searchsorted(bin_edges, values, side='left')

    # Values exactly at first edge go into first bin
    bin_indices[values == bin_edges[0]] = 1

    # Mark values outside valid range or NaN as invalid
    invalid_mask = (
        (bin_indices == 0) | (bin_indices == len(bin_edges)) | np.isnan(values)
    )

    return bin_edges, bin_indices, invalid_mask


def _group_by(
    values: FloatArray,
    weights: FloatArray | None,
    categories: list[tuple[FloatArray, NDArray[np.intp], NDArray[np.bool_]]],
) -> tuple[FloatArray, FloatArray | None, NDArray[np.intp]]:
    """Group observations by their multi-dimensional bin assignments."""
    all_edges, all_bin_indices, all_invalid_masks = zip(*categories, strict=True)

    # Filter out observations that fall outside valid bins in any dimension
    combined_invalid_mask = reduce(lambda x, y: x | y, all_invalid_masks)
    valid_values = values[~combined_invalid_mask]

    valid_weights = None
    if weights is not None:
        valid_weights = weights[~combined_invalid_mask]

    # Compute unique bin code for each observation using mixed-radix encoding
    # (like converting multi-digit number to single integer)
    bins_per_dimension = np.array([len(edges) - 1 for edges in all_edges])
    filtered_bin_indices = np.array(
        [indices[~combined_invalid_mask] - 1 for indices in all_bin_indices]
    )

    # Compute radix multipliers for mixed-radix encoding
    radix_multipliers = bins_per_dimension[::-1]
    radix_multipliers = np.cumprod(radix_multipliers) // radix_multipliers
    radix_multipliers = radix_multipliers[::-1]

    bin_codes = radix_multipliers @ filtered_bin_indices

    return valid_values, valid_weights, bin_codes.astype(np.intp)


def _compute_means(
    values: FloatArray,
    weights: FloatArray | None,
    bin_codes: NDArray[np.intp],
    compute_std_errors: bool,
) -> tuple[FloatArray, FloatArray | None, FloatArray]:
    """Compute weighted means for each bin."""
    if weights is None:
        weights = np.empty(0, dtype=np.float64)

    n_bins = len(np.unique(bin_codes))

    means, agg_weights, sort_indices = _compute_weighted_means_numba(
        values, weights, bin_codes, n_bins
    )

    standard_errors = None
    if compute_std_errors:
        standard_errors = _compute_std_errors_numba(
            values, weights, bin_codes, sort_indices, means, agg_weights
        )

    return means, standard_errors, agg_weights


@numba.njit  # type: ignore[untyped-decorator]
def _compute_weighted_means_numba(  # pragma: no cover
    values: FloatArray,
    weights: FloatArray,
    bin_codes: NDArray[np.intp],
    n_bins: int,
) -> tuple[FloatArray, FloatArray, NDArray[np.intp]]:
    """Compute weighted means using numba for performance."""
    _n_observations, n_columns = values.shape
    has_weights = len(weights) > 0

    means = np.zeros((n_bins, n_columns), dtype=np.float64)
    agg_weights = np.zeros(n_bins, dtype=np.float64)

    # Sort by bin codes for efficient sequential grouping
    sort_indices = np.argsort(bin_codes)

    bin_idx = -1
    prev_code = -1

    for idx in sort_indices:
        code = bin_codes[idx]
        weight = weights[idx] if has_weights else 1.0

        # Detect transition to new bin
        if code != prev_code:
            bin_idx += 1
            prev_code = code

        # Accumulate weighted values
        for col in range(n_columns):
            means[bin_idx, col] += values[idx, col] * weight
        agg_weights[bin_idx] += weight

    # Normalize accumulated values to get means
    for i in range(n_bins):
        for col in range(n_columns):
            means[i, col] /= agg_weights[i]

    return means, agg_weights, sort_indices


@numba.njit  # type: ignore[untyped-decorator]
def _compute_std_errors_numba(  # pragma: no cover
    values: FloatArray,
    weights: FloatArray,
    bin_codes: NDArray[np.intp],
    sort_indices: NDArray[np.intp],
    means: FloatArray,
    agg_weights: FloatArray,
) -> FloatArray:
    """Compute standard errors of weighted means.

    Uses the formula for standard error of a weighted mean:
        standard_error = weighted_std * sqrt(sum(normalized_weights^2))

    where normalized_weight = weight / sum(weights)
    """
    n_bins, n_columns = means.shape
    has_weights = len(weights) > 0

    sum_squared_weights = np.zeros(n_bins, dtype=np.float64)
    sum_squared_deviations = np.zeros((n_bins, n_columns), dtype=np.float64)

    bin_idx = -1
    prev_code = -1

    for idx in sort_indices:
        code = bin_codes[idx]
        weight = weights[idx] if has_weights else 1.0

        if code != prev_code:
            bin_idx += 1
            prev_code = code

        # Accumulate squared deviations from mean (for variance calculation)
        for col in range(n_columns):
            deviation = values[idx, col] - means[bin_idx, col]
            sum_squared_deviations[bin_idx, col] += weight * deviation * deviation
        sum_squared_weights[bin_idx] += weight * weight

    # Compute standard errors from accumulated values
    standard_errors = np.zeros((n_bins, n_columns), dtype=np.float64)
    for i in range(n_bins):
        total_weight = agg_weights[i]
        # Denominator for weighted variance (reliability weights)
        variance_denom = total_weight - sum_squared_weights[i] / total_weight
        # Factor to convert std dev to standard error of mean
        se_factor = np.sqrt(sum_squared_weights[i] / (total_weight * total_weight))

        for col in range(n_columns):
            weighted_var = sum_squared_deviations[i, col] / variance_denom
            weighted_std = np.sqrt(weighted_var)
            standard_errors[i, col] = weighted_std * se_factor

    return standard_errors
