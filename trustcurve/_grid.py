"""Grid generation and data utilities."""

from __future__ import annotations

from typing import Callable

import numpy as np
import polars as pl


def resolve_dataframe(X: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Materialize a LazyFrame if needed."""
    if isinstance(X, pl.LazyFrame):
        return X.collect()
    return X


def sample_if_needed(
    X: pl.DataFrame,
    max_samples: int | None,
    seed: int = 42,
) -> pl.DataFrame:
    """Subsample rows if the DataFrame exceeds *max_samples*."""
    if max_samples is None or len(X) <= max_samples:
        return X
    return X.sample(n=max_samples, seed=seed)


def feature_grid(
    X: pl.DataFrame,
    feature: str,
    grid_size: int = 50,
    grid_values: np.ndarray | None = None,
    percentile_range: tuple[float, float] = (0.02, 0.98),
) -> np.ndarray:
    """Build a 1-D grid for a numeric feature.

    Uses quantiles of the actual distribution so the grid covers where data
    actually lives, not wasted on empty tails.
    """
    if grid_values is not None:
        return np.asarray(grid_values, dtype=np.float64)

    col = X[feature].drop_nulls().to_numpy()
    if len(col) == 0:
        raise ValueError(f"Feature '{feature}' has no non-null values.")

    lo, hi = percentile_range
    quantiles = np.linspace(lo, hi, grid_size)
    grid = np.unique(np.quantile(col, quantiles))
    return grid.astype(np.float64)


def predict_in_batches(
    predict_fn: Callable[..., np.ndarray],
    X_np: np.ndarray,
    batch_size: int | None = None,
) -> np.ndarray:
    """Call *predict_fn* in chunks to bound memory usage."""
    if batch_size is None or len(X_np) <= batch_size:
        return np.asarray(predict_fn(X_np))

    parts = []
    for start in range(0, len(X_np), batch_size):
        chunk = X_np[start : start + batch_size]
        parts.append(np.asarray(predict_fn(chunk)))
    return np.concatenate(parts, axis=0)
