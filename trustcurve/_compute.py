"""Core computation for PDP, ICE, and ALE."""

from __future__ import annotations

from typing import Callable

import numpy as np
import polars as pl

from trustcurve._grid import (
    feature_grid,
    predict_in_batches,
    resolve_dataframe,
    sample_if_needed,
)
from trustcurve._types import ALEResult, ICEResult, InteractionResult, PDPResult


def pdp(
    predict_fn: Callable[..., np.ndarray],
    X: pl.DataFrame | pl.LazyFrame,
    feature: str,
    *,
    grid_size: int = 50,
    grid_values: np.ndarray | None = None,
    max_samples: int | None = 2000,
    batch_size: int | None = None,
    seed: int = 42,
) -> PDPResult:
    """Compute partial dependence for a single feature.

    Parameters
    ----------
    predict_fn : callable
        Any function that maps an (n, p) numpy array to an (n,) array of
        predictions (regression) or probabilities (classification).
    X : polars DataFrame or LazyFrame
        The dataset.  If a LazyFrame is passed it is collected first.
    feature : str
        Column name to compute PDP for.
    grid_size : int
        Number of quantile-spaced grid points (default 50).
    grid_values : array-like, optional
        Explicit grid values; overrides *grid_size*.
    max_samples : int or None
        Cap on the number of rows used (default 2000).  Set ``None`` to use
        all rows (slow on large data).
    batch_size : int or None
        If set, predictions are computed in chunks of this size.
    seed : int
        Random seed for sampling.
    """
    df = sample_if_needed(resolve_dataframe(X), max_samples, seed=seed)
    grid = feature_grid(df, feature, grid_size=grid_size, grid_values=grid_values)

    X_np = df.to_numpy()
    feature_idx = df.columns.index(feature)

    # ice_lines[i, j] = prediction for sample i when feature = grid[j]
    ice_lines = np.empty((len(X_np), len(grid)), dtype=np.float64)

    for j, val in enumerate(grid):
        X_mod = X_np.copy()
        X_mod[:, feature_idx] = val
        ice_lines[:, j] = predict_in_batches(predict_fn, X_mod, batch_size)

    avg = ice_lines.mean(axis=0)
    return PDPResult(
        feature=feature,
        grid_values=grid,
        avg_predictions=avg,
        ice_lines=ice_lines,
    )


def ice(
    predict_fn: Callable[..., np.ndarray],
    X: pl.DataFrame | pl.LazyFrame,
    feature: str,
    *,
    centered: bool = False,
    grid_size: int = 50,
    grid_values: np.ndarray | None = None,
    max_samples: int | None = 500,
    batch_size: int | None = None,
    seed: int = 42,
) -> ICEResult:
    """Compute Individual Conditional Expectation curves.

    Parameters
    ----------
    centered : bool
        If True, return centered ICE (c-ICE): each curve is shifted so it
        starts at zero.  This highlights heterogeneity in feature effects.
    max_samples : int or None
        Default 500 (ICE shows individual lines, so fewer samples keeps plots
        readable).
    """
    result = pdp(
        predict_fn,
        X,
        feature,
        grid_size=grid_size,
        grid_values=grid_values,
        max_samples=max_samples,
        batch_size=batch_size,
        seed=seed,
    )

    lines = result.ice_lines
    assert lines is not None

    if centered:
        lines = lines - lines[:, [0]]
        avg = lines.mean(axis=0)
    else:
        avg = result.avg_predictions

    return ICEResult(
        feature=feature,
        grid_values=result.grid_values,
        ice_lines=lines,
        avg_predictions=avg,
        centered=centered,
    )


def ale(
    predict_fn: Callable[..., np.ndarray],
    X: pl.DataFrame | pl.LazyFrame,
    feature: str,
    *,
    bins: int = 50,
    max_samples: int | None = None,
    batch_size: int | None = None,
    seed: int = 42,
) -> ALEResult:
    """Compute Accumulated Local Effects for a single feature.

    ALE plots are preferred over PDP when features are correlated because
    they avoid the extrapolation problem.

    Parameters
    ----------
    bins : int
        Number of quantile-based bins (default 50).
    max_samples : int or None
        ALE uses each row only once, so it's cheaper than PDP.  ``None``
        means use all data (the default).
    """
    df = sample_if_needed(resolve_dataframe(X), max_samples, seed=seed)
    col = df[feature].drop_nulls().to_numpy().astype(np.float64)

    # Quantile-based bin edges
    percentiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.unique(np.quantile(col, percentiles))
    n_bins = len(bin_edges) - 1

    if n_bins < 1:
        raise ValueError(f"Feature '{feature}' has too few unique values for ALE binning.")

    X_np = df.to_numpy().astype(np.float64)
    feature_idx = df.columns.index(feature)
    feature_col = X_np[:, feature_idx]

    ale_values = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    for k in range(n_bins):
        lo, hi = bin_edges[k], bin_edges[k + 1]
        if k < n_bins - 1:
            mask = (feature_col >= lo) & (feature_col < hi)
        else:
            mask = (feature_col >= lo) & (feature_col <= hi)

        if mask.sum() == 0:
            continue

        X_lo = X_np[mask].copy()
        X_hi = X_np[mask].copy()
        X_lo[:, feature_idx] = lo
        X_hi[:, feature_idx] = hi

        preds_lo = predict_in_batches(predict_fn, X_lo, batch_size)
        preds_hi = predict_in_batches(predict_fn, X_hi, batch_size)

        ale_values[k] = (preds_hi - preds_lo).mean()
        bin_counts[k] = int(mask.sum())

    # Accumulate
    ale_values = np.cumsum(ale_values)

    # Center: subtract weighted mean so ALE integrates to zero
    total = bin_counts.sum()
    if total > 0:
        weighted_mean = np.sum(ale_values * bin_counts) / total
        ale_values -= weighted_mean

    return ALEResult(
        feature=feature,
        bin_edges=bin_edges,
        ale_values=ale_values,
        bin_counts=bin_counts,
    )


def interaction(
    predict_fn: Callable[..., np.ndarray],
    X: pl.DataFrame | pl.LazyFrame,
    features: tuple[str, str] | list[str],
    *,
    grid_size: int = 25,
    max_samples: int | None = 1000,
    batch_size: int | None = None,
    seed: int = 42,
) -> InteractionResult:
    """Compute 2-D partial dependence for a pair of features.

    Parameters
    ----------
    features : pair of str
        The two feature names.
    grid_size : int
        Grid resolution per feature (default 25 → 625 grid combos).
    """
    f0, f1 = features[0], features[1]
    df = sample_if_needed(resolve_dataframe(X), max_samples, seed=seed)

    grid0 = feature_grid(df, f0, grid_size=grid_size)
    grid1 = feature_grid(df, f1, grid_size=grid_size)

    X_np = df.to_numpy()
    idx0 = df.columns.index(f0)
    idx1 = df.columns.index(f1)

    avg_preds = np.empty((len(grid0), len(grid1)), dtype=np.float64)

    for i, v0 in enumerate(grid0):
        for j, v1 in enumerate(grid1):
            X_mod = X_np.copy()
            X_mod[:, idx0] = v0
            X_mod[:, idx1] = v1
            preds = predict_in_batches(predict_fn, X_mod, batch_size)
            avg_preds[i, j] = preds.mean()

    return InteractionResult(
        features=(f0, f1),
        grid_values=(grid0, grid1),
        avg_predictions=avg_preds,
    )
