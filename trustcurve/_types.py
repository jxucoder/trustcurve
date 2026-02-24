"""Result types for TrustCurve computations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class PDPResult:
    """Result of a partial dependence computation.

    Attributes:
        feature: Name of the feature.
        grid_values: 1-D array of grid points where the feature was evaluated.
        avg_predictions: 1-D array of mean predictions at each grid point.
        ice_lines: 2-D array (n_samples, n_grid) of individual predictions.
                   None when only the average was requested.
    """

    feature: str
    grid_values: np.ndarray
    avg_predictions: np.ndarray
    ice_lines: np.ndarray | None = None

    def plot(self, ax: Axes | None = None, ice: bool = False, centered: bool = False) -> Figure:
        from trustcurve._plot import plot_pdp

        return plot_pdp(self, ax=ax, ice=ice, centered=centered)


@dataclass(frozen=True)
class ICEResult:
    """Result of an individual conditional expectation computation.

    Attributes:
        feature: Name of the feature.
        grid_values: 1-D array of grid points.
        ice_lines: 2-D array (n_samples, n_grid) of individual predictions.
        avg_predictions: 1-D array of mean predictions (the PDP line).
        centered: Whether the ICE lines are centered (c-ICE).
    """

    feature: str
    grid_values: np.ndarray
    ice_lines: np.ndarray
    avg_predictions: np.ndarray
    centered: bool = False

    def plot(self, ax: Axes | None = None, show_avg: bool = True) -> Figure:
        from trustcurve._plot import plot_ice

        return plot_ice(self, ax=ax, show_avg=show_avg)


@dataclass(frozen=True)
class ALEResult:
    """Result of an accumulated local effects computation.

    Attributes:
        feature: Name of the feature.
        bin_edges: 1-D array of bin boundaries.
        ale_values: 1-D array of ALE values at bin centers.
        bin_counts: 1-D array of sample counts per bin.
    """

    feature: str
    bin_edges: np.ndarray
    ale_values: np.ndarray
    bin_counts: np.ndarray

    def plot(self, ax: Axes | None = None) -> Figure:
        from trustcurve._plot import plot_ale

        return plot_ale(self, ax=ax)


@dataclass(frozen=True)
class InteractionResult:
    """Result of a 2-D partial dependence computation.

    Attributes:
        features: Tuple of the two feature names.
        grid_values: Tuple of two 1-D arrays (one per feature).
        avg_predictions: 2-D array (n_grid_0, n_grid_1) of mean predictions.
    """

    features: tuple[str, str]
    grid_values: tuple[np.ndarray, np.ndarray]
    avg_predictions: np.ndarray

    def plot(self, ax: Axes | None = None) -> Figure:
        from trustcurve._plot import plot_interaction

        return plot_interaction(self, ax=ax)
