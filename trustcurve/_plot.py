"""Plotting layer for TrustCurve results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from trustcurve._types import ALEResult, ICEResult, InteractionResult, PDPResult


def _ensure_ax(ax: Axes | None) -> tuple[Figure, Axes]:
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    return fig, ax


def plot_pdp(
    result: PDPResult,
    *,
    ax: Axes | None = None,
    ice: bool = False,
    centered: bool = False,
) -> Figure:
    """Plot partial dependence, optionally with ICE lines."""
    fig, ax = _ensure_ax(ax)

    if ice and result.ice_lines is not None:
        lines = result.ice_lines
        if centered:
            lines = lines - lines[:, [0]]
        for row in lines:
            ax.plot(result.grid_values, row, color="steelblue", alpha=0.05, linewidth=0.5)

    avg = result.avg_predictions
    if centered and result.ice_lines is not None:
        avg = (result.ice_lines - result.ice_lines[:, [0]]).mean(axis=0)

    ax.plot(result.grid_values, avg, color="black", linewidth=2, label="PDP")
    ax.set_xlabel(result.feature)
    ax.set_ylabel("Partial dependence")
    ax.set_title(f"PDP — {result.feature}")
    return fig


def plot_ice(
    result: ICEResult,
    *,
    ax: Axes | None = None,
    show_avg: bool = True,
) -> Figure:
    """Plot individual conditional expectation curves."""
    fig, ax = _ensure_ax(ax)

    for row in result.ice_lines:
        ax.plot(result.grid_values, row, color="steelblue", alpha=0.08, linewidth=0.5)

    if show_avg:
        ax.plot(result.grid_values, result.avg_predictions, color="black", linewidth=2, label="avg")

    label = "c-ICE" if result.centered else "ICE"
    ax.set_xlabel(result.feature)
    ax.set_ylabel("Prediction" if not result.centered else "Centered prediction")
    ax.set_title(f"{label} — {result.feature}")
    return fig


def plot_ale(
    result: ALEResult,
    *,
    ax: Axes | None = None,
) -> Figure:
    """Plot accumulated local effects."""
    fig, ax = _ensure_ax(ax)

    # Plot ALE at bin centers
    centers = (result.bin_edges[:-1] + result.bin_edges[1:]) / 2
    ax.plot(centers, result.ale_values, color="black", linewidth=2)
    ax.fill_between(centers, result.ale_values, alpha=0.15, color="steelblue")

    # Rug plot showing data density
    ax.plot(
        centers,
        np.full_like(centers, result.ale_values.min()),
        "|",
        color="gray",
        alpha=0.3,
        markersize=4,
    )

    ax.set_xlabel(result.feature)
    ax.set_ylabel("ALE")
    ax.set_title(f"ALE — {result.feature}")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    return fig


def plot_interaction(
    result: InteractionResult,
    *,
    ax: Axes | None = None,
) -> Figure:
    """Plot 2-D partial dependence as a heatmap."""
    fig, ax = _ensure_ax(ax)

    grid0, grid1 = result.grid_values
    im = ax.pcolormesh(
        grid1,
        grid0,
        result.avg_predictions,
        shading="auto",
        cmap="viridis",
    )
    fig.colorbar(im, ax=ax, label="Partial dependence")
    ax.set_xlabel(result.features[1])
    ax.set_ylabel(result.features[0])
    ax.set_title(f"Interaction — {result.features[0]} × {result.features[1]}")
    return fig
