"""Tests for TrustCurve."""

import numpy as np
import polars as pl
import pytest

import trustcurve as tc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_data(n: int = 200, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "x0": rng.standard_normal(n),
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
    })


def _linear_predict(X: np.ndarray) -> np.ndarray:
    """y = 2*x0 + 0.5*x1  (x2 is noise)."""
    return 2 * X[:, 0] + 0.5 * X[:, 1]


def _interaction_predict(X: np.ndarray) -> np.ndarray:
    """y = x0 * x1."""
    return X[:, 0] * X[:, 1]


@pytest.fixture
def data() -> pl.DataFrame:
    return _make_data()


# ---------------------------------------------------------------------------
# PDP
# ---------------------------------------------------------------------------

class TestPDP:
    def test_shape(self, data: pl.DataFrame) -> None:
        result = tc.pdp(_linear_predict, data, "x0", grid_size=20)
        assert result.grid_values.shape == (20,)
        assert result.avg_predictions.shape == (20,)
        assert result.ice_lines is not None
        assert result.ice_lines.shape[1] == 20

    def test_monotonic_for_linear(self, data: pl.DataFrame) -> None:
        result = tc.pdp(_linear_predict, data, "x0", grid_size=30, max_samples=None)
        # PDP of x0 should be roughly monotonically increasing (coeff=2)
        diffs = np.diff(result.avg_predictions)
        assert np.all(diffs > 0), "PDP should be monotonically increasing for x0"

    def test_flat_for_noise(self, data: pl.DataFrame) -> None:
        result = tc.pdp(_linear_predict, data, "x2", grid_size=20, max_samples=None)
        # x2 has zero coefficient, PDP should be roughly flat
        spread = result.avg_predictions.max() - result.avg_predictions.min()
        assert spread < 0.5, f"PDP for noise feature should be flat, got spread={spread:.3f}"

    def test_custom_grid(self, data: pl.DataFrame) -> None:
        grid = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = tc.pdp(_linear_predict, data, "x0", grid_values=grid)
        np.testing.assert_array_equal(result.grid_values, grid)

    def test_plot_returns_figure(self, data: pl.DataFrame) -> None:
        import matplotlib

        matplotlib.use("Agg")
        result = tc.pdp(_linear_predict, data, "x0", grid_size=10)
        fig = result.plot()
        assert fig is not None

    def test_accepts_lazyframe(self, data: pl.DataFrame) -> None:
        lazy = data.lazy()
        result = tc.pdp(_linear_predict, lazy, "x0", grid_size=10)
        assert result.avg_predictions.shape[0] == 10


# ---------------------------------------------------------------------------
# ICE
# ---------------------------------------------------------------------------

class TestICE:
    def test_shape(self, data: pl.DataFrame) -> None:
        result = tc.ice(_linear_predict, data, "x0", grid_size=15, max_samples=50)
        assert result.ice_lines.shape == (50, 15)
        assert result.avg_predictions.shape == (15,)
        assert result.centered is False

    def test_centered(self, data: pl.DataFrame) -> None:
        result = tc.ice(_linear_predict, data, "x0", centered=True, grid_size=15, max_samples=50)
        assert result.centered is True
        # All lines should start at 0
        np.testing.assert_allclose(result.ice_lines[:, 0], 0.0)

    def test_plot(self, data: pl.DataFrame) -> None:
        import matplotlib

        matplotlib.use("Agg")
        result = tc.ice(_linear_predict, data, "x0", grid_size=10, max_samples=30)
        fig = result.plot()
        assert fig is not None


# ---------------------------------------------------------------------------
# ALE
# ---------------------------------------------------------------------------

class TestALE:
    def test_shape(self, data: pl.DataFrame) -> None:
        result = tc.ale(_linear_predict, data, "x0", bins=20)
        assert len(result.bin_edges) == len(result.ale_values) + 1
        assert len(result.bin_counts) == len(result.ale_values)

    def test_centered(self, data: pl.DataFrame) -> None:
        result = tc.ale(_linear_predict, data, "x0", bins=20, max_samples=None)
        # ALE should be roughly centered around zero
        weighted_mean = np.average(result.ale_values, weights=result.bin_counts)
        assert abs(weighted_mean) < 0.3, f"ALE not centered: weighted_mean={weighted_mean:.3f}"

    def test_increasing_for_positive_coeff(self, data: pl.DataFrame) -> None:
        result = tc.ale(_linear_predict, data, "x0", bins=20, max_samples=None)
        # ALE for x0 (coeff=2) should be mostly increasing
        diffs = np.diff(result.ale_values)
        frac_positive = (diffs > 0).mean()
        assert frac_positive > 0.7, f"ALE should be mostly increasing, got {frac_positive:.0%}"

    def test_plot(self, data: pl.DataFrame) -> None:
        import matplotlib

        matplotlib.use("Agg")
        result = tc.ale(_linear_predict, data, "x0", bins=15)
        fig = result.plot()
        assert fig is not None


# ---------------------------------------------------------------------------
# Interaction
# ---------------------------------------------------------------------------

class TestInteraction:
    def test_shape(self, data: pl.DataFrame) -> None:
        result = tc.interaction(_linear_predict, data, ["x0", "x1"], grid_size=10)
        assert result.avg_predictions.shape[0] <= 10
        assert result.avg_predictions.shape[1] <= 10
        assert result.features == ("x0", "x1")

    def test_interaction_effect(self) -> None:
        data = _make_data(n=500)
        result = tc.interaction(_interaction_predict, data, ["x0", "x1"], grid_size=10)
        # For y = x0*x1, the interaction surface should show clear variation
        spread = result.avg_predictions.max() - result.avg_predictions.min()
        assert spread > 0.5, "Interaction plot should show meaningful variation for x0*x1"

    def test_plot(self, data: pl.DataFrame) -> None:
        import matplotlib

        matplotlib.use("Agg")
        result = tc.interaction(_linear_predict, data, ["x0", "x1"], grid_size=8)
        fig = result.plot()
        assert fig is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_value_feature(self) -> None:
        df = pl.DataFrame({"x0": [1.0] * 100, "x1": np.random.randn(100).tolist()})
        result = tc.pdp(_linear_predict, df, "x0", grid_size=10)
        # Should get a single grid point since all values are the same
        assert len(result.grid_values) >= 1

    def test_empty_feature_raises(self) -> None:
        df = pl.DataFrame({"x0": pl.Series([], dtype=pl.Float64), "x1": pl.Series([], dtype=pl.Float64)})
        with pytest.raises(ValueError, match="no non-null"):
            tc.pdp(_linear_predict, df, "x0")

    def test_sampling_reduces_rows(self) -> None:
        big = _make_data(n=5000)
        result = tc.pdp(_linear_predict, big, "x0", grid_size=5, max_samples=100)
        assert result.ice_lines is not None
        assert result.ice_lines.shape[0] == 100
