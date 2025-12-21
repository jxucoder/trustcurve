"""TrustCurve: Noise-robust XGBoost via loss-to-curvature screening."""

from trustcurve.core import (
    TrustCurveClassifier,
    compute_curvature_ratio,
    curvature_weights,
    estimate_noise_level,
)

__version__ = "0.1.0"
__all__ = [
    "TrustCurveClassifier",
    "compute_curvature_ratio",
    "curvature_weights",
    "estimate_noise_level",
]
