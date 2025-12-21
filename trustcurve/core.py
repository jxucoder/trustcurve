"""
TrustCurve: XGBoost with loss-to-curvature screening for noisy labels.

The key insight: for log-loss, the ratio R = Loss / Hessian distinguishes
noise from hard samples:
  - Hard samples: high loss, high Hessian (uncertain) → low R
  - Noisy samples: high loss, low Hessian (confidently wrong) → high R
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def compute_curvature_ratio(y_true: np.ndarray, y_pred_proba: np.ndarray) -> np.ndarray:
    """
    Compute loss-to-Hessian ratio for each sample.

    R = -log(p_correct) / (p_correct * (1 - p_correct))

    Higher R indicates likely mislabeled samples.
    """
    eps = 1e-15

    if y_pred_proba.ndim == 1:
        # Binary: y_pred_proba is P(y=1)
        p = np.clip(y_pred_proba, eps, 1 - eps)
        loss = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
        hessian = p * (1 - p)
    else:
        # Multiclass: use P(true class)
        n = len(y_true)
        p_correct = y_pred_proba[np.arange(n), y_true.astype(int)]
        p_correct = np.clip(p_correct, eps, 1 - eps)
        loss = -np.log(p_correct)
        hessian = p_correct * (1 - p_correct)

    return loss / (hessian + eps)


def gini(x: np.ndarray) -> float:
    """Gini coefficient measuring inequality in distribution."""
    x = np.sort(x.flatten())
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return (2 * idx - n - 1).dot(x) / (n * x.sum())


def estimate_noise_level(ratio: np.ndarray) -> float:
    """
    Estimate noise level from ratio distribution via Gini coefficient.

    Clean data: ratios are similar → low Gini
    Noisy data: some ratios explode → high Gini
    """
    g = gini(ratio)
    # Map Gini [0.2, 0.6] → noise [0, 1]
    return float(np.clip((g - 0.2) * 2.5, 0, 1))


def curvature_weights(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    tolerance: float = 5.0,
    power: float = 4.0,
    adaptive: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute sample weights from curvature ratios.

    Returns: (weights, ratios, screening_strength)
    """
    ratio = compute_curvature_ratio(y_true, y_pred_proba)
    noise_level = estimate_noise_level(ratio)

    # Adaptive tolerance: less aggressive on clean data
    if adaptive:
        scale = 1.0 + 2.0 * (1.0 - min(noise_level * 5, 1.0))
        tol = tolerance * scale
    else:
        tol = tolerance

    # Data-driven floor: median + 2*IQR
    med = np.median(ratio)
    iqr = np.percentile(ratio, 75) - np.percentile(ratio, 25)
    tol = max(tol, med + 2 * iqr)

    # Soft thresholding: w = 1 / (1 + (R/tol)^power)
    raw_w = 1.0 / (1.0 + (ratio / tol) ** power)

    # Blend with uniform weights based on detected noise
    if adaptive:
        strength = min(noise_level * 5, 1.0)
        weights = (1 - strength) + strength * raw_w
    else:
        strength = 1.0
        weights = raw_w

    return weights, ratio, strength


class TrustCurveClassifier(BaseEstimator, ClassifierMixin):
    """
    XGBoost classifier with automatic noise screening.

    Trains an initial model, identifies likely mislabeled samples via
    loss-to-Hessian ratio, then continues training with downweighted noise.

    Parameters
    ----------
    n_estimators : int
        Total boosting rounds.
    warmup_trees : int or "auto"
        Rounds before screening. "auto" waits for confident predictions.
    tolerance : float
        Ratio threshold for soft weighting.
    power : float
        Sharpness of weight decay.
    adaptive : bool
        Scale screening strength by detected noise level.
    xgb_params : dict
        Additional XGBoost parameters.
    verbose : int
        0=silent, 1=summary, 2=detailed.
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        warmup_trees: int | str = "auto",
        tolerance: float = 5.0,
        power: float = 4.0,
        adaptive: bool = True,
        xgb_params: dict | None = None,
        verbose: int = 1,
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.warmup_trees = warmup_trees
        self.tolerance = tolerance
        self.power = power
        self.adaptive = adaptive
        self.xgb_params = xgb_params
        self.verbose = verbose
        self.random_state = random_state

    def _log(self, msg: str, level: int = 1):
        if self.verbose >= level:
            print(msg)

    def _get_xgb_params(self) -> dict:
        params = {
            "objective": "binary:logistic" if self.n_classes_ == 2 else "multi:softprob",
            "eval_metric": "logloss",
            "eta": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbosity": 0,
        }
        if self.n_classes_ > 2:
            params["num_class"] = self.n_classes_
        if self.random_state is not None:
            params["seed"] = self.random_state
        if self.xgb_params:
            params.update(self.xgb_params)
        return params

    def _is_ready(self, X, y_enc) -> bool:
        """Check if model has enough confident predictions to screen."""
        proba = self.model_.predict(xgb.DMatrix(X))
        if self.n_classes_ == 2:
            p = proba
        else:
            p = proba.reshape(-1, self.n_classes_)[np.arange(len(y_enc)), y_enc]
        hessian = p * (1 - p)
        # Ready when ≥1% of samples have confidence >87% (hessian < 0.1)
        return (hessian < 0.1).mean() >= 0.01

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        """Fit the classifier."""
        X, y = check_X_y(X, y)
        n = len(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        label_map = {c: i for i, c in enumerate(self.classes_)}
        y_enc = np.array([label_map[yi] for yi in y])

        base_w = np.ones(n) if sample_weight is None else sample_weight.copy()
        params = self._get_xgb_params()
        dtrain = xgb.DMatrix(X, label=y_enc, weight=base_w)

        # Phase 1: Warmup
        if self.warmup_trees == "auto":
            self._log("TrustCurve: auto-warmup...")
            step = 10
            max_warmup = min(100, self.n_estimators // 3)

            # Subsample for efficiency
            if n > 10000:
                rng = np.random.RandomState(self.random_state or 42)
                idx = rng.choice(n, 10000, replace=False)
                X_sub, y_sub = X[idx], y_enc[idx]
            else:
                X_sub, y_sub = X, y_enc

            self.model_ = xgb.train(params, dtrain, num_boost_round=step)
            warmup = step

            for _ in range(max_warmup // step):
                if self._is_ready(X_sub, y_sub):
                    self._log(f"  Ready at {warmup} trees", level=2)
                    break
                self.model_ = xgb.train(params, dtrain, num_boost_round=step, xgb_model=self.model_)
                warmup += step

            self.warmup_trees_ = warmup
        else:
            self._log(f"TrustCurve: warmup ({self.warmup_trees} trees)...")
            self.model_ = xgb.train(params, dtrain, num_boost_round=self.warmup_trees)
            self.warmup_trees_ = self.warmup_trees

        # Phase 2: Compute weights
        proba = self.model_.predict(xgb.DMatrix(X))
        if self.n_classes_ == 2:
            y_proba = proba
        else:
            y_proba = proba.reshape(-1, self.n_classes_)

        weights, ratio, strength = curvature_weights(
            y_enc, y_proba,
            tolerance=self.tolerance,
            power=self.power,
            adaptive=self.adaptive,
        )

        self.curvature_ratio_ = ratio
        self.weights_ = weights
        self.screening_strength_ = strength
        self.estimated_noise_ = estimate_noise_level(ratio)

        self._log(f"  Noise: ~{self.estimated_noise_*100:.0f}%, screening: {strength*100:.0f}%")

        # Phase 3: Continue with weighted samples
        remaining = self.n_estimators - self.warmup_trees_
        if remaining > 0:
            dtrain = xgb.DMatrix(X, label=y_enc, weight=base_w * weights)
            self.model_ = xgb.train(params, dtrain, num_boost_round=remaining, xgb_model=self.model_)

        self._log(f"  Done: {self.model_.num_boosted_rounds()} trees")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        check_is_fitted(self)
        X = check_array(X)
        proba = self.model_.predict(xgb.DMatrix(X))
        if self.n_classes_ == 2:
            return np.column_stack([1 - proba, proba])
        return proba.reshape(-1, self.n_classes_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy."""
        return (self.predict(X) == y).mean()

    def get_noisy_samples(self, percentile: float = 10.0) -> np.ndarray:
        """Get indices of likely mislabeled samples (top percentile by ratio)."""
        check_is_fitted(self)
        thresh = np.percentile(self.curvature_ratio_, 100 - percentile)
        return np.where(self.curvature_ratio_ > thresh)[0]

    def get_curvature_ratio(self) -> np.ndarray:
        """Get loss-to-Hessian ratio for each training sample."""
        check_is_fitted(self)
        return self.curvature_ratio_.copy()

    def get_weights(self) -> np.ndarray:
        """Get computed sample weights."""
        check_is_fitted(self)
        return self.weights_.copy()
