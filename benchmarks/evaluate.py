"""Evaluation framework for TrustCurve benchmarks."""

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from trustcurve import TrustCurveClassifier


@dataclass
class EvalResult:
    dataset: str
    noise_level: float
    method: str
    accuracy: float
    f1: float
    auc: float
    train_time: float
    warmup_trees: int = 0
    screening_strength: float = 0.0


@dataclass
class BenchmarkResults:
    results: list = field(default_factory=list)

    def add(self, result: EvalResult):
        self.results.append(result)

    def summary_by_noise(self) -> dict:
        summary = defaultdict(lambda: {"trustcurve": [], "baseline": []})
        for r in self.results:
            summary[r.noise_level][r.method].append(r.accuracy)
        return {
            nl: {k: np.mean(v) if v else 0 for k, v in methods.items()}
            for nl, methods in summary.items()
        }

    def win_rate(self) -> float:
        summary = self.summary_by_noise()
        wins = sum(1 for v in summary.values() if v["trustcurve"] >= v["baseline"])
        return wins / len(summary) if summary else 0


def inject_noise(y: np.ndarray, rate: float, n_classes: int, seed: int = 42) -> np.ndarray:
    """Flip labels randomly at given rate."""
    if rate == 0:
        return y.copy()

    rng = np.random.RandomState(seed)
    y_noisy = y.copy()
    flip_idx = rng.choice(len(y), size=int(len(y) * rate), replace=False)

    for i in flip_idx:
        choices = [c for c in range(n_classes) if c != y_noisy[i]]
        y_noisy[i] = rng.choice(choices)

    return y_noisy


def evaluate_single(X_train, X_test, y_train, y_test, method="trustcurve", n_estimators=300):
    """Evaluate a single model."""
    start = time.time()

    if method == "trustcurve":
        model = TrustCurveClassifier(n_estimators=n_estimators, verbose=0)
        model.fit(X_train, y_train)
        warmup = getattr(model, 'warmup_trees_', 0)
        screening = getattr(model, 'screening_strength_', 0)
    else:
        model = xgb.XGBClassifier(n_estimators=n_estimators, verbosity=0, eval_metric='logloss')
        model.fit(X_train, y_train)
        warmup, screening = 0, 0

    train_time = time.time() - start
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    n_classes = len(np.unique(y_test))
    if n_classes == 2:
        auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        try:
            auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        except ValueError:
            auc = 0.0

    return {
        "accuracy": acc, "f1": f1, "auc": auc,
        "train_time": train_time, "warmup_trees": warmup, "screening_strength": screening,
    }


def evaluate_dataset(X, y, name, noise_levels=[0.0], n_trials=5, n_estimators=300):
    """Evaluate TrustCurve vs baseline across noise levels."""
    results = BenchmarkResults()
    n_classes = len(np.unique(y))

    for noise in noise_levels:
        for trial in range(n_trials):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42 + trial, stratify=y
            )
            y_train_noisy = inject_noise(y_train, noise, n_classes, seed=42 + trial)

            for method in ["trustcurve", "baseline"]:
                m = evaluate_single(X_train, X_test, y_train_noisy, y_test, method, n_estimators)
                results.add(EvalResult(
                    dataset=name, noise_level=noise, method=method,
                    accuracy=m["accuracy"], f1=m["f1"], auc=m["auc"],
                    train_time=m["train_time"], warmup_trees=m["warmup_trees"],
                    screening_strength=m["screening_strength"],
                ))

    return results


def print_results(results: BenchmarkResults, title: str = "Results"):
    """Print results table."""
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")

    grouped = defaultdict(lambda: defaultdict(lambda: {"trustcurve": [], "baseline": []}))
    for r in results.results:
        grouped[r.dataset][r.noise_level][r.method].append(r.accuracy)

    print(f"\n{'Dataset':<16} {'Noise':<8} {'TrustCurve':<12} {'Baseline':<12} {'Diff':<8}")
    print('-' * 56)

    wins, total = 0, 0
    for ds in sorted(grouped):
        for noise in sorted(grouped[ds]):
            tc = np.mean(grouped[ds][noise]["trustcurve"])
            bl = np.mean(grouped[ds][noise]["baseline"])
            diff = tc - bl
            print(f"{ds:<16} {noise:<8.0%} {tc:.4f}       {bl:.4f}       {diff:+.4f}")
            wins += diff >= 0
            total += 1

    print('-' * 56)
    print(f"Win rate: {wins}/{total} ({wins/total:.0%})")


def generate_synthetic(n_samples=2000, n_features=20, n_informative=15, n_classes=2, noise_rate=0.0, seed=42):
    """Generate synthetic data with optional noise."""
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=n_informative,
        n_redundant=n_features - n_informative, n_classes=n_classes,
        n_clusters_per_class=1, random_state=seed,
    )
    return X, y, inject_noise(y, noise_rate, n_classes, seed=seed)
