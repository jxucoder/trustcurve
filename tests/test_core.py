"""Tests for TrustCurveClassifier."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from trustcurve import TrustCurveClassifier, compute_curvature_ratio, curvature_weights


@pytest.fixture
def binary_data():
    X, y = make_classification(n_samples=500, n_features=20, n_informative=10, random_state=42)
    return X, y


@pytest.fixture
def multiclass_data():
    X, y = make_classification(
        n_samples=500, n_features=20, n_informative=10,
        n_classes=3, n_clusters_per_class=1, random_state=42
    )
    return X, y


@pytest.fixture
def noisy_data():
    X, y = make_classification(n_samples=500, n_features=20, n_informative=10, random_state=42)
    rng = np.random.RandomState(42)
    noise_idx = rng.choice(len(y), size=int(0.2 * len(y)), replace=False)
    y_noisy = y.copy()
    y_noisy[noise_idx] = 1 - y_noisy[noise_idx]
    return X, y, y_noisy, noise_idx


class TestTrustCurveClassifier:

    def test_defaults(self):
        clf = TrustCurveClassifier()
        assert clf.n_estimators == 300
        assert clf.warmup_trees == "auto"
        assert clf.tolerance == 5.0
        assert clf.power == 4.0
        assert clf.adaptive is True

    def test_fit_binary(self, binary_data):
        X, y = binary_data
        clf = TrustCurveClassifier(n_estimators=50, verbose=0)
        clf.fit(X, y)

        assert hasattr(clf, 'model_')
        assert len(clf.curvature_ratio_) == len(y)
        assert len(clf.weights_) == len(y)

    def test_fit_multiclass(self, multiclass_data):
        X, y = multiclass_data
        clf = TrustCurveClassifier(n_estimators=50, verbose=0)
        clf.fit(X, y)

        assert clf.n_classes_ == 3

    def test_predict(self, binary_data):
        X, y = binary_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = TrustCurveClassifier(n_estimators=50, verbose=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        assert len(y_pred) == len(y_test)
        assert set(y_pred).issubset({0, 1})

    def test_predict_proba(self, binary_data):
        X, y = binary_data
        clf = TrustCurveClassifier(n_estimators=50, verbose=0)
        clf.fit(X, y)

        proba = clf.predict_proba(X)

        assert proba.shape == (len(y), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_score(self, binary_data):
        X, y = binary_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = TrustCurveClassifier(n_estimators=50, verbose=0)
        clf.fit(X_train, y_train)

        assert clf.score(X_test, y_test) > 0.7

    def test_get_noisy_samples(self, noisy_data):
        X, y, y_noisy, noise_idx = noisy_data

        clf = TrustCurveClassifier(n_estimators=50, verbose=0)
        clf.fit(X, y_noisy)

        detected = clf.get_noisy_samples(percentile=20)
        assert len(detected) == int(0.2 * len(y))

        overlap = len(set(detected) & set(noise_idx))
        recall = overlap / len(noise_idx)
        assert recall > 0.3, f"Noise detection recall too low: {recall:.2%}"

    def test_clean_data_preserves_weights(self, binary_data):
        X, y = binary_data
        clf = TrustCurveClassifier(n_estimators=50, verbose=0)
        clf.fit(X, y)

        # Clean data should have most weights near 1
        assert (clf.weights_ > 0.5).mean() > 0.8

    def test_sample_weight(self, binary_data):
        X, y = binary_data
        w = np.random.random(len(y))

        clf = TrustCurveClassifier(n_estimators=50, verbose=0)
        clf.fit(X, y, sample_weight=w)

        assert hasattr(clf, 'model_')


class TestStandaloneFunctions:

    def test_curvature_ratio_ordering(self):
        y = np.array([1, 1, 1])
        p = np.array([0.99, 0.6, 0.01])  # easy, hard, noise

        ratio = compute_curvature_ratio(y, p)

        assert ratio[0] < ratio[1] < ratio[2]
        assert ratio[2] > 100  # noise ratio should be very high

    def test_curvature_weights_downweights_noise(self):
        y = np.array([1, 1, 1, 0, 0, 0])
        p = np.array([0.99, 0.99, 0.01, 0.01, 0.01, 0.99])
        # Noisy: idx 2 (y=1, p=0.01), idx 5 (y=0, p=0.99)

        weights, _, _ = curvature_weights(y, p, adaptive=False)

        assert weights[2] < weights[0]
        assert weights[5] < weights[3]
