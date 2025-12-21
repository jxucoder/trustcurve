"""
Benchmark Datasets
==================

15+ tabular datasets commonly used in ML benchmarks.
Some have known label noise issues.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from sklearn.datasets import (
    fetch_openml,
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler


class NoiseLevel(Enum):
    """Expected noise level in dataset."""
    CLEAN = "clean"          # <5% noise
    LOW = "low"              # 5-10% noise
    MODERATE = "moderate"    # 10-20% noise
    HIGH = "high"            # >20% noise


@dataclass
class DatasetInfo:
    """Dataset metadata."""
    name: str
    description: str
    n_samples: int
    n_features: int
    n_classes: int
    noise_level: NoiseLevel
    source: str


def _preprocess(X, y) -> tuple[np.ndarray, np.ndarray]:
    """Standard preprocessing: encode labels, handle missing, scale."""
    # Handle pandas DataFrames
    if hasattr(X, 'values'):
        # Fill missing values
        X = X.fillna(X.median(numeric_only=True))
        mode_df = X.mode()
        X = X.fillna(mode_df.iloc[0] if not mode_df.empty else 0)
        # Encode categoricals
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X = X.values

    X = np.asarray(X, dtype=float)

    # Handle NaN in numpy array
    if np.any(np.isnan(X)):
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])

    # Encode labels
    y = LabelEncoder().fit_transform(np.asarray(y).ravel())

    # Scale features
    X = StandardScaler().fit_transform(X)

    return X, y


# =============================================================================
# DATASET LOADERS
# =============================================================================

def load_breast_cancer_data() -> tuple[np.ndarray, np.ndarray, DatasetInfo]:
    """Wisconsin Breast Cancer - Clean medical imaging features."""
    data = load_breast_cancer()
    X, y = _preprocess(data.data, data.target)
    info = DatasetInfo(
        name="breast_cancer",
        description="Breast cancer diagnosis from cell nuclei features",
        n_samples=len(y), n_features=X.shape[1], n_classes=2,
        noise_level=NoiseLevel.CLEAN,
        source="sklearn"
    )
    return X, y, info


def load_wine_data() -> tuple[np.ndarray, np.ndarray, DatasetInfo]:
    """Wine recognition - Clean chemical analysis."""
    data = load_wine()
    X, y = _preprocess(data.data, data.target)
    info = DatasetInfo(
        name="wine",
        description="Wine cultivar classification from chemical analysis",
        n_samples=len(y), n_features=X.shape[1], n_classes=3,
        noise_level=NoiseLevel.CLEAN,
        source="sklearn"
    )
    return X, y, info


def load_digits_data() -> tuple[np.ndarray, np.ndarray, DatasetInfo]:
    """Handwritten digits - Some annotation errors expected."""
    data = load_digits()
    X, y = _preprocess(data.data, data.target)
    info = DatasetInfo(
        name="digits",
        description="Handwritten digit recognition (8x8 images)",
        n_samples=len(y), n_features=X.shape[1], n_classes=10,
        noise_level=NoiseLevel.LOW,
        source="sklearn"
    )
    return X, y, info


def load_iris_data() -> tuple[np.ndarray, np.ndarray, DatasetInfo]:
    """Iris - Classic clean dataset."""
    data = load_iris()
    X, y = _preprocess(data.data, data.target)
    info = DatasetInfo(
        name="iris",
        description="Iris flower classification",
        n_samples=len(y), n_features=X.shape[1], n_classes=3,
        noise_level=NoiseLevel.CLEAN,
        source="sklearn"
    )
    return X, y, info


def load_credit_approval() -> tuple[np.ndarray, np.ndarray, DatasetInfo]:
    """Credit Approval - Human annotation noise."""
    try:
        data = fetch_openml(name="credit-approval", version=1, as_frame=True, parser='auto')
        X, y = _preprocess(data.data, data.target)
        info = DatasetInfo(
            name="credit_approval",
            description="Credit card approval (human annotated)",
            n_samples=len(y), n_features=X.shape[1], n_classes=2,
            noise_level=NoiseLevel.MODERATE,
            source="openml"
        )
        return X, y, info
    except Exception:
        return None, None, None


def load_diabetes() -> tuple[np.ndarray, np.ndarray, DatasetInfo]:
    """Pima Diabetes - Known measurement noise."""
    try:
        data = fetch_openml(data_id=37, as_frame=True, parser='auto')
        X, y = _preprocess(data.data, data.target)
        info = DatasetInfo(
            name="diabetes",
            description="Pima Indians diabetes (measurement noise)",
            n_samples=len(y), n_features=X.shape[1], n_classes=2,
            noise_level=NoiseLevel.MODERATE,
            source="openml"
        )
        return X, y, info
    except Exception:
        return None, None, None


def load_heart_disease() -> tuple[np.ndarray, np.ndarray, DatasetInfo]:
    """Heart Disease - Subjective diagnosis criteria."""
    try:
        data = fetch_openml(name="heart-statlog", version=1, as_frame=True, parser='auto')
        X, y = _preprocess(data.data, data.target)
        info = DatasetInfo(
            name="heart_disease",
            description="Heart disease diagnosis (subjective criteria)",
            n_samples=len(y), n_features=X.shape[1], n_classes=2,
            noise_level=NoiseLevel.MODERATE,
            source="openml"
        )
        return X, y, info
    except Exception:
        return None, None, None


def load_ionosphere() -> tuple[np.ndarray, np.ndarray, DatasetInfo]:
    """Ionosphere - Radar signal classification."""
    try:
        data = fetch_openml(name="ionosphere", version=1, as_frame=True, parser='auto')
        X, y = _preprocess(data.data, data.target)
        info = DatasetInfo(
            name="ionosphere",
            description="Radar signal classification",
            n_samples=len(y), n_features=X.shape[1], n_classes=2,
            noise_level=NoiseLevel.LOW,
            source="openml"
        )
        return X, y, info
    except Exception:
        return None, None, None


def load_sonar() -> tuple[np.ndarray, np.ndarray, DatasetInfo]:
    """Sonar - Small noisy dataset."""
    try:
        data = fetch_openml(name="sonar", version=1, as_frame=True, parser='auto')
        X, y = _preprocess(data.data, data.target)
        info = DatasetInfo(
            name="sonar",
            description="Sonar signal classification (rocks vs mines)",
            n_samples=len(y), n_features=X.shape[1], n_classes=2,
            noise_level=NoiseLevel.MODERATE,
            source="openml"
        )
        return X, y, info
    except Exception:
        return None, None, None


def load_vehicle() -> tuple[np.ndarray, np.ndarray, DatasetInfo]:
    """Vehicle silhouettes - 4-class classification."""
    try:
        data = fetch_openml(name="vehicle", version=1, as_frame=True, parser='auto')
        X, y = _preprocess(data.data, data.target)
        info = DatasetInfo(
            name="vehicle",
            description="Vehicle silhouette classification",
            n_samples=len(y), n_features=X.shape[1], n_classes=4,
            noise_level=NoiseLevel.LOW,
            source="openml"
        )
        return X, y, info
    except Exception:
        return None, None, None


def load_segment() -> tuple[np.ndarray, np.ndarray, DatasetInfo]:
    """Image segmentation - 7-class."""
    try:
        data = fetch_openml(name="segment", version=1, as_frame=True, parser='auto')
        X, y = _preprocess(data.data, data.target)
        info = DatasetInfo(
            name="segment",
            description="Image segmentation (7 classes)",
            n_samples=len(y), n_features=X.shape[1], n_classes=7,
            noise_level=NoiseLevel.CLEAN,
            source="openml"
        )
        return X, y, info
    except Exception:
        return None, None, None


def load_australian() -> tuple[np.ndarray, np.ndarray, DatasetInfo]:
    """Australian Credit - Financial data with noise."""
    try:
        data = fetch_openml(name="australian", version=1, as_frame=True, parser='auto')
        X, y = _preprocess(data.data, data.target)
        info = DatasetInfo(
            name="australian",
            description="Australian credit approval",
            n_samples=len(y), n_features=X.shape[1], n_classes=2,
            noise_level=NoiseLevel.MODERATE,
            source="openml"
        )
        return X, y, info
    except Exception:
        return None, None, None


def load_german_credit() -> tuple[np.ndarray, np.ndarray, DatasetInfo]:
    """German Credit - Known label noise issues."""
    try:
        data = fetch_openml(name="credit-g", version=1, as_frame=True, parser='auto')
        X, y = _preprocess(data.data, data.target)
        info = DatasetInfo(
            name="german_credit",
            description="German credit risk (known label issues)",
            n_samples=len(y), n_features=X.shape[1], n_classes=2,
            noise_level=NoiseLevel.HIGH,
            source="openml"
        )
        return X, y, info
    except Exception:
        return None, None, None


def load_banknote() -> tuple[np.ndarray, np.ndarray, DatasetInfo]:
    """Banknote authentication - Clean dataset."""
    try:
        data = fetch_openml(name="banknote-authentication", version=1, as_frame=True, parser='auto')
        X, y = _preprocess(data.data, data.target)
        info = DatasetInfo(
            name="banknote",
            description="Banknote authentication from image features",
            n_samples=len(y), n_features=X.shape[1], n_classes=2,
            noise_level=NoiseLevel.CLEAN,
            source="openml"
        )
        return X, y, info
    except Exception:
        return None, None, None


def load_spambase() -> tuple[np.ndarray, np.ndarray, DatasetInfo]:
    """Spambase - Email spam detection."""
    try:
        data = fetch_openml(name="spambase", version=1, as_frame=True, parser='auto')
        X, y = _preprocess(data.data, data.target)
        info = DatasetInfo(
            name="spambase",
            description="Email spam classification",
            n_samples=len(y), n_features=X.shape[1], n_classes=2,
            noise_level=NoiseLevel.LOW,
            source="openml"
        )
        return X, y, info
    except Exception:
        return None, None, None


def load_phoneme() -> tuple[np.ndarray, np.ndarray, DatasetInfo]:
    """Phoneme - Speech recognition with noise."""
    try:
        data = fetch_openml(name="phoneme", version=1, as_frame=True, parser='auto')
        X, y = _preprocess(data.data, data.target)
        info = DatasetInfo(
            name="phoneme",
            description="Phoneme classification (nasal vs oral)",
            n_samples=len(y), n_features=X.shape[1], n_classes=2,
            noise_level=NoiseLevel.MODERATE,
            source="openml"
        )
        return X, y, info
    except Exception:
        return None, None, None


def load_magic() -> tuple[np.ndarray, np.ndarray, DatasetInfo]:
    """MAGIC Gamma Telescope - Large dataset."""
    try:
        data = fetch_openml(name="MagicTelescope", version=1, as_frame=True, parser='auto')
        X, y = _preprocess(data.data, data.target)
        info = DatasetInfo(
            name="magic",
            description="MAGIC gamma telescope (gamma vs hadron)",
            n_samples=len(y), n_features=X.shape[1], n_classes=2,
            noise_level=NoiseLevel.LOW,
            source="openml"
        )
        return X, y, info
    except Exception:
        return None, None, None


def load_electricity() -> tuple[np.ndarray, np.ndarray, DatasetInfo]:
    """Electricity - Time series classification."""
    try:
        data = fetch_openml(name="electricity", version=1, as_frame=True, parser='auto')
        X, y = _preprocess(data.data, data.target)
        info = DatasetInfo(
            name="electricity",
            description="Electricity price prediction (up/down)",
            n_samples=len(y), n_features=X.shape[1], n_classes=2,
            noise_level=NoiseLevel.MODERATE,
            source="openml"
        )
        return X, y, info
    except Exception:
        return None, None, None


# =============================================================================
# DATASET REGISTRY
# =============================================================================

DATASET_LOADERS = {
    # sklearn datasets (always available)
    "breast_cancer": load_breast_cancer_data,
    "wine": load_wine_data,
    "digits": load_digits_data,
    "iris": load_iris_data,
    # OpenML datasets
    "credit_approval": load_credit_approval,
    "diabetes": load_diabetes,
    "heart_disease": load_heart_disease,
    "ionosphere": load_ionosphere,
    "sonar": load_sonar,
    "vehicle": load_vehicle,
    "segment": load_segment,
    "australian": load_australian,
    "german_credit": load_german_credit,
    "banknote": load_banknote,
    "spambase": load_spambase,
    "phoneme": load_phoneme,
    "magic": load_magic,
    "electricity": load_electricity,
}


def load_dataset(name: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[DatasetInfo]]:
    """Load dataset by name."""
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_LOADERS.keys())}")
    return DATASET_LOADERS[name]()


def load_all_datasets(verbose: bool = True):
    """Load all available datasets."""
    datasets = []
    for name, loader in DATASET_LOADERS.items():
        try:
            X, y, info = loader()
            if X is not None:
                datasets.append((X, y, info))
                if verbose:
                    print(f"  ✓ {name}: {info.n_samples} samples, {info.n_features} features")
        except Exception as e:
            if verbose:
                print(f"  ✗ {name}: {e}")
    return datasets


def get_datasets_by_noise(noise_level: NoiseLevel, verbose: bool = False):
    """Get datasets with specific noise level."""
    datasets = []
    for name, loader in DATASET_LOADERS.items():
        try:
            X, y, info = loader()
            if X is not None and info.noise_level == noise_level:
                datasets.append((X, y, info))
        except Exception:
            pass
    return datasets

