<p align="center">
  <img src="assets/trustcurve_logo.png" alt="TrustCurve" width="400">
</p>

# TrustCurve

> **Note**: This is research in progress. The API may change and results are preliminary.

Noise-robust gradient boosting using loss-to-curvature sample screening.

## Motivation

When training on noisy labels, high-loss samples fall into two categories:

| Type | p = P(true class) | Loss | Hessian | Ratio R |
|------|-------------------|------|---------|---------|
| Hard sample | ~0.5 (uncertain) | High | High | Low |
| Mislabeled | ~0 (confident, wrong) | High | Low | High |

Standard approaches treat all high-loss samples equally. TrustCurve uses the loss-to-Hessian ratio to distinguish them:

```
R = -log(p) / (p × (1-p))
```

Hard samples have high loss *and* high uncertainty (the model is unsure). Mislabeled samples have high loss but low uncertainty (the model is confidently wrong). The ratio separates these cases.

## Installation

```bash
git clone https://github.com/jxucoder/TrustCurve.git
cd TrustCurve
pip install -e .
```

## Usage

```python
from trustcurve import TrustCurveClassifier

model = TrustCurveClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Inspect detected noise
noisy_indices = model.get_noisy_samples(percentile=10)
ratios = model.get_curvature_ratio()
```

### Standalone functions

```python
from trustcurve import compute_curvature_ratio, curvature_weights

# Compute ratios from existing predictions
ratios = compute_curvature_ratio(y_train, y_proba)

# Get sample weights
weights, ratios, strength = curvature_weights(y_train, y_proba)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 300 | Total boosting rounds |
| `warmup_trees` | "auto" | Trees before screening starts |
| `tolerance` | 5.0 | Ratio threshold for weighting |
| `power` | 4.0 | Sharpness of weight decay |
| `adaptive` | True | Scale screening by detected noise level |

## How it works

1. **Warmup**: Train XGBoost until predictions stabilize
2. **Screen**: Compute R = Loss / Hessian for each sample
3. **Weight**: Apply soft threshold: `w = 1 / (1 + (R/tolerance)^power)`
4. **Continue**: Train remaining trees with weighted samples

The Gini coefficient of the ratio distribution estimates noise prevalence. When noise is low, screening is gentle. When noise is high, screening is aggressive.

## Running benchmarks

```bash
python benchmarks/run_benchmark.py              # all benchmarks
python benchmarks/run_benchmark.py --mode real  # real-world datasets only
python benchmarks/run_benchmark.py --mode synthetic
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
