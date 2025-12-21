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

## Benchmark Results

On 17 datasets with known label noise (moderate/high):

| Dataset | TrustCurve | XGBoost | Diff |
|---------|------------|---------|------|
| sonar | 0.897 | 0.857 | **+4.0%** |
| diabetes | 0.751 | 0.734 | **+1.7%** |
| madelon | 0.812 | 0.801 | **+1.0%** |
| credit_approval | 0.872 | 0.862 | **+1.0%** |
| adult | 0.872 | 0.868 | +0.4% |
| satimage | 0.925 | 0.923 | +0.2% |
| ozone | 0.944 | 0.943 | +0.1% |
| steel_plates | 1.000 | 1.000 | 0.0% |
| mushroom | 1.000 | 1.000 | 0.0% |
| phoneme | 0.890 | 0.890 | 0.0% |
| kr_vs_kp | 0.995 | 0.995 | 0.0% |
| splice | 0.962 | 0.963 | -0.1% |
| german_credit | 0.750 | 0.752 | -0.2% |
| bank_marketing | 0.902 | 0.905 | -0.3% |
| heart_disease | 0.765 | 0.790 | -2.5% |
| letter | 0.933 | 0.963 | -3.0% |
| electricity | 0.888 | 0.926 | -3.8% |

TrustCurve helps most on smaller datasets with label noise. On large multiclass datasets (letter) or datasets with concept drift (electricity), the screening overhead can hurt.

## Running benchmarks

```bash
python benchmarks/run_benchmark.py --mode noisy      # datasets with known noise
python benchmarks/run_benchmark.py --mode real       # all real-world datasets
python benchmarks/run_benchmark.py --mode injected   # clean data + synthetic noise
python benchmarks/run_benchmark.py --mode synthetic  # pure synthetic data
```

## Limitations

Assumes that confident predictions after warmup are correct. This can fail for underrepresented subgroups, insufficient warmup, or limited model capacity—where the model may be confidently wrong rather than the label being wrong.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
