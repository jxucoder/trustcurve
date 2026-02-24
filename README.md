# TrustCurve

Model-agnostic interpretability plots. Polars-first, works with any predict function.

## Features

- **PDP** — Partial Dependence Plots
- **ICE** — Individual Conditional Expectation (+ centered c-ICE)
- **ALE** — Accumulated Local Effects (handles correlated features)
- **2-D Interaction** — Heatmaps for feature pairs

Works with any `f(X) -> y` callable: sklearn, XGBoost, LightGBM, PyTorch, or your own function.

## Install

```bash
pip install trustcurve
```

## Quick start

```python
import polars as pl
import trustcurve as tc

# Any predict function that takes a numpy array
result = tc.pdp(model.predict, X, "age")
result.plot()

# ICE with centering
result = tc.ice(model.predict, X, "income", centered=True)
result.plot()

# ALE (better than PDP when features are correlated)
result = tc.ale(model.predict, X, "age")
result.plot()

# 2-D interaction
result = tc.interaction(model.predict, X, ["age", "income"])
result.plot()
```

## Handles large data

Pass a Polars `DataFrame` or `LazyFrame`. Large datasets are automatically
sampled (configurable via `max_samples`). Predictions can be batched via
`batch_size` to bound memory.

```python
big_data = pl.scan_parquet("huge_dataset.parquet")
result = tc.pdp(model.predict, big_data, "age", max_samples=5000, batch_size=10000)
```

## API

| Function | What it computes |
|----------|-----------------|
| `tc.pdp(predict_fn, X, feature)` | Partial dependence (+ ICE lines) |
| `tc.ice(predict_fn, X, feature, centered=True)` | Individual conditional expectation |
| `tc.ale(predict_fn, X, feature)` | Accumulated local effects |
| `tc.interaction(predict_fn, X, [f1, f2])` | 2-D partial dependence heatmap |

All functions return result objects with a `.plot()` method.

## Dependencies

- polars >= 1.0
- numpy >= 1.24
- matplotlib >= 3.7
