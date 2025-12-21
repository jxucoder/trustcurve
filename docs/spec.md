# TrustCurve Technical Specification

> **Note**: This is research in progress. The method and analysis are preliminary.

## Core Idea

Use the loss-to-Hessian ratio to identify mislabeled samples:

```
R = L / H = -log(p) / (p(1-p))
```

For log-loss, the Hessian `H = p(1-p)` measures prediction uncertainty:
- Maximum at p=0.5 (uncertain)
- Minimum at p=0 or p=1 (confident)

The ratio R answers: "Is high loss due to uncertainty or confident error?"

| Case | Loss | Hessian | Ratio | Interpretation |
|------|------|---------|-------|----------------|
| Easy (p≈1) | Low | Low | ~1 | Correct, confident |
| Hard (p≈0.5) | High | High | ~2 | Uncertain |
| Noise (p≈0) | High | Low | ~460 | Confidently wrong |

## Adaptive Screening

Fixed thresholds don't generalize across datasets with different noise levels.

Use Gini coefficient of the ratio distribution to estimate noise:
- Clean data: ratios are similar, low Gini, apply light screening
- Noisy data: some ratios are outliers, high Gini, apply aggressive screening

```python
noise_level = clip((gini(R) - 0.2) * 2.5, 0, 1)
screening_strength = min(noise_level * 5, 1)
```

## Weight Function

Soft thresholding with Lorentzian decay:

```python
w = 1 / (1 + (R / tolerance)^power)
```

Final weights blend with uniform based on screening strength:

```python
weights = (1 - strength) * 1.0 + strength * raw_weights
```

## Training Phases

1. **Warmup**: Train incrementally until at least 1% of samples have >87% confidence
2. **Screen**: Compute ratios and weights from warmup predictions
3. **Train**: Continue boosting with weighted samples

## Limitations and Future Work

- Currently only supports classification (binary and multiclass)
- Single screening pass; iterative refinement may improve results
- Tested primarily on tabular data
- Theoretical guarantees not yet established
