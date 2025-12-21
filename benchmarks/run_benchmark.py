#!/usr/bin/env python3
"""
TrustCurve Benchmark Suite

Usage:
    python benchmarks/run_benchmark.py              # all benchmarks
    python benchmarks/run_benchmark.py --mode real  # real-world only
    python benchmarks/run_benchmark.py --mode synthetic
"""

import argparse
import warnings

import numpy as np
from datasets import DATASET_LOADERS, NoiseLevel, load_dataset
from evaluate import (
    BenchmarkResults,
    EvalResult,
    evaluate_dataset,
    evaluate_single,
    generate_synthetic,
    print_results,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


def benchmark_real_world(n_trials=5):
    """Benchmark on real-world datasets."""
    print("\n" + "=" * 60)
    print("REAL-WORLD DATASETS")
    print("=" * 60)

    results = BenchmarkResults()

    for name in DATASET_LOADERS:
        try:
            X, y, info = load_dataset(name)
            if X is None:
                continue

            print(f"\n{name}: {info.n_samples} samples, {info.n_classes} classes")

            r = evaluate_dataset(X, y, name, noise_levels=[0.0], n_trials=n_trials)
            for res in r.results:
                results.add(res)

            tc = np.mean([x.accuracy for x in r.results if x.method == "trustcurve"])
            bl = np.mean([x.accuracy for x in r.results if x.method == "baseline"])
            print(f"  TC: {tc:.4f}, Base: {bl:.4f}, Diff: {tc-bl:+.4f}")

        except Exception as e:
            print(f"  ✗ {name}: {e}")

    print_results(results, "REAL-WORLD SUMMARY")
    return results


def benchmark_with_noise(noise_levels=[0.0, 0.1, 0.2, 0.3], n_trials=5):
    """Benchmark with injected noise."""
    print("\n" + "=" * 60)
    print("DATASETS + SYNTHETIC NOISE")
    print("=" * 60)

    datasets = ["breast_cancer", "wine", "digits", "ionosphere", "vehicle"]
    results = BenchmarkResults()

    for name in datasets:
        try:
            X, y, _ = load_dataset(name)
            if X is None:
                continue

            print(f"\n{name}:")
            r = evaluate_dataset(X, y, name, noise_levels=noise_levels, n_trials=n_trials)

            for res in r.results:
                results.add(res)

            for noise in noise_levels:
                tc = np.mean([x.accuracy for x in r.results if x.method == "trustcurve" and x.noise_level == noise])
                bl = np.mean([x.accuracy for x in r.results if x.method == "baseline" and x.noise_level == noise])
                print(f"  {noise:>3.0%}: TC={tc:.4f}, Base={bl:.4f}, Diff={tc-bl:+.4f}")

        except Exception as e:
            print(f"  ✗ {name}: {e}")

    print_results(results, "SYNTHETIC NOISE SUMMARY")
    return results


def benchmark_noisy_datasets(n_trials=5):
    """Benchmark on datasets with known label noise (MODERATE or HIGH)."""
    print("\n" + "=" * 60)
    print("DATASETS WITH KNOWN LABEL NOISE")
    print("=" * 60)

    results = BenchmarkResults()

    for name in DATASET_LOADERS:
        try:
            X, y, info = load_dataset(name)
            if X is None:
                continue
            if info.noise_level not in (NoiseLevel.MODERATE, NoiseLevel.HIGH):
                continue

            print(f"\n{name} [{info.noise_level.value}]: {info.n_samples} samples, {info.n_classes} classes")

            r = evaluate_dataset(X, y, name, noise_levels=[0.0], n_trials=n_trials)
            for res in r.results:
                results.add(res)

            tc = np.mean([x.accuracy for x in r.results if x.method == "trustcurve"])
            bl = np.mean([x.accuracy for x in r.results if x.method == "baseline"])
            print(f"  TC: {tc:.4f}, Base: {bl:.4f}, Diff: {tc-bl:+.4f}")

        except Exception as e:
            print(f"  ✗ {name}: {e}")

    print_results(results, "NOISY DATASETS SUMMARY")
    return results


def benchmark_synthetic(noise_levels=[0.0, 0.05, 0.1, 0.2, 0.3], n_trials=5):
    """Benchmark on synthetic data."""
    print("\n" + "=" * 60)
    print("PURE SYNTHETIC DATA")
    print("=" * 60)

    configs = [
        {"name": "easy_binary", "n_samples": 2000, "n_features": 20, "n_informative": 15, "n_classes": 2},
        {"name": "hard_binary", "n_samples": 2000, "n_features": 50, "n_informative": 10, "n_classes": 2},
        {"name": "multiclass", "n_samples": 2000, "n_features": 20, "n_informative": 15, "n_classes": 5},
    ]

    results = BenchmarkResults()

    for cfg in configs:
        cfg = cfg.copy()
        name = cfg.pop("name")
        print(f"\n{name}:")

        for noise in noise_levels:
            tc_acc, bl_acc = [], []

            for trial in range(n_trials):
                X, y_clean, y_noisy = generate_synthetic(**cfg, noise_rate=noise, seed=42 + trial)
                X_tr, X_te, y_tr, y_te = train_test_split(X, y_clean, test_size=0.2, random_state=42 + trial)
                _, _, y_tr_noisy, _ = train_test_split(X, y_noisy, test_size=0.2, random_state=42 + trial)

                tc = evaluate_single(X_tr, X_te, y_tr_noisy, y_te, "trustcurve")
                bl = evaluate_single(X_tr, X_te, y_tr_noisy, y_te, "baseline")

                tc_acc.append(tc["accuracy"])
                bl_acc.append(bl["accuracy"])

                results.add(EvalResult(name, noise, "trustcurve", tc["accuracy"], tc["f1"], tc["auc"], tc["train_time"]))
                results.add(EvalResult(name, noise, "baseline", bl["accuracy"], bl["f1"], bl["auc"], bl["train_time"]))

            print(f"  {noise:>3.0%}: TC={np.mean(tc_acc):.4f}, Base={np.mean(bl_acc):.4f}, Diff={np.mean(tc_acc)-np.mean(bl_acc):+.4f}")

    print_results(results, "SYNTHETIC DATA SUMMARY")
    return results


def main():
    parser = argparse.ArgumentParser(description="TrustCurve Benchmark")
    parser.add_argument("--mode", choices=["all", "real", "noisy", "injected", "synthetic"], default="all")
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    if args.mode == "all":
        benchmark_real_world(args.trials)
        benchmark_with_noise(n_trials=args.trials)
        benchmark_synthetic(n_trials=args.trials)
    elif args.mode == "real":
        benchmark_real_world(args.trials)
    elif args.mode == "noisy":
        benchmark_noisy_datasets(n_trials=args.trials)
    elif args.mode == "injected":
        benchmark_with_noise(n_trials=args.trials)
    else:
        benchmark_synthetic(n_trials=args.trials)


if __name__ == "__main__":
    main()
