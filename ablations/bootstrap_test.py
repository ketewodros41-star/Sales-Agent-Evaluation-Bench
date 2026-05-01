"""
Paired bootstrap significance test for ablation results.

Tests whether the score lift in Delta A or Delta B is statistically significant.
Uses paired bootstrap resampling (1000 iterations by default).

Usage:
    python ablations/bootstrap_test.py \
        --results ablations/ablation_results.json \
        --n-bootstrap 1000 \
        --output ablations/significance.json
"""

import argparse
import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def bootstrap_mean_lift(lifts: list[float], n_bootstrap: int, seed: int) -> dict:
    """
    Paired bootstrap test for mean lift > 0.

    Returns:
        mean_lift, ci_lower, ci_upper (95%), p_value (one-tailed: lift > 0)
    """
    random.seed(seed)
    n = len(lifts)
    observed_mean = sum(lifts) / n

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = [random.choice(lifts) for _ in range(n)]
        bootstrap_means.append(sum(sample) / n)

    bootstrap_means.sort()
    ci_lower = bootstrap_means[int(0.025 * n_bootstrap)]
    ci_upper = bootstrap_means[int(0.975 * n_bootstrap)]

    # One-tailed p-value: fraction of bootstrap samples with mean ≤ 0
    p_value = sum(1 for m in bootstrap_means if m <= 0) / n_bootstrap

    return {
        "mean_lift": round(observed_mean, 4),
        "ci_95_lower": round(ci_lower, 4),
        "ci_95_upper": round(ci_upper, 4),
        "p_value_one_tailed": round(p_value, 4),
        "significant_at_0.05": p_value < 0.05,
        "n_bootstrap": n_bootstrap,
        "n_tasks": n,
    }


def main():
    parser = argparse.ArgumentParser(description="Paired bootstrap significance test")
    parser.add_argument("--results", required=True, help="ablation_results.json")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--output", required=True, help="Output significance.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    results = data.get("results", [])
    significance = {}

    for delta_label in ["delta_a", "delta_b"]:
        delta_results = [r for r in results if r.get("delta") == delta_label]
        if not delta_results:
            log.info("No results found for %s — skipping", delta_label)
            continue

        lifts = [r["score_lift"] for r in delta_results]
        log.info(
            "%s: n=%d mean_lift=%.4f",
            delta_label, len(lifts), sum(lifts) / len(lifts)
        )

        stats = bootstrap_mean_lift(lifts, args.n_bootstrap, args.seed)
        significance[delta_label] = stats

        log.info(
            "%s significance: mean=%.4f 95CI=[%.4f, %.4f] p=%.4f significant=%s",
            delta_label,
            stats["mean_lift"],
            stats["ci_95_lower"],
            stats["ci_95_upper"],
            stats["p_value_one_tailed"],
            stats["significant_at_0.05"],
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(significance, f, indent=2)

    log.info("Significance results written to %s", output_path)

    # Print interpretation
    for label, stats in significance.items():
        if stats["significant_at_0.05"]:
            print(
                f"\n{label.upper()}: SIGNIFICANT (p={stats['p_value_one_tailed']:.4f})\n"
                f"  Mean lift: {stats['mean_lift']:.4f}\n"
                f"  95% CI: [{stats['ci_95_lower']:.4f}, {stats['ci_95_upper']:.4f}]"
            )
        else:
            print(
                f"\n{label.upper()}: NOT SIGNIFICANT (p={stats['p_value_one_tailed']:.4f})\n"
                f"  Mean lift: {stats['mean_lift']:.4f}\n"
                f"  95% CI: [{stats['ci_95_lower']:.4f}, {stats['ci_95_upper']:.4f}]\n"
                f"  -> Report this honestly. A null result is still a publishable finding."
            )


if __name__ == "__main__":
    main()
