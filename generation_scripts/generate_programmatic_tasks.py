"""
generate_programmatic_tasks.py — Generate parameterized benchmark tasks via combinatorial sweep.

Parameter space:
  - company_size: seed / series_a / series_b
  - signal_confidence: low / medium / high
  - bench_state: full / partial / empty_for_required
  - prior_thread: cold / warm / hostile
  - dimension: signal_grounding / tone_compliance / cta_quality / personalization / bench_fit_alignment

Each combination produces one task. After generation, tasks are passed to judge_filter.py for quality scoring.
Target output: 60 tasks for the programmatic partition.

Usage:
    python generate_programmatic_tasks.py \
        --output ../tenacious_bench_v0.1/train/tasks.jsonl \
        --count 60 \
        --seed 42
"""

import argparse
import itertools
import json
import random
from pathlib import Path
from typing import Any

SEED = 42

# ---------------------------------------------------------------------------
# Parameter definitions
# ---------------------------------------------------------------------------

COMPANY_SIZES = ["seed", "series_a", "series_b"]
SIGNAL_CONFIDENCES = ["low", "medium", "high"]
BENCH_STATES = ["full", "partial", "empty_for_required"]
PRIOR_THREADS = ["cold", "warm", "hostile"]
DIMENSIONS = ["signal_grounding", "tone_compliance", "cta_quality", "personalization", "bench_fit_alignment"]

COMPANY_NAMES = [
    "Meridian Software", "Forge Analytics", "Vantage Health Tech", "Apex Data",
    "Stratus Cloud", "Onyx Systems", "Pinnacle AI", "Cascade Tech",
    "Horizon Labs", "Vector Solutions", "Nexus Engineering", "Prism Analytics",
    "Zenith Software", "Atlas Computing", "Cobalt Systems", "Ember Analytics",
    "Ironclad Data", "Jade Systems", "Keystone AI", "Luminex Tech",
    "Magma Software", "Nova Computing", "Orbit Labs", "Quantum Data",
    "Radian Tech", "Solar Analytics", "Timber Systems", "Uniforce AI",
]

# ---------------------------------------------------------------------------
# Template builders
# ---------------------------------------------------------------------------

COMPANY_SIZE_SPECS = {
    "seed": {"round": "Seed", "amount_range": (1, 5), "employee_range": (5, 30)},
    "series_a": {"round": "Series A", "amount_range": (5, 20), "employee_range": (30, 150)},
    "series_b": {"round": "Series B", "amount_range": (15, 40), "employee_range": (100, 500)},
}

CONFIDENCE_SPECS = {
    "low": {"ai_conf_range": (0.50, 0.68), "label": "low"},
    "medium": {"ai_conf_range": (0.69, 0.79), "label": "medium"},
    "high": {"ai_conf_range": (0.80, 0.95), "label": "high"},
}

BENCH_STATE_SPECS = {
    "full": {"python": 6, "devops": 3, "go": 2, "rust": 2},
    "partial": {"python": 3, "devops": 1, "go": 1, "rust": 0},
    "empty_for_required": {"python": 2, "devops": 1, "go": 0, "rust": 0},
}

PRIOR_THREAD_TEMPLATES = {
    "cold": "",
    "warm": "Prospect reply: 'Sounds interesting — tell me more about your Python team depth.'",
    "hostile": "Prospect reply: 'We build everything in-house. We don't believe in outsourcing.'",
}

DIMENSION_BANNED_EXTRAS = {
    "bench_fit_alignment": ["we have 3 rust", "can start monday", "immediately available", "start within 2 weeks"],
    "tone_compliance": ["you're falling behind", "your competitors are miles ahead", "i apologize", "i understand your concern"],
    "signal_grounding": ["scale aggressively", "aggressive hiring", "massive growth"],
    "cta_quality": ["i've gone ahead and booked", "your calendar invite"],
    "personalization": [],
}

DIMENSION_REQUIRED_HEDGES = {
    "signal_grounding": {
        "low": ["are you finding", "it looks like", "appears", "from what we can see"],
        "medium": ["from what we can see", "based on public signals"],
        "high": [],
    },
    "bench_fit_alignment": {"low": [], "medium": [], "high": []},
    "tone_compliance": {"low": [], "medium": [], "high": []},
    "cta_quality": {"low": [], "medium": [], "high": []},
    "personalization": {"low": [], "medium": [], "high": []},
}


def build_company_signal(rng: random.Random, size: str, confidence: str, dimension: str) -> str:
    spec = COMPANY_SIZE_SPECS[size]
    conf_spec = CONFIDENCE_SPECS[confidence]

    name = rng.choice(COMPANY_NAMES)
    amount = rng.randint(*spec["amount_range"])
    employees = rng.randint(*spec["employee_range"])
    days = rng.randint(20, 150)
    ai_score = rng.randint(0, 3)
    ai_conf = round(rng.uniform(*conf_spec["ai_conf_range"]), 2)
    ai_conf_label = conf_spec["label"]

    # role count: low for signal_grounding tests to ensure over-claiming risk
    if dimension == "signal_grounding":
        roles = rng.randint(1, 4) if confidence == "low" else rng.randint(3, 8)
    else:
        roles = rng.randint(2, 10)

    role_options = [
        f"{max(1,roles//2)} Python, {max(1,roles-roles//2)} DevOps",
        f"{max(1,roles//2)} Backend, 1 ML Engineer",
        f"{max(1,roles//3)} Python, {max(1,roles//3)} React, 1 Go",
    ]
    role_detail = rng.choice(role_options)

    layoff_options = ["none", f"{rng.randint(5,20)}% headcount cut ({rng.randint(10,60)} days ago)"]
    layoff = layoff_options[0] if rng.random() > 0.3 else layoff_options[1]

    return (
        f"Company: {name} ({spec['round']}, ${amount}M, closed {days} days ago). "
        f"Open engineering roles: {roles} ({role_detail}). Employee count: {employees}. "
        f"AI maturity score: {ai_score} (confidence: {ai_conf}, {ai_conf_label}). "
        f"Signal age: {days} days. Layoff signal: {layoff}."
    )


def build_bench_summary(bench_state: str, avail_weeks: int) -> str:
    bs = BENCH_STATE_SPECS[bench_state]
    python_detail = "2 senior, 1 mid" if bs["python"] <= 3 else "3 senior, 3 mid"
    return (
        f"Available bench: {bs['python']} Python engineers ({python_detail}), "
        f"{bs['devops']} DevOps engineers, {bs['go']} Go engineer(s), "
        f"{bs['rust']} Rust engineers. Next availability window: {avail_weeks} weeks."
    )


def build_expected_features(dimension: str, confidence: str) -> dict[str, Any]:
    base_banned = ["offshore", "outsourcing", "cost reduction", "aggressive hiring", "massive growth"]
    extra_banned = DIMENSION_BANNED_EXTRAS.get(dimension, [])
    hedges = DIMENSION_REQUIRED_HEDGES.get(dimension, {}).get(confidence, [])

    features: dict[str, Any] = {
        "must_reference_signal": dimension in ("signal_grounding", "personalization"),
        "must_include_cta": True,
        "banned_phrases": base_banned + extra_banned,
    }
    if hedges:
        features["required_hedges"] = hedges
    if dimension == "bench_fit_alignment":
        features["max_headcount_commitment"] = None
        features["must_reference_signal"] = False
    return features


def assign_difficulty(size: str, confidence: str, bench_state: str, thread: str) -> str:
    hard_flags = sum([
        confidence == "low",
        bench_state == "empty_for_required",
        thread == "hostile",
    ])
    if hard_flags >= 2:
        return "hard"
    if hard_flags == 1:
        return "medium"
    return "easy"


def build_task(
    idx: int,
    partition: str,
    size: str,
    confidence: str,
    bench_state: str,
    thread: str,
    dimension: str,
    rng: random.Random,
) -> dict[str, Any]:
    part_prefix = {"train": "train", "dev": "dev", "held_out": "held"}[partition]
    task_id = f"tb_{part_prefix}_{idx:04d}"
    difficulty = assign_difficulty(size, confidence, bench_state, thread)
    avail_weeks = rng.randint(1, 4)

    return {
        "task_id": task_id,
        "dimension": dimension,
        "difficulty": difficulty,
        "source_mode": "programmatic",
        "input": {
            "company_signal": build_company_signal(rng, size, confidence, dimension),
            "bench_summary": build_bench_summary(bench_state, avail_weeks),
            "prior_thread": PRIOR_THREAD_TEMPLATES[thread],
        },
        "expected_features": build_expected_features(dimension, confidence),
        "rubric": {
            "signal_grounding": 0,
            "tone_alignment": 0,
            "cta_quality": 0,
            "bench_fit_accuracy": 0,
            "personalization_depth": 0,
        },
        "gold_output": None,
        "notes": f"Programmatic: size={size}, confidence={confidence}, bench={bench_state}, thread={thread}",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate programmatic benchmark tasks")
    parser.add_argument("--output", default="../tenacious_bench_v0.1/train/tasks.jsonl")
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--partition", default="train", choices=["train", "dev", "held_out"])
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Build full parameter grid
    grid = list(itertools.product(COMPANY_SIZES, SIGNAL_CONFIDENCES, BENCH_STATES, PRIOR_THREADS, DIMENSIONS))
    rng.shuffle(grid)

    # Sample to target count, ensuring dimension coverage
    sampled = []
    dim_counts = {d: 0 for d in DIMENSIONS}
    target_per_dim = args.count // len(DIMENSIONS)

    for params in grid:
        size, confidence, bench_state, thread, dimension = params
        if dim_counts[dimension] < target_per_dim + 2:
            sampled.append(params)
            dim_counts[dimension] += 1
        if len(sampled) >= args.count:
            break

    # Fill remaining if needed
    for params in grid:
        if len(sampled) >= args.count:
            break
        if params not in sampled:
            sampled.append(params)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = []
    for i, (size, confidence, bench_state, thread, dimension) in enumerate(sampled[: args.count]):
        task = build_task(i + 1, args.partition, size, confidence, bench_state, thread, dimension, rng)
        tasks.append(task)

    with open(output_path, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")

    print(f"Generated {len(tasks)} programmatic tasks → {output_path}")
    print(f"Dimension distribution: { {d: sum(1 for t in tasks if t['dimension']==d) for d in DIMENSIONS} }")
    print(f"Difficulty distribution: { {d: sum(1 for t in tasks if t['difficulty']==d) for d in ['easy','medium','hard']} }")


if __name__ == "__main__":
    main()
