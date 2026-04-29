"""
generate_trace_tasks.py — Generate trace-derived benchmark tasks.

Reads Week 10 trace_log.jsonl (failed trials with reward=0.0),
classifies each failure against the probe taxonomy, and outputs
benchmark tasks in Tenacious-Bench task schema format.

Usage:
    python generate_trace_tasks.py \
        --trace-log ../../trp_week10/trace_log.jsonl \
        --output ../tenacious_bench_v0.1/train/trace_tasks.jsonl \
        --max-tasks 60 \
        --seed 42
"""

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

SEED = 42

# ---------------------------------------------------------------------------
# Probe → dimension mapping (from Week 10 probe library)
# ---------------------------------------------------------------------------

PROBE_DIMENSION_MAP = {
    "P-005": "signal_grounding",
    "P-006": "signal_grounding",
    "P-007": "signal_grounding",
    "P-008": "signal_grounding",
    "P-012": "tone_compliance",
    "P-013": "tone_compliance",
    "P-014": "tone_compliance",
    "P-015": "tone_compliance",
    "P-009": "bench_fit_alignment",
    "P-010": "bench_fit_alignment",
    "P-011": "bench_fit_alignment",
    "P-021": "cta_quality",
    "P-022": "cta_quality",
    "P-023": "cta_quality",
    "P-027": "personalization",
    "P-028": "personalization",
    "P-029": "personalization",
    "P-033": "personalization",
}

# ---------------------------------------------------------------------------
# Synthetic signal templates (replace real prospect data)
# ---------------------------------------------------------------------------

COMPANY_SIGNAL_TEMPLATES = [
    "Company: {name} (Series {round}, ${amount}M, closed {days} days ago). Open engineering roles: {roles} ({role_detail}). Employee count: {employees}. AI maturity score: {ai_score} (confidence: {ai_conf}, {ai_conf_label}). Signal age: {days} days. Layoff signal: {layoff}.",
    "Company: {name} (Seed, ${amount}M, closed {days} days ago). Open engineering roles: {roles} ({role_detail}). Employee count: {employees}. AI maturity score: {ai_score} (confidence: {ai_conf}, {ai_conf_label}). Signal age: {days} days. Layoff signal: {layoff}.",
]

BENCH_SUMMARY_TEMPLATES = [
    "Available bench: {python_count} Python engineers ({python_detail}), {devops_count} DevOps engineers, {go_count} Go engineer(s), {rust_count} Rust engineers. Next availability window: {avail_weeks} weeks.",
    "Available bench: {python_count} Python engineers ({python_detail}), {react_count} React engineers, {go_count} Go engineer(s), {rust_count} Rust engineers. Next availability window: {avail_weeks} weeks.",
]

COMPANY_NAMES = [
    "Meridian Software", "Forge Analytics", "Vantage Health Tech", "Apex Data",
    "Stratus Cloud", "Onyx Systems", "Pinnacle AI", "Cascade Tech",
    "Horizon Labs", "Vector Solutions", "Nexus Engineering", "Prism Analytics",
    "Zenith Software", "Atlas Computing", "Cobalt Systems", "Ember Analytics",
    "Glitch-Free Tech", "Ironclad Data", "Jade Systems", "Keystone AI",
    "Luminex Tech", "Magma Software", "Nova Computing", "Orbit Labs",
]

# ---------------------------------------------------------------------------
# Trace-to-task conversion
# ---------------------------------------------------------------------------

def classify_failure(trace: dict[str, Any]) -> tuple[str, str, str]:
    """
    Returns (probe_id, dimension, difficulty) for a failed trace.
    Uses task_id ranges as a heuristic (from Week 10 probe mapping).
    """
    task_id = str(trace.get("task_id", "0"))
    agent_cost = trace.get("agent_cost", 0.0)
    duration = trace.get("duration", 0.0)

    # High cost + long duration → bench over-commitment or cost pathology
    if agent_cost > 0.035 and duration > 200:
        return "P-009", "bench_fit_alignment", "hard"
    # Medium cost, long duration → signal grounding failure
    if duration > 150:
        return "P-005", "signal_grounding", "medium"
    # Low cost → tone compliance or CTA failure
    if agent_cost < 0.015:
        return "P-013", "tone_compliance", "easy"
    # Default → personalization failure
    return "P-028", "personalization", "medium"


def build_company_signal(rng: random.Random, probe_id: str) -> str:
    """Generate a synthetic company signal appropriate for the given probe."""
    name = rng.choice(COMPANY_NAMES)
    round_type = rng.choice(["A", "B", "B"])
    amount = rng.choice([8, 12, 15, 18, 22])
    days = rng.randint(20, 120)
    roles = rng.randint(1, 10)

    role_detail_options = [
        f"{rng.randint(1,3)} Python, {rng.randint(1,2)} DevOps",
        f"{rng.randint(1,3)} Backend, {rng.randint(1,2)} ML Engineer",
        f"{rng.randint(1,4)} Python, {rng.randint(1,2)} React, 1 Go",
    ]
    role_detail = rng.choice(role_detail_options)

    # P-005: force low role count to trigger signal over-claiming scenario
    if probe_id == "P-005":
        roles = rng.randint(1, 4)

    employees = rng.randint(50, 500)
    ai_score = rng.randint(0, 3)
    ai_conf = round(rng.uniform(0.55, 0.90), 2)
    ai_conf_label = "low" if ai_conf < 0.70 else "medium" if ai_conf < 0.80 else "high"
    layoff = "none" if rng.random() > 0.3 else f"{rng.randint(5, 20)}% headcount cut ({rng.randint(10, 60)} days ago)"

    return (
        f"Company: {name} (Series {round_type}, ${amount}M, closed {days} days ago). "
        f"Open engineering roles: {roles} ({role_detail}). Employee count: {employees}. "
        f"AI maturity score: {ai_score} (confidence: {ai_conf}, {ai_conf_label}). "
        f"Signal age: {days} days. Layoff signal: {layoff}."
    )


def build_bench_summary(rng: random.Random, probe_id: str) -> str:
    """Generate a bench summary appropriate for the given probe."""
    python_count = rng.randint(3, 8)
    python_detail = f"{rng.randint(1,3)} senior, {rng.randint(1,3)} mid"
    devops_count = rng.randint(1, 3)
    go_count = rng.randint(0, 2)
    avail_weeks = rng.randint(1, 4)

    # P-009/P-010: force 0 Rust to trigger bench over-commitment scenario
    rust_count = 0 if probe_id in ("P-009", "P-010") else rng.randint(0, 2)

    return (
        f"Available bench: {python_count} Python engineers ({python_detail}), "
        f"{devops_count} DevOps engineers, {go_count} Go engineer(s), "
        f"{rust_count} Rust engineers. Next availability window: {avail_weeks} weeks."
    )


def build_prior_thread(rng: random.Random, probe_id: str) -> str:
    """Generate a prior thread context appropriate for the given probe."""
    if probe_id in ("P-009", "P-010", "P-011"):
        options = [
            "Prospect reply: 'We need 3 Rust engineers for a 6-month systems rewrite. Can Tenacious deliver that quickly?'",
            "Prospect reply: 'Can someone from your team start Monday? The project is urgent.'",
            "Prospect reply: 'How many Python engineers can you put on this immediately?'",
        ]
        return rng.choice(options)
    if probe_id in ("P-012", "P-014"):
        options = [
            "Multi-turn: prospect has asked about pricing 3 times.",
            "Prospect reply: 'We build everything in-house. We don't believe in outsourcing.'",
        ]
        return rng.choice(options)
    if probe_id == "P-021":
        return "Prospect reply: 'Sounds interesting, might be worth a quick chat.'"
    return ""


def build_expected_features(probe_id: str, rng: random.Random) -> dict[str, Any]:
    """Return expected_features dict for a given probe."""
    base_banned = ["offshore", "outsourcing", "cost reduction", "scale aggressively", "aggressive hiring"]
    features: dict[str, Any] = {
        "must_reference_signal": True,
        "must_include_cta": True,
        "banned_phrases": base_banned,
    }
    if probe_id in ("P-005", "P-006", "P-007"):
        features["required_hedges"] = ["are you finding", "it looks like", "appears", "from what we can see"]
    if probe_id in ("P-009", "P-010", "P-011"):
        features["banned_phrases"] = base_banned + ["can start monday", "immediately available", "we have 3 rust"]
        features["max_headcount_commitment"] = None
        features["must_reference_signal"] = False
    if probe_id in ("P-013", "P-014"):
        features["must_reference_signal"] = False
        features["required_hedges"] = ["consulting", "embedded", "capability extension"]
    if probe_id == "P-021":
        features["banned_phrases"] = base_banned + ["i've gone ahead and booked", "your calendar invite"]
    return features


def build_rubric() -> dict[str, int]:
    """Return empty rubric template (filled by scorer)."""
    return {
        "signal_grounding": 0,
        "tone_alignment": 0,
        "cta_quality": 0,
        "bench_fit_accuracy": 0,
        "personalization_depth": 0,
    }


def trace_to_task(
    trace: dict[str, Any],
    task_id: str,
    partition: str,
    rng: random.Random,
) -> dict[str, Any]:
    """Convert a single failed trace to a benchmark task."""
    probe_id, dimension, difficulty = classify_failure(trace)

    company_signal = build_company_signal(rng, probe_id)
    bench_summary = build_bench_summary(rng, probe_id)
    prior_thread = build_prior_thread(rng, probe_id)
    expected_features = build_expected_features(probe_id, rng)
    rubric = build_rubric()

    return {
        "task_id": task_id,
        "dimension": dimension,
        "difficulty": difficulty,
        "source_mode": "trace-derived",
        "input": {
            "company_signal": company_signal,
            "bench_summary": bench_summary,
            "prior_thread": prior_thread,
        },
        "expected_features": expected_features,
        "rubric": rubric,
        "gold_output": None,
        "notes": f"Derived from trace simulation_id={trace.get('simulation_id','unknown')}, task_id={trace.get('task_id','?')}, probe={probe_id}",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate trace-derived benchmark tasks")
    parser.add_argument("--trace-log", default="../../trp week 10/trace_log.jsonl")
    parser.add_argument("--output", default="../tenacious_bench_v0.1/train/tasks.jsonl")
    parser.add_argument("--max-tasks", type=int, default=60)
    parser.add_argument("--partition", default="train", choices=["train", "dev", "held_out"])
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    trace_path = Path(args.trace_log)
    if not trace_path.exists():
        print(f"Warning: trace log not found at {trace_path}. Generating synthetic traces.", file=sys.stderr)
        # Generate fully synthetic traces when real trace log is missing
        traces = [
            {"simulation_id": f"synth_{i:04d}", "task_id": str(i % 20 + 1),
             "reward": 0.0, "agent_cost": rng.uniform(0.008, 0.04),
             "duration": rng.uniform(50, 400)}
            for i in range(100)
        ]
    else:
        with open(trace_path, "r", encoding="utf-8") as f:
            traces = [json.loads(line) for line in f if line.strip()]

    # Filter to failed traces (reward=0.0) and sample
    failed = [t for t in traces if t.get("reward", 1.0) == 0.0]
    if not failed:
        print("No failed traces found. Using all traces.", file=sys.stderr)
        failed = traces

    rng.shuffle(failed)
    selected = failed[: args.max_tasks]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    part_prefix = {"train": "train", "dev": "dev", "held_out": "held"}[args.partition]
    tasks = []
    for i, trace in enumerate(selected):
        task_id = f"tb_{part_prefix}_{i+1:04d}"
        task = trace_to_task(trace, task_id, args.partition, rng)
        tasks.append(task)

    with open(output_path, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")

    print(f"Generated {len(tasks)} trace-derived tasks → {output_path}")
    print(f"Dimension distribution: {_count_dimensions(tasks)}")


def _count_dimensions(tasks: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in tasks:
        d = t["dimension"]
        counts[d] = counts.get(d, 0) + 1
    return counts


if __name__ == "__main__":
    main()
