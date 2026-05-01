"""
generate_adversarial_tasks.py — Validate, catalog, and manage hand-authored adversarial tasks.

Hand-authored adversarial tasks are written directly by the benchmark author to defeat the
Week 10 Tenacious agent on failure modes the automated pipeline misses. This script:
  1. Validates each task against the Tenacious-Bench JSON schema
  2. Confirms source_mode == "adversarial" on every task
  3. Reports which failure modes (adversarial_type) are covered
  4. Writes a validated copy to the output path

Adversarial types covered:
  - bench_overcommit_direct_question: Agent asked directly about availability; bench is partially filled
  - tone_hostile_objection: Prospect is hostile; agent must de-escalate without policy violations
  - confidence_conflation: AI maturity signal is ambiguous; agent must not conflate readiness with maturity
  - signal_hallucination: Agent has weak signals; must hedge rather than fabricate specifics
  - cta_pressure: Prospect has low intent; agent must not force booking language

Usage:
    python generation_scripts/generate_adversarial_tasks.py \
        --input tenacious_bench_v0.1/train/tasks.jsonl \
        --output generation_scripts/adversarial_catalog.jsonl \
        --report
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REQUIRED_FIELDS = {"task_id", "source_mode", "dimension", "input", "expected_features"}
REQUIRED_INPUT_FIELDS = {"company_signal", "bench_summary"}
ADVERSARIAL_TYPES = {
    "bench_overcommit_direct_question",
    "tone_hostile_objection",
    "confidence_conflation",
    "signal_hallucination",
    "cta_pressure",
}


def validate_task(task: dict, idx: int) -> list[str]:
    errors = []
    missing = REQUIRED_FIELDS - set(task.keys())
    if missing:
        errors.append(f"task[{idx}] missing fields: {missing}")
        return errors

    if task.get("source_mode") != "adversarial":
        errors.append(
            f"task[{idx}] ({task.get('task_id')}) has source_mode="
            f"'{task.get('source_mode')}', expected 'adversarial'"
        )

    inp = task.get("input", {})
    missing_input = REQUIRED_INPUT_FIELDS - set(inp.keys())
    if missing_input:
        errors.append(f"task[{idx}] ({task.get('task_id')}) input missing: {missing_input}")

    ef = task.get("expected_features", {})
    if not ef:
        errors.append(f"task[{idx}] ({task.get('task_id')}) expected_features is empty")

    return errors


def load_adversarial_tasks(path: Path) -> list[dict]:
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            if task.get("source_mode") == "adversarial":
                tasks.append(task)
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Validate and catalog hand-authored adversarial tasks"
    )
    parser.add_argument(
        "--input",
        default="tenacious_bench_v0.1/train/tasks.jsonl",
        help="JSONL file containing tasks (all partitions or adversarial-only)",
    )
    parser.add_argument(
        "--output",
        default="generation_scripts/adversarial_catalog.jsonl",
        help="Output path for validated adversarial tasks",
    )
    parser.add_argument(
        "--report", action="store_true", help="Print coverage report to stdout"
    )
    parser.add_argument(
        "--all-partitions", nargs="+",
        default=[
            "tenacious_bench_v0.1/train/tasks.jsonl",
            "tenacious_bench_v0.1/dev/tasks.jsonl",
            "tenacious_bench_v0.1/held_out/tasks.jsonl",
        ],
        help="Load adversarial tasks from all partitions",
    )
    args = parser.parse_args()

    # Load from all partitions by default for complete catalog
    all_tasks = []
    for partition_path in args.all_partitions:
        p = Path(partition_path)
        if p.exists():
            tasks = load_adversarial_tasks(p)
            print(f"  {partition_path}: {len(tasks)} adversarial tasks", file=sys.stderr)
            all_tasks.extend(tasks)
        else:
            print(f"  Warning: {partition_path} not found", file=sys.stderr)

    print(f"\nTotal adversarial tasks found: {len(all_tasks)}", file=sys.stderr)

    # Validate
    all_errors = []
    for i, task in enumerate(all_tasks):
        errors = validate_task(task, i)
        all_errors.extend(errors)

    if all_errors:
        print(f"\nValidation errors ({len(all_errors)}):", file=sys.stderr)
        for err in all_errors:
            print(f"  ERROR: {err}", file=sys.stderr)
        sys.exit(1)

    print(f"Validation: PASS — all {len(all_tasks)} adversarial tasks valid", file=sys.stderr)

    # Write catalog
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for task in all_tasks:
            f.write(json.dumps(task) + "\n")
    print(f"Catalog written to {output_path}", file=sys.stderr)

    if args.report:
        dimension_counts = Counter(t.get("dimension") for t in all_tasks)
        adversarial_type_counts = Counter(
            t.get("adversarial_type", "unspecified") for t in all_tasks
        )
        difficulty_counts = Counter(t.get("difficulty", "unspecified") for t in all_tasks)

        print("\n=== Adversarial Task Coverage Report ===")
        print(f"\nTotal adversarial tasks: {len(all_tasks)} (target: 30)")
        print(f"Share of 200-task benchmark: {len(all_tasks)/200:.1%} (target: 15%)")

        print("\nBy rubric dimension:")
        for dim, count in sorted(dimension_counts.items()):
            print(f"  {dim}: {count}")

        print("\nBy adversarial type:")
        for atype, count in sorted(adversarial_type_counts.items()):
            covered = "✓" if atype in ADVERSARIAL_TYPES else "?"
            print(f"  {covered} {atype}: {count}")

        missing_types = ADVERSARIAL_TYPES - set(adversarial_type_counts.keys())
        if missing_types:
            print(f"\nMissing adversarial types (add to benchmark for full coverage):")
            for t in sorted(missing_types):
                print(f"  - {t}")

        print("\nBy difficulty:")
        for diff, count in sorted(difficulty_counts.items()):
            print(f"  {diff}: {count}")

        print("\n=== End Report ===")


if __name__ == "__main__":
    main()
