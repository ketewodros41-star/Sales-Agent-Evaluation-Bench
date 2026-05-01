"""
Construct preference pairs for SimPO training.

For each training task:
  1. Load the baseline agent output (from --baseline-outputs JSONL).
  2. Generate a "chosen" output via OpenRouter (DeepSeek V3.2 by default — dev-tier, spec-recommended).
  3. Score both outputs with scoring_evaluator.py logic.
  4. If score_gap >= --delta, emit a preference pair.

Output format (one JSON per line):
  {
    "task_id": "...",
    "task": {...},
    "chosen": {"output": "...", "scores": {...}, "final_score": 0.xx},
    "rejected": {"output": "...", "scores": {...}, "final_score": 0.xx},
    "score_gap": 0.xx,
    "dimension_gaps": {"signal_grounding": 0.x, ...}
  }

Run:
    python training_data/build_pairs.py \
        --tasks tenacious_bench_v0.1/train/tasks.jsonl \
        --baseline-outputs training_data/baseline_outputs.jsonl \
        --output training_data/preference_pairs.jsonl \
        --delta 0.20 \
        --seed 42

If pair count < 80, lower --delta to 0.15 for signal_grounding pairs only:
    python training_data/build_pairs.py ... --delta 0.20 --signal-grounding-delta 0.15
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Add parent to path so scoring_evaluator is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
from scoring_evaluator import score_task  # noqa: E402


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

GENERATION_SYSTEM_PROMPT = """You are an expert B2B sales development representative writing outreach emails for Tenacious, an AI-powered talent platform.

You will be given a task description with:
- company_signal: company context, signals, and their confidence levels
- bench_summary: available engineers on the bench
- prior_thread: any existing conversation context

Write a high-quality outreach email that satisfies ALL of the following:
1. SIGNAL GROUNDING: Reference the EXACT signal values from company_signal. If confidence < 0.75, phrase as a question ("are you finding..."), not an assertion.
2. TONE COMPLIANCE: Never use these banned phrases: offshore, outsourcing, cost reduction, aggressive hiring, massive growth, scale aggressively, explosive growth. Match tone to AI maturity stage.
3. CTA QUALITY: End with an exploratory CTA ("worth a quick call?", "would it be useful to compare notes?") — NEVER force a booking or commit to a specific meeting time.
4. PERSONALIZATION: Reference at least two specific details from the company_signal (e.g. specific funding round amount, specific role types, specific AI maturity score).
5. BENCH FIT: Say "we have engineers available" — NEVER state specific headcounts or make hard availability commitments.

Email should be 150-220 words, direct, and professional. Output the email body only."""


def generate_chosen_output(task: dict, api_key: str) -> Optional[str]:
    """Generate a high-quality output via OpenRouter (DeepSeek V3.2 — dev-tier)."""
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        inp = task.get("input", {})
        user_message = (
            f"COMPANY SIGNAL:\n{inp.get('company_signal', '')}\n\n"
            f"BENCH AVAILABLE:\n{inp.get('bench_summary', '')}\n"
        )
        if inp.get("prior_thread"):
            user_message += f"\nPRIOR THREAD:\n{inp['prior_thread']}\n"
        user_message += "\nWrite the outreach email body only."

        response = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324",
            max_tokens=400,
            temperature=0.7,
            messages=[
                {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        log.error("Generation failed for task %s: %s", task.get("task_id"), e)
        return None


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_output(task: dict, output: str) -> dict:
    """Score an output using the deterministic scoring evaluator."""
    result = score_task(task, output)
    # score_task returns flat keys at top level — extract them explicitly
    scores = {
        "signal_grounding": result.get("signal_grounding", 0),
        "tone_compliance": result.get("tone_alignment", 0),   # evaluator uses tone_alignment
        "cta_quality": result.get("cta_quality", 0),
        "personalization_depth": result.get("personalization_depth", 0),
        "bench_fit_alignment": result.get("bench_fit_accuracy", 0),  # evaluator uses bench_fit_accuracy
    }
    return {
        "output": output,
        "scores": scores,
        "final_score": result.get("final_score", 0.0),
    }


def compute_dimension_gaps(chosen_scores: dict, rejected_scores: dict) -> dict:
    dimensions = [
        "signal_grounding", "tone_compliance", "cta_quality",
        "personalization_depth", "bench_fit_alignment"
    ]
    gaps = {}
    for dim in dimensions:
        chosen_val = chosen_scores.get(dim, 0)
        rejected_val = rejected_scores.get(dim, 0)
        c = float(chosen_val) / 3.0
        r = float(rejected_val) / 3.0
        gaps[dim] = round(c - r, 3)
    return gaps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build SimPO preference pairs")
    parser.add_argument("--tasks", required=True, help="Train tasks JSONL")
    parser.add_argument("--baseline-outputs", required=True,
                        help="Baseline agent outputs JSONL — one per line: {task_id, output}")
    parser.add_argument("--output", required=True, help="Output preference pairs JSONL")
    parser.add_argument("--delta", type=float, default=0.20,
                        help="Minimum score gap for a valid pair (normalized 0-1 scale)")
    parser.add_argument("--signal-grounding-delta", type=float, default=None,
                        help="Override delta for signal_grounding dimension pairs (use 0.15 if count < 80)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        log.error("OPENROUTER_API_KEY environment variable not set.")
        sys.exit(1)

    # Load tasks
    tasks = {}
    with open(args.tasks, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                task = json.loads(line)
                tasks[task["task_id"]] = task
    log.info("Loaded %d training tasks", len(tasks))

    # Load baseline outputs
    baseline_outputs = {}
    with open(args.baseline_outputs, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                baseline_outputs[entry["task_id"]] = entry["output"]
    log.info("Loaded %d baseline outputs", len(baseline_outputs))

    missing = set(tasks.keys()) - set(baseline_outputs.keys())
    if missing:
        log.warning("%d tasks have no baseline output — will be skipped: %s",
                    len(missing), sorted(missing)[:5])

    pairs = []
    skipped_no_gap = 0
    skipped_no_generation = 0
    skipped_no_baseline = 0

    task_ids = sorted(set(tasks.keys()) & set(baseline_outputs.keys()))
    random.shuffle(task_ids)

    for task_id in task_ids:
        task = tasks[task_id]
        baseline_output = baseline_outputs[task_id]

        log.info("Processing %s ...", task_id)

        # Score baseline (rejected)
        rejected_result = score_output(task, baseline_output)

        # Generate chosen
        chosen_text = generate_chosen_output(task, api_key)
        if not chosen_text:
            skipped_no_generation += 1
            continue

        # Score chosen
        chosen_result = score_output(task, chosen_text)

        score_gap = round(chosen_result["final_score"] - rejected_result["final_score"], 3)
        dimension_gaps = compute_dimension_gaps(
            chosen_result["scores"], rejected_result["scores"]
        )

        # Determine effective delta for this task
        effective_delta = args.delta
        if args.signal_grounding_delta is not None:
            sg_gap = dimension_gaps.get("signal_grounding", 0.0)
            if sg_gap >= args.signal_grounding_delta and score_gap < args.delta:
                effective_delta = args.signal_grounding_delta
                log.debug(
                    "Applying signal_grounding exception for %s (gap=%.3f)", task_id, sg_gap
                )

        if score_gap < effective_delta:
            log.debug(
                "Skipping %s — gap %.3f below threshold %.2f",
                task_id, score_gap, effective_delta
            )
            skipped_no_gap += 1
            continue

        pair = {
            "task_id": task_id,
            "task": task,
            "chosen": chosen_result,
            "rejected": rejected_result,
            "score_gap": score_gap,
            "dimension_gaps": dimension_gaps,
            "effective_delta": effective_delta,
        }
        pairs.append(pair)
        log.info(
            "Valid pair %s — chosen=%.3f rejected=%.3f gap=%.3f",
            task_id,
            chosen_result["final_score"],
            rejected_result["final_score"],
            score_gap,
        )

        # Throttle to avoid rate limit
        time.sleep(0.5)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    log.info("=" * 60)
    log.info("Preference pair construction complete")
    log.info("  Valid pairs:       %d", len(pairs))
    log.info("  Skipped (no gap):  %d", skipped_no_gap)
    log.info("  Skipped (no gen):  %d", skipped_no_generation)
    log.info("  Skipped (no base): %d", skipped_no_baseline)

    if len(pairs) < 80:
        log.warning(
            "Only %d valid pairs — below the 80-pair target. "
            "Re-run with --signal-grounding-delta 0.15 to increase yield.",
            len(pairs)
        )
    else:
        log.info("Target of ≥80 pairs met. Ready for SimPO training.")

    # Summary statistics
    if pairs:
        gaps = [p["score_gap"] for p in pairs]
        log.info(
            "Gap statistics — min=%.3f median=%.3f max=%.3f",
            min(gaps),
            sorted(gaps)[len(gaps) // 2],
            max(gaps),
        )


if __name__ == "__main__":
    main()
