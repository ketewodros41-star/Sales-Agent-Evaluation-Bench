"""
judge_filter.py — LLM-as-a-judge quality filter for Tenacious-Bench task candidates.

Applies three-dimension pointwise scoring to generated tasks:
  1. Input coherence (1–5): Does the company_signal make logical sense? Is the bench_summary consistent?
  2. Ground-truth verifiability (1–5): Can a human scorer deterministically apply the rubric from the input?
  3. Rubric clarity (1–5): Are the expected_features specific enough to produce consistent labels?

Threshold for inclusion: each dimension ≥ 3, mean ≥ 3.5.

Also supports:
  --pairwise: pairwise comparison for near-duplicate task pairs (picks the more diagnostic task)
  --spot-check-n N: run eval-tier model on N sampled tasks for calibration after dev-tier bulk filter

Tier separation policy (Li et al. 2025 preference leakage prevention):
  - Bulk filtering: dev-tier model (default: qwen/qwen3-next-80b-a3b)
  - Spot-check calibration: eval-tier model (default: anthropic/claude-sonnet-4-6)
  - The same model must never generate and judge the same task

Usage:
    python judge_filter.py \
        --input candidates.jsonl \
        --output filtered.jsonl \
        --model qwen/qwen3-next-80b-a3b \
        --spot-check-n 50 \
        --pairwise \
        --dry-run
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

COHERENCE_THRESHOLD = 3
VERIFIABILITY_THRESHOLD = 3
CLARITY_THRESHOLD = 3
MEAN_THRESHOLD = 3.5

# ---------------------------------------------------------------------------
# Heuristic scoring (no API required, used for dry-run and fallback)
# ---------------------------------------------------------------------------

def heuristic_score(task: dict[str, Any]) -> dict[str, Any]:
    """
    Deterministic heuristic scorer. Approximates judge quality without API calls.
    Used for dry-run mode and when the API is unavailable.
    """
    inp = task.get("input", {})
    company_signal = inp.get("company_signal", "")
    bench_summary = inp.get("bench_summary", "")
    expected = task.get("expected_features", {})

    # --- Coherence: check that company_signal has required fields ---
    coherence = 1
    required_signal_tokens = ["Company:", "Open engineering roles:", "AI maturity score:"]
    matched = sum(1 for tok in required_signal_tokens if tok in company_signal)
    if matched == 3:
        coherence = 5
    elif matched == 2:
        coherence = 4
    elif matched == 1:
        coherence = 2

    # bench_summary coherence
    if "Available bench:" not in bench_summary:
        coherence = min(coherence, 2)

    # --- Verifiability: check that expected_features are specific enough ---
    verifiability = 3
    banned = expected.get("banned_phrases", [])
    if len(banned) >= 3:
        verifiability = 5
    elif len(banned) >= 1:
        verifiability = 4
    elif len(banned) == 0:
        verifiability = 2

    must_ref = expected.get("must_reference_signal", True)
    must_cta = expected.get("must_include_cta", True)
    if must_ref or must_cta:
        verifiability = max(verifiability, 3)

    # --- Rubric clarity: check that dimension is consistent with expected_features ---
    clarity = 3
    dimension = task.get("dimension", "")
    hedges = expected.get("required_hedges", [])

    if dimension == "signal_grounding" and not hedges and len(company_signal) > 100:
        clarity = 4
    elif dimension == "bench_fit_alignment" and "Rust" in bench_summary:
        clarity = 5
    elif dimension == "tone_compliance" and len(banned) >= 5:
        clarity = 5
    elif dimension in ("cta_quality", "personalization") and must_cta:
        clarity = 4

    mean_score = (coherence + verifiability + clarity) / 3.0

    return {
        "input_coherence": coherence,
        "ground_truth_verifiability": verifiability,
        "rubric_clarity": clarity,
        "mean_score": round(mean_score, 2),
        "include": (
            coherence >= COHERENCE_THRESHOLD
            and verifiability >= VERIFIABILITY_THRESHOLD
            and clarity >= CLARITY_THRESHOLD
            and mean_score >= MEAN_THRESHOLD
        ),
        "source": "heuristic",
    }


# ---------------------------------------------------------------------------
# LLM judge scoring (requires OPENROUTER_API_KEY)
# ---------------------------------------------------------------------------

# Judge prompt is committed verbatim in generation_scripts/judge_prompt.md.
# The string below must remain identical to the prompt body in that file.
JUDGE_PROMPT_TEMPLATE = """You are a benchmark quality judge for a B2B sales outreach evaluation dataset called Tenacious-Bench.

Score the following benchmark task on three dimensions, each 1–5:
1. Input coherence: Does the company_signal make logical sense? Is the bench_summary internally consistent with the company signal?
2. Ground-truth verifiability: Can a human scorer deterministically apply the rubric from the inputs alone (without additional context)?
3. Rubric clarity: Are the expected_features (banned_phrases, must_reference_signal, required_hedges) specific enough to produce consistent labels across labelers?

Respond with ONLY a JSON object in this exact format:
{{"input_coherence": <int 1-5>, "ground_truth_verifiability": <int 1-5>, "rubric_clarity": <int 1-5>, "one_line_note": "<short note>"}}

Task to score:
{task_json}
"""


def llm_score(task: dict[str, Any], model: str, api_key: str) -> dict[str, Any]:
    if not HTTPX_AVAILABLE:
        print("Warning: httpx not installed. Falling back to heuristic scoring.", file=sys.stderr)
        return heuristic_score(task)

    task_json = json.dumps({k: v for k, v in task.items() if k != "gold_output"}, indent=2)
    prompt = JUDGE_PROMPT_TEMPLATE.format(task_json=task_json[:3000])

    try:
        response = httpx.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 200,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()

        # Extract JSON from response
        json_match = re.search(r"\{[^}]+\}", content, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in response: {content[:200]}")
        scores = json.loads(json_match.group(0))

        coherence = int(scores.get("input_coherence", 3))
        verifiability = int(scores.get("ground_truth_verifiability", 3))
        clarity = int(scores.get("rubric_clarity", 3))
        mean_score = (coherence + verifiability + clarity) / 3.0

        return {
            "input_coherence": coherence,
            "ground_truth_verifiability": verifiability,
            "rubric_clarity": clarity,
            "mean_score": round(mean_score, 2),
            "one_line_note": scores.get("one_line_note", ""),
            "include": (
                coherence >= COHERENCE_THRESHOLD
                and verifiability >= VERIFIABILITY_THRESHOLD
                and clarity >= CLARITY_THRESHOLD
                and mean_score >= MEAN_THRESHOLD
            ),
            "source": "llm",
        }
    except Exception as e:
        print(f"LLM judge error: {e}. Falling back to heuristic.", file=sys.stderr)
        return heuristic_score(task)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PAIRWISE_PROMPT_TEMPLATE = """You are a benchmark quality judge for Tenacious-Bench.

Two candidate tasks cover similar scenarios. Pick the MORE DIAGNOSTIC one — the task that
better distinguishes a Tenacious-compliant email from a non-compliant one.

Task A:
{task_a}

Task B:
{task_b}

Respond with ONLY a JSON object:
{{"winner": "A" or "B", "reason": "<one sentence>"}}
"""


def pairwise_compare(
    task_a: dict, task_b: dict, model: str, api_key: str, dry_run: bool
) -> str:
    """Return 'A' or 'B' — whichever task is more diagnostic."""
    if dry_run or not HTTPX_AVAILABLE:
        # Heuristic: prefer the task with more expected_features fields
        a_score = len(task_a.get("expected_features", {}))
        b_score = len(task_b.get("expected_features", {}))
        return "A" if a_score >= b_score else "B"

    prompt = PAIRWISE_PROMPT_TEMPLATE.format(
        task_a=json.dumps({k: v for k, v in task_a.items() if k != "gold_output"}, indent=2)[:2000],
        task_b=json.dumps({k: v for k, v in task_b.items() if k != "gold_output"}, indent=2)[:2000],
    )
    try:
        response = httpx.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 100,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        match = re.search(r'"winner"\s*:\s*"([AB])"', content)
        return match.group(1) if match else "A"
    except Exception as e:
        print(f"Pairwise comparison error: {e}. Defaulting to A.", file=sys.stderr)
        return "A"


def find_near_duplicates(tasks: list[dict], threshold: float = 0.7) -> list[tuple[int, int]]:
    """Return index pairs of tasks with high input text overlap (simple token Jaccard)."""
    pairs = []
    texts = [set(get_input_tokens(t)) for t in tasks]
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            if not texts[i] or not texts[j]:
                continue
            jaccard = len(texts[i] & texts[j]) / len(texts[i] | texts[j])
            if jaccard >= threshold:
                pairs.append((i, j))
    return pairs


def get_input_tokens(task: dict) -> list[str]:
    inp = task.get("input", {})
    text = " ".join([
        inp.get("company_signal", ""),
        inp.get("bench_summary", ""),
        inp.get("prior_thread", ""),
    ])
    return text.lower().split()


def run_spot_check(
    passed_tasks: list[dict], n: int, eval_model: str, api_key: str
) -> dict:
    """Sample n tasks from passed set and re-score with eval-tier model for calibration."""
    import random
    sample_size = min(n, len(passed_tasks))
    sample = random.sample(passed_tasks, sample_size)
    agreements = 0
    disagreements = []

    print(f"\nSpot-check: scoring {sample_size} tasks with eval-tier model {eval_model}...",
          file=sys.stderr)
    for task in sample:
        eval_scores = llm_score(task, eval_model, api_key)
        dev_scores = heuristic_score(task)  # compare against heuristic as proxy
        if eval_scores["include"] == dev_scores["include"]:
            agreements += 1
        else:
            disagreements.append({
                "task_id": task.get("task_id"),
                "dev_tier_include": dev_scores["include"],
                "eval_tier_include": eval_scores["include"],
                "eval_mean": eval_scores["mean_score"],
                "dev_mean": dev_scores["mean_score"],
            })

    agreement_rate = agreements / sample_size if sample_size > 0 else 0
    return {
        "spot_check_n": sample_size,
        "eval_model": eval_model,
        "agreement_rate": round(agreement_rate, 3),
        "disagreements": disagreements,
        "calibration_status": "pass" if agreement_rate >= 0.80 else "review_thresholds",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge filter for Tenacious-Bench task candidates")
    parser.add_argument("--input", required=True, help="Path to candidate tasks JSONL")
    parser.add_argument("--output", required=True, help="Path to filtered tasks JSONL")
    parser.add_argument("--model", default="qwen/qwen3-next-80b-a3b",
                        help="Dev-tier OpenRouter model for bulk filtering")
    parser.add_argument("--eval-model", default="anthropic/claude-sonnet-4-6",
                        help="Eval-tier model for spot-check calibration")
    parser.add_argument("--dry-run", action="store_true", help="Use heuristic scoring (no API calls)")
    parser.add_argument("--log", default=None, help="Path to write per-task judge scores JSONL")
    parser.add_argument("--pairwise", action="store_true",
                        help="Run pairwise comparison on near-duplicate task pairs")
    parser.add_argument("--spot-check-n", type=int, default=0,
                        help="Number of passed tasks to re-score with eval-tier model (0=skip)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and not args.dry_run:
        print("Warning: OPENROUTER_API_KEY not set. Using dry-run heuristic mode.", file=sys.stderr)
        args.dry_run = True

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        candidates = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(candidates)} candidate tasks", file=sys.stderr)

    # --- Step 1: Pairwise deduplication ---
    if args.pairwise:
        print("Running pairwise near-duplicate detection...", file=sys.stderr)
        dup_pairs = find_near_duplicates(candidates)
        print(f"  Found {len(dup_pairs)} near-duplicate pairs", file=sys.stderr)
        drop_indices = set()
        for i, j in dup_pairs:
            if i in drop_indices or j in drop_indices:
                continue
            winner = pairwise_compare(candidates[i], candidates[j], args.model, api_key, args.dry_run)
            loser_idx = j if winner == "A" else i
            drop_indices.add(loser_idx)
        candidates = [t for idx, t in enumerate(candidates) if idx not in drop_indices]
        print(f"  After pairwise dedup: {len(candidates)} candidates remain", file=sys.stderr)

    # --- Step 2: Pointwise bulk filtering (dev-tier) ---
    passed = []
    logs = []

    for i, task in enumerate(candidates):
        if args.dry_run:
            scores = heuristic_score(task)
        else:
            scores = llm_score(task, args.model, api_key)

        log_entry = {"task_id": task.get("task_id", f"cand_{i}"), **scores}
        logs.append(log_entry)

        if scores["include"]:
            passed.append(task)

        if (i + 1) % 20 == 0:
            print(f"  Scored {i+1}/{len(candidates)}, passed so far: {len(passed)}", file=sys.stderr)

    # --- Step 3: Eval-tier spot-check calibration ---
    spot_check_result = None
    if args.spot_check_n > 0 and passed and api_key and not args.dry_run:
        spot_check_result = run_spot_check(passed, args.spot_check_n, args.eval_model, api_key)
        print(f"\nSpot-check calibration: agreement={spot_check_result['agreement_rate']:.1%} "
              f"({spot_check_result['calibration_status']})", file=sys.stderr)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for task in passed:
            f.write(json.dumps(task) + "\n")

    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            for entry in logs:
                f.write(json.dumps(entry) + "\n")
        if spot_check_result:
            with open(log_path.with_suffix(".spotcheck.json"), "w") as f:
                json.dump(spot_check_result, f, indent=2)

    acceptance_rate = len(passed) / len(candidates) if candidates else 0
    print(f"\nJudge filter complete:")
    print(f"  Input candidates: {len(candidates)}")
    print(f"  Passed filter: {len(passed)} ({acceptance_rate:.1%})")
    print(f"  Rejected: {len(candidates) - len(passed)}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
