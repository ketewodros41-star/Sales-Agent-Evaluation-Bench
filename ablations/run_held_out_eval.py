"""
Held-out ablation evaluation for Tenacious-Bench v0.1.

Delta A: trained pipeline (DeepSeek V3.2 + Tenacious-specialized GENERATION_SYSTEM_PROMPT)
         vs Week 10 baseline (gpt-4o-mini + Week 10 system prompt)
         Scored by deterministic evaluator.

Delta B: trained pipeline (DeepSeek V3.2 + specialized prompt)
         vs prompt-only backbone (DeepSeek V3.2 + generic prompt, no training-informed constraints)
         Tests whether the training-informed prompt beats a plain prompt on the same backbone.

Run Delta A:
    python ablations/run_held_out_eval.py \
        --held-out tenacious_bench_v0.1/held_out/tasks.jsonl \
        --delta a \
        --output ablations/ablation_results.json \
        --seed 42

Run Delta B (appends to same file):
    python ablations/run_held_out_eval.py \
        --held-out tenacious_bench_v0.1/held_out/tasks.jsonl \
        --delta b \
        --output ablations/ablation_results.json \
        --append \
        --seed 42

Recommended models (week11.md):
  - Trained pipeline:  deepseek/deepseek-chat-v3-0324  (dev-tier)
  - Baseline:          openai/gpt-4o-mini               (dev-tier, Week 10 backbone)
  - Eval-tier scoring: deterministic evaluator (validated κ >= 0.80 on all dimensions)
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from scoring_evaluator import score_task  # noqa: E402

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from openai import OpenAI

# --------------------------------------------------------------------------
# System prompts
# --------------------------------------------------------------------------

# Trained pipeline prompt — mirrors GENERATION_SYSTEM_PROMPT in build_pairs.py
# This is what the SimPO training teaches the judge to prefer.
TRAINED_PIPELINE_PROMPT = """You are an expert B2B sales development representative writing outreach emails for Tenacious, an AI-powered talent platform.

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

# Week 10 baseline prompt — same as run_baseline.py
BASELINE_PROMPT = """You are writing outbound email for Tenacious Consulting and Outsourcing.

Tenacious tone markers:
1. DIRECT: Short sentences. No fluff. Get to the point in the first line.
2. GROUNDED: Every claim references a specific, verifiable signal. No vague assertions.
3. RESPECTFUL: Never condescending. Never assume the prospect doesn't know their market.
4. HONEST UNDER UNCERTAINTY: Low-confidence signals are phrased as questions, not assertions.
5. HUMAN: Reads like a thoughtful person wrote it, not a template.
6. NEVER: 'offshore' or 'outsourcing' in subject lines. Never 'scale faster than your competitors'.
7. ACV REFERENCE: Don't quote specific pricing. Route deeper pricing to human.
8. CHANNEL: Email is for cold outreach. SMS only for scheduling from warm leads who replied.

Rules:
- Every factual claim must be grounded in the provided company_signal
- If a signal has confidence < 0.75, phrase as a question, not an assertion
- Never claim 'aggressive hiring' if fewer than 5 open roles
- Never commit to bench capacity — say 'we have engineers available' not specific counts
- Never mention competitors by name in outreach emails
- Keep emails under 180 words
- Do not fabricate additional case studies beyond those in seed materials"""

# Generic prompt for Delta B — no training-informed constraints, plain backbone
GENERIC_PROMPT = """You are a sales development representative. Write a professional B2B outreach email for a talent platform called Tenacious.

Use the provided company information and available engineers to write a personalized, concise email.
Output the email body only."""


# --------------------------------------------------------------------------
# Generation helpers
# --------------------------------------------------------------------------

def build_user_message(task: dict) -> str:
    inp = task.get("input", {})
    msg = (
        f"COMPANY SIGNAL:\n{inp.get('company_signal', '')}\n\n"
        f"BENCH AVAILABLE:\n{inp.get('bench_summary', '')}\n"
    )
    if inp.get("prior_thread"):
        msg += f"\nPRIOR THREAD:\n{inp['prior_thread']}\n"
    msg += "\nWrite the outreach email body only. Under 220 words."
    return msg


def generate_email(task: dict, system_prompt: str, model: str, client: OpenAI) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=400,
            temperature=0.7,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user_message(task)},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        log.error("Generation failed for %s: %s", task.get("task_id"), e)
        return ""


def score_output(task: dict, output: str) -> dict:
    result = score_task(task, output)
    return {
        "final_score": result.get("final_score", 0.0),
        "scores": {
            "signal_grounding": result.get("signal_grounding", 0),
            "tone_compliance": result.get("tone_alignment", 0),
            "cta_quality": result.get("cta_quality", 0),
            "personalization_depth": result.get("personalization_depth", 0),
            "bench_fit_alignment": result.get("bench_fit_accuracy", 0),
        },
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run held-out ablation evaluation")
    parser.add_argument("--held-out", required=True, help="Held-out tasks JSONL (40 tasks)")
    parser.add_argument(
        "--delta",
        required=True,
        choices=["a", "b"],
        help="a: trained-pipeline vs baseline; b: trained-pipeline vs prompt-only backbone",
    )
    parser.add_argument("--output", required=True, help="ablation_results.json path")
    parser.add_argument("--append", action="store_true", help="Append to existing results file")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        log.error("OPENROUTER_API_KEY not set")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    # Model selection (week11.md recommended models)
    TRAINED_MODEL = "deepseek/deepseek-chat-v3-0324"   # dev-tier, same as training data generation
    BASELINE_MODEL = "openai/gpt-4o-mini"               # Week 10 backbone

    # Load held-out tasks
    held_out_tasks = []
    with open(args.held_out, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                held_out_tasks.append(json.loads(line))
    log.info("Loaded %d held-out tasks", len(held_out_tasks))

    delta_label = f"delta_{args.delta}"
    results = []
    traces = []

    for i, task in enumerate(held_out_tasks, 1):
        task_id = task["task_id"]
        log.info("[%d/%d] %s", i, len(held_out_tasks), task_id)

        if args.delta == "a":
            # Delta A: trained pipeline (DeepSeek + specialized prompt) vs baseline (gpt-4o-mini)
            trained_output = generate_email(task, TRAINED_PIPELINE_PROMPT, TRAINED_MODEL, client)
            time.sleep(0.4)
            baseline_output = generate_email(task, BASELINE_PROMPT, BASELINE_MODEL, client)
            comparison_label = "baseline_week10"
            comparison_model = BASELINE_MODEL

        else:
            # Delta B: trained pipeline vs prompt-only backbone (same DeepSeek, generic prompt)
            trained_output = generate_email(task, TRAINED_PIPELINE_PROMPT, TRAINED_MODEL, client)
            time.sleep(0.4)
            baseline_output = generate_email(task, GENERIC_PROMPT, TRAINED_MODEL, client)
            comparison_label = "prompt_only_backbone"
            comparison_model = TRAINED_MODEL

        if not trained_output or not baseline_output:
            log.warning("Skipping %s — generation failed", task_id)
            continue

        trained_scored = score_output(task, trained_output)
        baseline_scored = score_output(task, baseline_output)

        score_lift = round(trained_scored["final_score"] - baseline_scored["final_score"], 3)

        result = {
            "task_id": task_id,
            "delta": delta_label,
            "baseline_type": comparison_label,
            "trained_score": trained_scored["final_score"],
            "baseline_score": baseline_scored["final_score"],
            "score_lift": score_lift,
            "trained_scores": trained_scored["scores"],
            "baseline_scores": baseline_scored["scores"],
            "trained_model": TRAINED_MODEL,
            "baseline_model": comparison_model,
        }
        results.append(result)

        trace = {
            "task_id": task_id,
            "delta": delta_label,
            "trained_output": trained_output[:600],
            "baseline_output": baseline_output[:600],
            "trained_score": trained_scored["final_score"],
            "baseline_score": baseline_scored["final_score"],
            "score_lift": score_lift,
        }
        traces.append(trace)

        log.info(
            "  trained=%.3f  baseline=%.3f  lift=%+.3f",
            trained_scored["final_score"],
            baseline_scored["final_score"],
            score_lift,
        )

        time.sleep(0.4)

    # Write results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_results = []
    existing_summary = {}
    if args.append and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
            existing_results = existing.get("results", [])
            existing_summary = existing.get("summary", {})

    all_results = existing_results + results

    avg_lift = sum(r["score_lift"] for r in results) / len(results) if results else 0.0
    wins = sum(1 for r in results if r["score_lift"] > 0)
    losses = sum(1 for r in results if r["score_lift"] < 0)
    ties = len(results) - wins - losses

    summary = existing_summary.copy()
    summary[delta_label] = {
        "n_tasks": len(results),
        "avg_score_lift": round(avg_lift, 4),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": round(wins / len(results), 3) if results else 0.0,
        "trained_model": TRAINED_MODEL,
        "baseline_model": comparison_model if results else "",
        "scoring": "deterministic_evaluator",
    }

    output_data = {"results": all_results, "summary": summary}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    # Write traces
    traces_path = output_path.parent / "held_out_traces.jsonl"
    mode = "a" if args.append else "w"
    with open(traces_path, mode, encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace) + "\n")

    log.info("=" * 60)
    log.info(
        "%s: n=%d  avg_lift=%.4f  wins=%d  losses=%d  ties=%d  win_rate=%.3f",
        delta_label.upper(), len(results), avg_lift, wins, losses, ties,
        wins / len(results) if results else 0.0,
    )
    log.info("Results → %s", output_path)
    log.info("Traces  → %s", traces_path)

    if avg_lift > 0:
        log.info("Positive lift. Run bootstrap_test.py for significance.")
    else:
        log.warning("Non-positive lift. Report this honestly in the blog post.")


if __name__ == "__main__":
    main()
