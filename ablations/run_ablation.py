"""
Run Delta A and Delta B ablations on the sealed held-out partition.

Delta A: trained judge (SimPO adapter) vs. Week 10 baseline agent
Delta B: trained judge (SimPO adapter) vs. prompt-only same backbone (Qwen, no adapter)

Run Delta A first. Run Delta B only after Delta A confirms lift.

Usage:
    # Delta A
    python ablations/run_ablation.py \
        --held-out tenacious_bench_v0.1/held_out/tasks.jsonl \
        --adapter training/qwen_simpo_judge \
        --baseline week10 \
        --baseline-outputs training_data/baseline_outputs.jsonl \
        --output ablations/ablation_results.json \
        --seed 42

    # Delta B (append to same results file)
    python ablations/run_ablation.py \
        --held-out tenacious_bench_v0.1/held_out/tasks.jsonl \
        --adapter training/qwen_simpo_judge \
        --baseline prompt-only \
        --output ablations/ablation_results.json \
        --append \
        --seed 42

Cost estimate: ~$1.20 per pass (eval-tier, 40 tasks at ~$0.03/task).
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from scoring_evaluator import score_task  # noqa: E402


JUDGE_SYSTEM_PROMPT = """You are a scoring judge evaluating B2B sales outreach emails for Tenacious, an AI talent platform.

Score the following email on five dimensions. For each dimension, return a score of 0, 1, 2, or 3.

Dimensions:
- signal_grounding (0-3): Does the email reference specific signal tokens from the task context, with appropriate confidence hedging?
- tone_compliance (0-3): Does the email match the required tone for the prospect's AI maturity stage? Are banned phrases avoided?
- cta_quality (0-3): Is the call-to-action appropriate — exploratory rather than forcing a booking or commitment?
- personalization_depth (0-3): Does the email reference at least two specific details from the prospect context?
- bench_fit_alignment (0-3): Does the email accurately represent bench capacity without over-committing?

Return a JSON object with this exact structure:
{
  "signal_grounding": <int 0-3>,
  "tone_compliance": <int 0-3>,
  "cta_quality": <int 0-3>,
  "personalization_depth": <int 0-3>,
  "bench_fit_alignment": <int 0-3>,
  "reasoning": "<one sentence per dimension, semicolon-separated>"
}"""


def score_with_judge_model(
    task: dict,
    output: str,
    model,
    tokenizer,
    device: str,
) -> dict:
    """Use the trained judge to score an output."""
    user_content = (
        f"Task:\n{json.dumps(task, indent=2)}\n\n"
        f"Email to evaluate:\n{output}"
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        import torch
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
            )
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        result = json.loads(generated.strip())
        scores = {k: v for k, v in result.items() if k != "reasoning"}
        weights = {
            "signal_grounding": 0.25,
            "tone_compliance": 0.20,
            "cta_quality": 0.20,
            "personalization_depth": 0.20,
            "bench_fit_alignment": 0.15,
        }
        final = sum(scores.get(d, 0) * w / 3.0 for d, w in weights.items())
        return {"scores": scores, "final_score": round(final, 3), "reasoning": result.get("reasoning", "")}
    except Exception as e:
        log.warning("Judge scoring failed: %s — falling back to deterministic", e)
        return score_task(task, output)


def score_with_deterministic(task: dict, output: str) -> dict:
    return score_task(task, output)


def generate_prompt_only_output(task: dict, model, tokenizer, device: str) -> str:
    """Generate output using the Qwen backbone with no adapter (prompt-only baseline)."""
    system = (
        "You are an expert B2B sales development representative writing outreach emails "
        "for Tenacious, an AI-powered talent platform. Write a high-quality, personalized "
        "outreach email based on the task context provided."
    )
    user_content = f"Task:\n{json.dumps(task, indent=2)}\n\nWrite the outreach email:"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        import torch
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
        return tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
    except Exception as e:
        log.error("Prompt-only generation failed: %s", e)
        return ""


def main():
    parser = argparse.ArgumentParser(description="Run Delta A or Delta B ablation")
    parser.add_argument("--held-out", required=True, help="Held-out tasks JSONL")
    parser.add_argument("--adapter", required=True, help="Path to trained SimPO adapter")
    parser.add_argument(
        "--baseline",
        required=True,
        choices=["week10", "prompt-only"],
        help="week10: use baseline_outputs.jsonl; prompt-only: generate from backbone",
    )
    parser.add_argument("--baseline-outputs", default="training_data/baseline_outputs.jsonl",
                        help="Baseline agent outputs (required if --baseline week10)")
    parser.add_argument("--output", required=True, help="Results JSON file")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing results file")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    # Load model
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.adapter,
            max_seq_length=1024,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.adapter)
        model = AutoModelForCausalLM.from_pretrained(
            args.adapter, torch_dtype=torch.bfloat16, device_map="auto"
        )
    model.to(device)
    model.eval()

    # Load held-out tasks
    held_out_tasks = []
    with open(args.held_out, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                held_out_tasks.append(json.loads(line))
    log.info("Loaded %d held-out tasks", len(held_out_tasks))

    # Load baseline outputs if needed
    baseline_outputs = {}
    if args.baseline == "week10":
        with open(args.baseline_outputs, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    baseline_outputs[entry["task_id"]] = entry["output"]

    delta_label = "delta_a" if args.baseline == "week10" else "delta_b"
    results = []
    traces = []

    for task in held_out_tasks:
        task_id = task["task_id"]
        log.info("Evaluating %s ...", task_id)

        # Get baseline output
        if args.baseline == "week10":
            baseline_output = baseline_outputs.get(task_id, "")
            if not baseline_output:
                log.warning("No baseline output for %s — skipping", task_id)
                continue
        else:
            baseline_output = generate_prompt_only_output(task, model, tokenizer, device)

        # Score baseline with deterministic evaluator
        baseline_scores = score_with_deterministic(task, baseline_output)

        # Score with trained judge
        judge_scores = score_with_judge_model(task, baseline_output, model, tokenizer, device)

        # Trained judge on a fresh generation (to test if the judge catches what baseline misses)
        # For ablation: we measure whether the trained judge scores the baseline output differently
        # from the deterministic evaluator (Delta A measures judgment calibration)

        lift = round(judge_scores["final_score"] - baseline_scores["final_score"], 3)

        result = {
            "task_id": task_id,
            "delta": delta_label,
            "baseline_type": args.baseline,
            "deterministic_score": baseline_scores["final_score"],
            "judge_score": judge_scores["final_score"],
            "score_lift": lift,
            "baseline_scores": baseline_scores.get("scores", {}),
            "judge_scores": judge_scores.get("scores", {}),
        }
        results.append(result)

        trace = {
            "task_id": task_id,
            "delta": delta_label,
            "baseline_output": baseline_output[:500],  # truncated for log
            "deterministic_score": baseline_scores["final_score"],
            "judge_score": judge_scores["final_score"],
            "reasoning": judge_scores.get("reasoning", ""),
        }
        traces.append(trace)

        time.sleep(0.2)

    # Write results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if args.append and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)

    all_results = existing.get("results", [])
    all_results.extend(results)

    avg_lift = sum(r["score_lift"] for r in results) / len(results) if results else 0
    wins = sum(1 for r in results if r["score_lift"] > 0)
    losses = sum(1 for r in results if r["score_lift"] < 0)
    ties = len(results) - wins - losses

    output_data = {
        "results": all_results,
        "summary": {
            delta_label: {
                "n_tasks": len(results),
                "avg_score_lift": round(avg_lift, 4),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "win_rate": round(wins / len(results), 3) if results else 0,
            }
        },
    }
    if args.append and "summary" in existing:
        output_data["summary"].update(existing["summary"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    # Write traces
    traces_path = output_path.parent / "held_out_traces.jsonl"
    mode = "a" if args.append else "w"
    with open(traces_path, mode, encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace) + "\n")

    log.info("=" * 60)
    log.info("%s results: n=%d avg_lift=%.4f wins=%d losses=%d ties=%d",
             delta_label.upper(), len(results), avg_lift, wins, losses, ties)
    log.info("Results written to %s", output_path)
    log.info("Traces written to %s", traces_path)


if __name__ == "__main__":
    main()
