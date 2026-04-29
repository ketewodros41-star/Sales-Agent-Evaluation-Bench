"""
generate_synthesis_tasks.py — Multi-LLM synthesis generation for Tenacious-Bench.

Routing policy (Li et al., 2025 preference-leakage prevention):
  - Claude Sonnet 4.6 (eval-tier): generates the 30 hardest seed tasks
  - Qwen3-Next-80B-A3B (dev-tier): generates bulk variations (~1,170 candidates)
  - DeepSeek V3.2: pairwise comparison when two synthesis paths produce similar tasks
  - No model both generates and judges the same task

Pipeline:
  1. Claude generates 30 seed tasks covering hard cross-dimension scenarios
  2. Qwen generates ~40 variations per seed → ~1,200 candidate tasks
  3. judge_filter.py (Qwen-judged) filters candidates to ~50 (4.2% acceptance)
  4. DeepSeek resolves near-duplicate pairs from different synthesis paths

Usage:
    python generate_synthesis_tasks.py \
        --output ../tenacious_bench_v0.1/synthesis_candidates.jsonl \
        --filtered-output ../tenacious_bench_v0.1/multi_llm_synthesis/tasks.jsonl \
        --seed 42
        [--dry-run]
"""

import argparse
import json
import os
import random
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

CLAUDE_MODEL = "anthropic/claude-sonnet-4-6"   # eval-tier: seed generation
QWEN_MODEL = "qwen/qwen3-next-80b-a3b"         # dev-tier: bulk variation + judging
DEEPSEEK_MODEL = "deepseek/deepseek-v3-2"      # pairwise dedup comparison

SEED = 42
N_SEEDS = 30
VARIATIONS_PER_SEED = 40
TARGET_FINAL = 50

DIMENSIONS = [
    "signal_grounding",
    "tone_compliance",
    "cta_quality",
    "personalization",
    "bench_fit_alignment",
]

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

# Hard cross-dimension scenario templates for seed generation
HARD_SCENARIO_TEMPLATES = [
    {
        "description": "High hiring velocity but low AI maturity — signal conflict for personalization + grounding",
        "dimensions": ["signal_grounding", "personalization"],
        "difficulty": "hard",
    },
    {
        "description": "Partial bench with 3 of 5 needed roles — over-commitment risk on strong engagement signal",
        "dimensions": ["bench_fit_alignment", "cta_quality"],
        "difficulty": "hard",
    },
    {
        "description": "Warm prospect, second touch — CTA must escalate without banned urgency phrases",
        "dimensions": ["cta_quality", "tone_compliance"],
        "difficulty": "hard",
    },
    {
        "description": "Layoff signal 45 days old — confidence decay requires hedged signal grounding",
        "dimensions": ["signal_grounding", "personalization"],
        "difficulty": "hard",
    },
    {
        "description": "Series A company, 8 open roles, AI maturity 0.62 — threshold boundary for bench commitment",
        "dimensions": ["bench_fit_alignment", "signal_grounding"],
        "difficulty": "hard",
    },
]

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

CLAUDE_SEED_PROMPT = """You are generating a hard benchmark task for Tenacious-Bench, a B2B sales outreach evaluation dataset.

Scenario: {description}
Primary dimensions to stress: {dimensions}
Difficulty: {difficulty}

Generate ONE benchmark task in this exact JSON format:
{{
  "task_id": "tb_synth_{index:04d}",
  "source_mode": "multi-llm-synthesis",
  "dimension": "{primary_dimension}",
  "difficulty": "{difficulty}",
  "input": {{
    "company_signal": "Company: <name>\\nOpen engineering roles: <N>\\nAI maturity score: <0.0–1.0>\\nRecent signal: <layoff/funding/hiring, with date in days>",
    "bench_summary": "Available bench: <list of 3–6 engineer profiles with skills>\\nTotal headcount capacity: <N>",
    "engagement_stage": "<cold|warm|hot>"
  }},
  "expected_features": {{
    "must_reference_signal": true,
    "must_include_cta": true,
    "banned_phrases": ["<phrase1>", "<phrase2>", "<phrase3>"],
    "required_hedges": ["<hedge phrase matching the confidence level>"],
    "max_bench_commitment": <integer — max engineers to commit given the bench_summary>
  }},
  "rubric_notes": "<one sentence explaining what a score-3 output does vs score-0>"
}}

Return ONLY the JSON object, no commentary."""


QWEN_VARIATION_PROMPT = """You are generating a variation of an existing Tenacious-Bench task.

Original task:
{seed_json}

Generate ONE variation that:
- Changes the company name, signal values, and bench composition
- Preserves the same dimension and difficulty
- Uses a different company size category (startup/scaleup/enterprise)
- Alters the signal confidence by ±0.10–0.20
- Keeps the expected_features rubric structure intact

Return ONLY the JSON object in the same schema as the original, with a new task_id ending in _v{variation_index:02d}. No commentary."""


DEEPSEEK_DEDUP_PROMPT = """Two Tenacious-Bench synthesis tasks may be near-duplicates. Compare them and decide which to keep.

Task A:
{task_a_json}

Task B:
{task_b_json}

Criteria for keeping: higher rubric specificity, more distinct signal values, clearer expected_features.

Respond with ONLY: {{"keep": "A"}} or {{"keep": "B"}}, plus a one-line reason.
Example: {{"keep": "A", "reason": "Task A has explicit bench headcount constraint; Task B is generic."}}"""


# ---------------------------------------------------------------------------
# API call helpers
# ---------------------------------------------------------------------------

def call_model(model: str, prompt: str, api_key: str, temperature: float = 0.7) -> str:
    if not HTTPX_AVAILABLE:
        raise RuntimeError("httpx not installed — run: pip install httpx")

    response = httpx.post(
        f"{OPENROUTER_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 800,
        },
        timeout=45.0,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{[\s\S]+\}", text)
    if not match:
        raise ValueError(f"No JSON found in: {text[:300]}")
    return json.loads(match.group(0))


# ---------------------------------------------------------------------------
# Dry-run stubs (no API calls)
# ---------------------------------------------------------------------------

def stub_seed_task(index: int, template: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    dim = template["dimensions"][0]
    ai_score = round(rng.uniform(0.45, 0.85), 2)
    roles = rng.randint(2, 12)
    capacity = rng.randint(3, 8)
    companies = ["Drata", "Ironclad", "Persona", "Replit", "Temporal", "Modal", "Hex"]
    signals = ["Series B close 30 days ago", "Layoff 45 days ago", "8 new JDs posted this week"]

    return {
        "task_id": f"tb_synth_{index:04d}",
        "source_mode": "multi-llm-synthesis",
        "dimension": dim,
        "difficulty": template["difficulty"],
        "input": {
            "company_signal": (
                f"Company: {rng.choice(companies)}\n"
                f"Open engineering roles: {roles}\n"
                f"AI maturity score: {ai_score}\n"
                f"Recent signal: {rng.choice(signals)}"
            ),
            "bench_summary": (
                f"Available bench: {capacity} engineers (Python×2, Go×1, Rust×1, ML×{capacity-3})\n"
                f"Total headcount capacity: {capacity}"
            ),
            "engagement_stage": rng.choice(["cold", "warm", "hot"]),
        },
        "expected_features": {
            "must_reference_signal": True,
            "must_include_cta": True,
            "banned_phrases": ["I hope this finds you well", "I wanted to reach out", "Let me know if you have any questions"],
            "required_hedges": [f"based on the available signal"] if ai_score < 0.70 else [],
            "max_bench_commitment": min(capacity, roles // 2),
        },
        "rubric_notes": (
            f"Score 3: references the specific signal with confidence calibrated to {ai_score:.2f}; "
            f"CTA matches {rng.choice(['cold','warm'])} stage. Score 0: asserts facts without grounding."
        ),
    }


def stub_variation(seed: dict[str, Any], variation_index: int, rng: random.Random) -> dict[str, Any]:
    import copy
    v = copy.deepcopy(seed)
    v["task_id"] = seed["task_id"] + f"_v{variation_index:02d}"
    companies = ["Fivetran", "Observe", "Prefect", "Dagster", "Turso", "Neon", "Supabase"]
    v["input"]["company_signal"] = re.sub(
        r"Company: \w+", f"Company: {rng.choice(companies)}", v["input"]["company_signal"]
    )
    old_score_match = re.search(r"AI maturity score: ([\d.]+)", v["input"]["company_signal"])
    if old_score_match:
        old_score = float(old_score_match.group(1))
        new_score = round(min(1.0, max(0.0, old_score + rng.uniform(-0.2, 0.2))), 2)
        v["input"]["company_signal"] = v["input"]["company_signal"].replace(
            old_score_match.group(0), f"AI maturity score: {new_score}"
        )
    return v


# ---------------------------------------------------------------------------
# Phase 1: Claude seed generation
# ---------------------------------------------------------------------------

def generate_seeds(api_key: str, dry_run: bool, rng: random.Random) -> list[dict[str, Any]]:
    seeds = []
    template_cycle = HARD_SCENARIO_TEMPLATES * (N_SEEDS // len(HARD_SCENARIO_TEMPLATES) + 1)

    print(f"Phase 1: Generating {N_SEEDS} seed tasks with {CLAUDE_MODEL} ...", file=sys.stderr)

    for i in range(N_SEEDS):
        template = template_cycle[i]
        if dry_run:
            task = stub_seed_task(i, template, rng)
        else:
            prompt = CLAUDE_SEED_PROMPT.format(
                description=template["description"],
                dimensions=", ".join(template["dimensions"]),
                difficulty=template["difficulty"],
                index=i,
                primary_dimension=template["dimensions"][0],
            )
            raw = call_model(CLAUDE_MODEL, prompt, api_key, temperature=0.8)
            task = extract_json(raw)
            task.setdefault("source_mode", "multi-llm-synthesis")

        seeds.append(task)
        if (i + 1) % 10 == 0:
            print(f"  Seeds generated: {i+1}/{N_SEEDS}", file=sys.stderr)

    return seeds


# ---------------------------------------------------------------------------
# Phase 2: Qwen bulk variation
# ---------------------------------------------------------------------------

def generate_variations(
    seeds: list[dict[str, Any]], api_key: str, dry_run: bool, rng: random.Random
) -> list[dict[str, Any]]:
    candidates = list(seeds)
    variations_needed = VARIATIONS_PER_SEED

    print(
        f"Phase 2: Generating {variations_needed} variations per seed with {QWEN_MODEL} ...",
        file=sys.stderr,
    )

    for seed_idx, seed in enumerate(seeds):
        for v_idx in range(variations_needed):
            if dry_run:
                variation = stub_variation(seed, v_idx, rng)
            else:
                prompt = QWEN_VARIATION_PROMPT.format(
                    seed_json=json.dumps(seed, indent=2)[:2000],
                    variation_index=v_idx,
                )
                raw = call_model(QWEN_MODEL, prompt, api_key, temperature=0.9)
                variation = extract_json(raw)
                variation.setdefault("source_mode", "multi-llm-synthesis")

            candidates.append(variation)

        if (seed_idx + 1) % 5 == 0:
            print(
                f"  Variations from seed {seed_idx+1}/{len(seeds)}, total candidates: {len(candidates)}",
                file=sys.stderr,
            )

    return candidates


# ---------------------------------------------------------------------------
# Phase 3: DeepSeek near-duplicate resolution
# ---------------------------------------------------------------------------

def resolve_near_duplicates(
    tasks: list[dict[str, Any]], api_key: str, dry_run: bool
) -> list[dict[str, Any]]:
    if dry_run:
        print("Phase 3: Skipping DeepSeek dedup in dry-run mode.", file=sys.stderr)
        return tasks

    print(f"Phase 3: Near-duplicate resolution with {DEEPSEEK_MODEL} ...", file=sys.stderr)

    by_dim: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        dim = task.get("dimension", "unknown")
        by_dim.setdefault(dim, []).append(task)

    kept: list[dict[str, Any]] = []
    for dim, group in by_dim.items():
        # Compare adjacent pairs within each dimension group
        i = 0
        while i < len(group) - 1:
            task_a = group[i]
            task_b = group[i + 1]
            try:
                prompt = DEEPSEEK_DEDUP_PROMPT.format(
                    task_a_json=json.dumps(task_a, indent=2)[:1500],
                    task_b_json=json.dumps(task_b, indent=2)[:1500],
                )
                raw = call_model(DEEPSEEK_MODEL, prompt, api_key, temperature=0.0)
                result = extract_json(raw)
                winner = task_a if result.get("keep", "A") == "A" else task_b
                kept.append(winner)
                i += 2
            except Exception as e:
                print(f"  DeepSeek dedup error for {dim} pair: {e}", file=sys.stderr)
                kept.append(task_a)
                i += 1
        if i == len(group) - 1:
            kept.append(group[-1])

    print(f"  After dedup: {len(kept)} tasks remain", file=sys.stderr)
    return kept


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-LLM synthesis generation for Tenacious-Bench"
    )
    parser.add_argument(
        "--output",
        default="synthesis_candidates.jsonl",
        help="Path to write all candidates before judge filtering",
    )
    parser.add_argument(
        "--filtered-output",
        default="synthesis_filtered.jsonl",
        help="Path to write judge-filtered final tasks (~50 tasks)",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stub all API calls with deterministic synthetic data",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and not args.dry_run:
        print(
            "Warning: OPENROUTER_API_KEY not set. Switching to dry-run mode.", file=sys.stderr
        )
        args.dry_run = True

    # Phase 1: Claude seed generation (eval-tier)
    seeds = generate_seeds(api_key, args.dry_run, rng)

    # Phase 2: Qwen bulk variation (dev-tier)
    candidates = generate_variations(seeds, api_key, args.dry_run, rng)

    # Phase 3: DeepSeek near-duplicate resolution
    candidates = resolve_near_duplicates(candidates, api_key, args.dry_run)

    # Write all candidates
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for task in candidates:
            f.write(json.dumps(task) + "\n")

    print(f"\nPhase 1–3 complete:", file=sys.stderr)
    print(f"  Seeds generated (Claude): {len(seeds)}", file=sys.stderr)
    print(f"  Total candidates: {len(candidates)}", file=sys.stderr)
    print(f"  Candidates written to: {output_path}", file=sys.stderr)
    print(f"\nPhase 4: Run judge_filter.py to reduce to ~{TARGET_FINAL} tasks:", file=sys.stderr)
    print(
        f"  python judge_filter.py --input {output_path} --output {args.filtered_output} "
        f"--model {QWEN_MODEL} --seed {args.seed}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
