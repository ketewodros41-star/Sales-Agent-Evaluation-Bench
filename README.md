# Tenacious-Bench v0.1
**A sales-domain evaluation benchmark for B2B outreach agents**

**Status:** Week 11 Interim (Acts I–II complete) | **Author:** Kidus Gashaw | **Date:** 2026-04-28

---

## Overview

Tenacious-Bench v0.1 is a 200-task benchmark dataset that evaluates whether the Tenacious Conversion Engine performs well on Tenacious-specific B2B outreach tasks, where generic benchmarks like τ²-Bench retail fail. The benchmark targets five documented failure dimensions from the Week 10 probe library and is machine-scoreable with no human in the evaluation loop.

**Why this benchmark exists:** τ²-Bench retail scores Tenacious's agent at 0.76 mean reward — but the tasks are retail shopping tasks. The score is measuring the wrong thing. This benchmark measures signal grounding fidelity, tone compliance, CTA quality, personalization depth, and bench-fit alignment on realistic B2B outreach scenarios drawn from Week 10 failure evidence.

---

## Directory Structure

```
/
├── README.md                         ← This file
├── audit_memo.md                     ← 600-word audit: what τ²-Bench misses and why
├── schema.json                       ← Machine-verifiable task schema + 3 example tasks
├── scoring_evaluator.py              ← Deterministic 5-dimension scorer
├── datasheet.md                      ← Gebru + Pushkarna dataset documentation
├── methodology.md                    ← Path B selection, split protocol, contamination rules
├── inter_rater_agreement.md          ← 30-task relabeling, Cohen's κ by dimension
├── cost_log.md                       ← Every API and compute charge to date
├── report_draft.md                   ← PDF report content (composition, samples, plan)
├── synthesis_memos/
│   ├── synthetic_data_best_practices.md   ← Liu et al. COLM 2024 synthesis + critique
│   └── datasheets_and_data_cards.md       ← Gebru + Pushkarna synthesis + critique
├── generation_scripts/
│   ├── generate_trace_tasks.py       ← Trace-derived task authoring from trace_log.jsonl
│   ├── generate_programmatic_tasks.py← Parametric sweep task generation
│   ├── contamination_check.py        ← N-gram + embedding contamination verification
│   └── judge_filter.py               ← LLM-as-judge quality filter (3 dimensions, 1–5 each)
└── tenacious_bench_v0.1/
    ├── train/
    │   └── tasks.jsonl               ← 100 training tasks
    ├── dev/
    │   └── tasks.jsonl               ← 60 dev tasks
    ├── held_out/
    │   └── tasks.jsonl               ← 40 held-out tasks (sealed for ablation)
    └── contamination_check.json      ← Contamination report (n-gram check: PASS)
```

---

## Setup

```bash
# Python 3.11+ required
pip install sentence-transformers numpy httpx

# Extract a single task from the JSONL, then run the scorer
head -1 tenacious_bench_v0.1/train/tasks.jsonl > /tmp/sample_task.json
python scoring_evaluator.py \
  --task /tmp/sample_task.json \
  --output "We noticed Meridian is adding backend capacity — are you finding it harder to hire senior engineers? Happy to share how teams at similar stage have approached this. Would a 15-minute call this week work?" \
  --pretty

# Run contamination check
python generation_scripts/contamination_check.py \
  --train tenacious_bench_v0.1/train/tasks.jsonl \
  --dev tenacious_bench_v0.1/dev/tasks.jsonl \
  --held-out tenacious_bench_v0.1/held_out/tasks.jsonl \
  --output tenacious_bench_v0.1/contamination_check.json

# Generate trace-derived tasks (requires Week 10 trace_log.jsonl)
python generation_scripts/generate_trace_tasks.py \
  --trace-log "../trp week 10/trace_log.jsonl" \
  --output tenacious_bench_v0.1/train/tasks.jsonl \
  --seed 42

# Run judge filter on candidates
python generation_scripts/judge_filter.py \
  --input candidates.jsonl \
  --output filtered.jsonl \
  --dry-run   # omit --dry-run and set OPENROUTER_API_KEY for LLM scoring
```

---

## Benchmark Stats (Interim)

| Metric | Value |
|---|---|
| Total tasks | 200 |
| Train / Dev / Held-out | 100 / 60 / 40 |
| Dimensions | 5 (balanced, 40 tasks each) |
| Source modes | trace-derived 30%, programmatic 30%, multi-llm-synthesis 25%, adversarial 15% |
| Contamination (n-gram) | PASS (max 8-gram overlap: 0.66, 0 full violations) |
| Inter-rater agreement | κ ≥ 0.82 on all 5 dimensions (post rubric revision) |
| Scoring | Deterministic, no human in loop |
| License | CC-BY-4.0 |

---

## Interim Deliverables (Acts I–II)

| Deliverable | Status |
|---|---|
| audit_memo.md | Complete — 5 gaps, 11 probe IDs, 6 trace IDs |
| schema.json | Complete — full schema + 3 example tasks |
| scoring_evaluator.py | Complete — 5 dimensions, deterministic, CLI |
| methodology.md | Complete — Path B, split protocol, contamination, IRA |
| tenacious_bench_v0.1/ | Complete — 200 tasks, 3 partitions |
| contamination_check.json | Complete — n-gram PASS, embedding pending |
| inter_rater_agreement.md | Complete — κ ≥ 0.82 all dimensions |
| datasheet.md | Complete — all 7 Gebru sections + Pushkarna layers |
| synthesis_memos/ (2) | Complete — Liu et al. + Gebru/Pushkarna |
| cost_log.md | Complete — $4.96 spent of $10 budget |
| report_draft.md | Complete — composition table, IRA, 3 samples, plan |

---

## Plan for Days 4–7

| Day | Priority |
|---|---|
| Day 4 | Path-specific synthesis memos (SimPO, Prometheus 2, Preference Leakage) + convert train → preference pairs |
| Day 5 | SimPO training run (Qwen 3.5 2B, Unsloth Colab T4) + Delta A/B ablation on held-out |
| Day 6 | Statistical significance testing, held-out trace analysis, model card |
| Day 7 | HuggingFace publication, blog post, community engagement |

---

## Reproducibility

All generation scripts use `--seed 42` by default. The dataset was generated with `gen_dataset.py` (committed) using Python 3.11 and the random library. No external API calls were required for the dataset files themselves. The judge filter uses a heuristic fallback (`--dry-run`) when `OPENROUTER_API_KEY` is not set, making the full pipeline reproducible without API credentials.

A stranger can clone this repo, run `pip install sentence-transformers numpy`, and reproduce the contamination check and scoring in under 10 minutes.

---

## Attribution

Built on the Tenacious Conversion Engine (Week 10, TRP1 cohort). Benchmark design informed by: Liu et al. (COLM 2024), Gebru et al. (2021), Pushkarna et al. (FAccT 2022), Chen et al. (EMNLP 2025), Gu et al. (2024–2025), Kim et al. (2024), Li et al. (2025), Rafailov et al. (NeurIPS 2023), Meng et al. (NeurIPS 2024). Tenacious named as the workflow domain only; no private prospect data included.
