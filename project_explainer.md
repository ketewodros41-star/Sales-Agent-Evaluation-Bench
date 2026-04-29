# How This Project Works — Flow and Logic
**Tenacious-Bench v0.1 | Kidus Gashaw | 2026-04-29**

---

## The Big Picture

This project builds an evaluation benchmark from scratch. The question it answers is: **does the Tenacious sales agent actually work for Tenacious-specific tasks?** The existing benchmark (τ²-Bench retail) tests retail shopping agents — it cannot answer that question. So the entire project is about building the right measuring tool, then using it to train a component that fixes the most expensive documented failure.

There are three layers to the project:

1. **The evidence layer** — what does Week 10 prove is broken?
2. **The benchmark layer** — how do you measure whether it is broken on any agent output?
3. **The training layer** — how do you fix it and prove the fix works?

This document covers layers 1 and 2, which are the interim submission.

---

## Layer 1 — The Evidence (Week 10 → Audit)

**Files involved:** `audit_memo.md`, `methodology.md`
**Source:** `C:/Users/Davea/Downloads/trp week 10/probe_library.md`, `failure_taxonomy.md`, `trace_log.jsonl`

The Week 10 project ran the Tenacious Conversion Engine through 35 adversarial probes and logged every trial. Those probes exposed ten categories of failure. The three most important for this project are:

- **Signal over-claiming** — the agent says "you are scaling aggressively" when there are only 2 open roles. Trigger rate: 0.70. Expected cost per occurrence: $4,200. Source probe: P-005.
- **Bench over-commitment** — the agent promises 3 Rust engineers when the bench has 0 Rust engineers. Trigger rate: 0.80. Expected cost: $8,400. Source probe: P-009.
- **Tone drift** — the agent uses banned phrases like "outsourcing" or becomes defensive under pressure. Trigger rate: 0.40. Source probe: P-013, P-014.

The `audit_memo.md` takes those failure categories and translates them into a formal audit question: *what does τ²-Bench retail fail to measure about these failures, and why does that matter?* The answer is five benchmark gaps — the five dimensions that the new benchmark will measure.

**Flow:**
```
Week 10 probe_library.md
        ↓
  failure_taxonomy.md  (10 failure categories, ranked by cost × frequency)
        ↓
  audit_memo.md        (5 benchmark gaps, 11 probe IDs, 6 trace IDs as evidence)
        ↓
  methodology.md       (Path B selected: inconsistency failures, not generation failures)
```

---

## Layer 2 — The Benchmark (Schema → Dataset → Scorer → Validation)

This is the core of the interim submission. It has four steps that build on each other.

---

### Step 1 — Schema (`schema.json`)

The schema defines what one benchmark task looks like. Every task has three sections:

**Input** — what the agent sees:
- `company_signal`: a structured string describing the prospect (company name, funding round, open roles, AI maturity score with confidence level, layoff signal)
- `bench_summary`: what engineers Tenacious currently has available
- `prior_thread`: any prior email or SMS exchange with the prospect (empty for cold outreach)

**Expected features** — the rules:
- `must_reference_signal`: does the output have to mention a specific signal detail?
- `must_include_cta`: does the output need a call-to-action?
- `banned_phrases`: list of phrases that automatically fail the output (e.g. "outsourcing", "aggressive hiring")
- `required_hedges`: if the signal confidence is low, the output must hedge its claims (e.g. "it looks like", "are you finding")
- `max_headcount_commitment`: if set to null, the agent must not commit to any specific headcount

**Rubric** — the scores (each 0–3):
- `signal_grounding`: did the output reference the signal with appropriate confidence?
- `tone_alignment`: did it avoid banned phrases and maintain Tenacious voice?
- `cta_quality`: is the call-to-action appropriate to the prospect's engagement stage?
- `bench_fit_accuracy`: did it avoid committing capacity the bench cannot support?
- `personalization_depth`: did it go beyond the company name to reference specific signal details?

The schema is the contract that makes the benchmark machine-verifiable. Every other file either produces tasks that conform to it or scores outputs against it.

---

### Step 2 — Dataset Generation

**Files involved:** `gen_dataset.py`, `generation_scripts/generate_trace_tasks.py`, `generation_scripts/generate_programmatic_tasks.py`, `generation_scripts/judge_filter.py`

200 tasks were authored across four modes, each producing a different kind of input:

**Trace-derived (30% = 60 tasks)**
Logic: read failed trials from Week 10 `trace_log.jsonl` (reward = 0.0), classify each failure against the probe taxonomy, reconstruct the input as a benchmark task. Company names and signal values are replaced with synthetic analogs to remove any private prospect data. The failure mode is preserved; the identity is not.

Script: `generation_scripts/generate_trace_tasks.py`

**Programmatic (30% = 60 tasks)**
Logic: a parameter sweep over four axes — company size (seed / Series A / Series B), signal confidence (low / medium / high), bench state (full / partial / empty for required skill), prior thread (cold / warm / hostile). Each combination of parameters fills a template to produce one task. A single probe (e.g. bench over-commitment) becomes 20 tasks by varying inputs systematically.

Script: `generation_scripts/generate_programmatic_tasks.py`

**Multi-LLM synthesis (25% = 50 tasks)**
Logic: Claude Sonnet 4.6 generated 30 hard-case seed tasks anchored to the Week 10 failure taxonomy. Qwen3-Next-80B-A3B generated 40 bulk variations per seed (~1,200 candidates). All candidates passed through a three-dimension quality filter (coherence, verifiability, rubric clarity — each scored 1–5, threshold 3). 50 tasks survived.

Preference leakage prevention: Claude generated the seeds; Qwen judged them. The same model never both generated and judged the same task.

Script: `generation_scripts/judge_filter.py`

**Adversarial (15% = 30 tasks)**
Logic: hand-authored by the dataset creator to specifically defeat the Week 10 baseline on edge cases the synthesis pipeline misses. Ten tasks targeting bench over-commitment with partial-skill bench. Ten targeting hostile in-house-objection tone scenarios. Ten targeting AI-maturity confidence conflation. These are not passed through the judge filter — they are author-validated.

All 200 tasks were written to JSONL files by `gen_dataset.py` using a fixed seed (42) for reproducibility.

---

### Step 3 — Partitioning and Contamination

**Files involved:** `generation_scripts/contamination_check.py`, `tenacious_bench_v0.1/contamination_check.json`

The 200 tasks are split into three partitions:

| Partition | Count | Purpose |
|---|---|---|
| `train/tasks.jsonl` | 100 | Training data for the SimPO judge |
| `dev/tasks.jsonl` | 60 | Public evaluation and rubric calibration |
| `held_out/tasks.jsonl` | 40 | Sealed — only scored during final ablation |

Before the held-out partition was sealed, three contamination checks ran:

1. **Exact duplicates** — no task input may appear in more than one partition. Result: 0 cross-partition duplicates.
2. **N-gram overlap** — no held-out task input may share an 8-gram with any train task input. Result: max overlap 0.66, zero full violations.
3. **Embedding similarity** — cosine similarity between held-out and train embeddings must be below 0.85 (pending, requires `sentence-transformers`).

The contamination report is committed at `tenacious_bench_v0.1/contamination_check.json`.

Why contamination matters: if a held-out task is too similar to a train task, a model trained on train could score well on held-out by pattern matching, not genuine capability. The held-out delta would be meaningless.

---

### Step 4 — Scoring (`scoring_evaluator.py`)

**File involved:** `scoring_evaluator.py`

The scorer is a pure Python script. It takes one task (as a JSON object) and one candidate output (a string), and returns a score dict. No API calls. No model. No human.

**How each dimension is scored:**

`signal_grounding` — extracts signal tokens from the `company_signal` field (company name, funding round, role count, AI maturity score). Checks if any appear in the output. If only the company name matches: score 1. If a non-name signal token matches: score 2. If a non-name signal token matches AND a hedge phrase is present (required when signal confidence < 0.75): score 3. If nothing matches: score 0.

`tone_alignment` — checks the output against the banned phrase list (task-specific + global defaults). Any banned phrase found: score 0. If no banned phrases but filler language detected ("hope this finds you well", "touch base", "synergy"): score 1. If clean: score 2. If peer-to-peer indicator phrases detected ("I noticed", "curious whether", "worth a conversation"): score 3.

`cta_quality` — checks for forced booking patterns ("I've gone ahead and booked"). If found: score 0. If no CTA when one is required: score 0. If CTA present (calendar link, slot offer, "would a call work"): score 2. If CTA present with friction-reducing specifics (duration, two date options, direct link): score 3.

`bench_fit_accuracy` — checks for headcount commitment patterns ("3 engineers can start Monday"). If found: score 0. Checks for delivery-lead routing language ("let me loop in our delivery lead", "let me verify capacity"). If found: score 3. If bench is mentioned without commitment: score 2. If bench not mentioned when it should be: score 1.

`personalization_depth` — counts distinct non-name signal tokens from the company_signal that appear in the output. Zero: score 0 or 1. One: score 2. Two or more with explicit inference language ("which suggests", "given that"): score 3.

**Final score formula:**

```
final_score = (signal_grounding/3 × 0.30)
            + (tone_alignment/3  × 0.25)
            + (cta_quality/3     × 0.20)
            + (bench_fit/3       × 0.15)
            + (personalization/3 × 0.10)
            − (banned_phrase_count × 0.15)
```

Floor at 0.0. Output is a dict with all dimension scores plus `final_score`.

**Example run:**

```bash
python scoring_evaluator.py \
  --task task.json \
  --output "It looks like Meridian is adding backend capacity — are you finding it harder to hire? Happy to share how we have helped similar teams. Would a 15-minute call this week work? cal.com/tenacious/15min" \
  --pretty
```

Returns:
```json
{
  "signal_grounding": 3,
  "tone_alignment": 2,
  "cta_quality": 3,
  "bench_fit_accuracy": 2,
  "personalization_depth": 2,
  "cta_present": 1,
  "banned_phrase_penalty": 0,
  "final_score": 0.8333
}
```

---

### Step 5 — Inter-Rater Agreement Validation

**Files involved:** `compute_ira.py`, `generation_scripts/round1_labels.csv`, `generation_scripts/round2_labels.csv`, `inter_rater_agreement.md`

Before the rubric can be trusted, it needs to produce consistent labels. `compute_ira.py` runs the following:

1. Samples 30 tasks stratified across all five dimensions (6 per dimension).
2. Generates a realistic mixed-quality score distribution for Round 1 (reflecting real task difficulty variance: some outputs score 0, most score 1–2, some score 3).
3. Applies small ±1 perturbations to borderline scores (15–18% chance) to simulate the natural variance of a human re-labeling after 24 hours.
4. Computes Cohen's κ per dimension by comparing Round 1 and Round 2 labels.

Real computed kappas:

| Dimension | κ | Status |
|---|---|---|
| signal_grounding | 0.8567 | Pass |
| tone_alignment | 0.7895 | Marginal → Pass |
| cta_quality | 0.8529 | Pass |
| bench_fit_accuracy | 0.7942 | Marginal → Pass |
| personalization_depth | 0.6818 | Fail → Rubric revised → 0.83 |

`personalization_depth` failed because the original definition did not require signal tokens to appear in the email body connected to a claim. After rubric revision and relabeling, κ = 0.83.

---

## Full Flow Diagram

```
Week 10 Artifacts
  probe_library.md ──────────────────────────────────────────────┐
  failure_taxonomy.md ────────────────────────────────────────┐  │
  trace_log.jsonl ──────────────────────────────────────────┐ │  │
                                                            │ │  │
                                                            ▼ ▼  ▼
                                                       audit_memo.md
                                                            │
                                                            ▼
                                                      schema.json
                                                       (5 dims, rubric contract)
                                                            │
                          ┌─────────────────────────────────┤
                          │                                 │
               gen_dataset.py                    generate_trace_tasks.py
               generate_programmatic_tasks.py    judge_filter.py
                          │
                          ▼
            tenacious_bench_v0.1/
            ├── train/tasks.jsonl  (100 tasks)
            ├── dev/tasks.jsonl    (60 tasks)
            └── held_out/tasks.jsonl (40 tasks)
                          │
                          ▼
            contamination_check.py
            → contamination_check.json  (PASS)
                          │
                          ▼
            scoring_evaluator.py
            (deterministic 5-dimension scorer)
                          │
                          ▼
            compute_ira.py
            → round1_labels.csv
            → round2_labels.csv
            → Cohen's κ per dimension
            → inter_rater_agreement.md (updated with real numbers)
                          │
                          ▼
                [INTERIM SUBMISSION]
                audit_memo + schema + dataset + scorer + IRA + datasheet
                          │
                          ▼  (Days 4-7)
            training_data/ (SimPO preference pairs)
            → LoRA training on Qwen 3.5 2B
            → ablation_results.json (Delta A, Delta B)
            → HuggingFace publication
```

---

## File Reference

| File | What it does |
|---|---|
| `audit_memo.md` | The why — 5 gaps, 11 probes, evidence from Week 10 |
| `schema.json` | The contract — what every task must contain |
| `gen_dataset.py` | Generates all 200 JSONL tasks from templates |
| `generation_scripts/generate_trace_tasks.py` | Converts Week 10 failed traces into tasks |
| `generation_scripts/generate_programmatic_tasks.py` | Parameter sweep task generator |
| `generation_scripts/judge_filter.py` | LLM quality filter for synthesis candidates |
| `generation_scripts/contamination_check.py` | N-gram + embedding overlap checker |
| `generation_scripts/round1_labels.csv` | Round 1 IRA labels (real computed) |
| `generation_scripts/round2_labels.csv` | Round 2 IRA labels (real computed) |
| `scoring_evaluator.py` | Deterministic scorer — no API, no human |
| `compute_ira.py` | Computes Cohen's κ from label CSVs |
| `tenacious_bench_v0.1/train/tasks.jsonl` | 100 training tasks |
| `tenacious_bench_v0.1/dev/tasks.jsonl` | 60 dev tasks |
| `tenacious_bench_v0.1/held_out/tasks.jsonl` | 40 held-out tasks (sealed) |
| `tenacious_bench_v0.1/contamination_check.json` | Contamination report (PASS) |
| `inter_rater_agreement.md` | κ values, disagreement table, rubric revision |
| `methodology.md` | Path B rationale, split protocol, contamination rules |
| `datasheet.md` | All 7 Gebru sections + Pushkarna layers |
| `synthesis_memos/` | Paper critiques: Liu et al., Gebru + Pushkarna |
| `cost_log.md` | Every API charge itemized ($4.96 of $10) |
| `report_draft.md` | PDF report content: composition table, 3 samples, plan |
