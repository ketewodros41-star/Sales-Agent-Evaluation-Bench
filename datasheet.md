# Datasheet for Tenacious-Bench v0.1
**Following:** Gebru et al. (2021) *Datasheets for Datasets* + Pushkarna et al. (FAccT 2022) *Data Cards* (telescopic / periscopic / microscopic layers)

**Dataset version:** 0.1 | **Date:** 2026-04-28 | **License:** CC-BY-4.0

---

## Telescopic Summary (one paragraph)

Tenacious-Bench v0.1 is a 200-task evaluation benchmark for B2B sales outreach agents operating in the Tenacious staffing-and-consulting domain. It measures five failure dimensions — signal grounding fidelity, tone compliance, CTA quality, personalization depth, and bench-fit alignment — that are undetectable by general-purpose benchmarks such as τ²-Bench retail. Tasks are drawn from four authoring modes: trace-derived (30%), programmatic (30%), multi-LLM synthesis (25%), and hand-authored adversarial (15%). The dataset is partitioned into train (100), dev (60), and held-out (40) splits, with contamination-free held-out verified by n-gram overlap, embedding similarity, and time-shift checks. Every task is machine-scoreable by the accompanying `scoring_evaluator.py` with no human in the loop.

---

## 1. Motivation

### Why was this dataset created?
Tenacious-Bench was created because no existing public benchmark evaluates B2B sales outreach agents on the specific failure modes documented in the Tenacious Conversion Engine's Week 10 probe library. τ²-Bench retail (the prior evaluation infrastructure) uses retail shopping tasks with binary ground truth and cannot grade signal calibration, tone adherence to a documented style guide, or bench-inventory-constrained capacity claims.

### Who funded the creation?
This dataset was created as part of the TRP1 program (10 Academy). No external funding was applied. Total authoring cost: approximately $4.20 in dev-tier OpenRouter API calls (Qwen3-Next-80B-A3B). Eval-tier scoring cost (held-out slice): approximately $1.80.

### Who will benefit from this dataset?
- Teams deploying sales outreach agents who need to measure domain-specific failure modes beyond general benchmarks.
- Researchers building preference-tuned judges for constrained-domain agents.
- The open evaluation community studying LLM-as-a-judge calibration on structured rubrics.

---

## 2. Composition

### Periscopic Detail

| Attribute | Value |
|---|---|
| Total tasks | 200 |
| Train partition | 100 tasks (50%) |
| Dev partition | 60 tasks (30%) |
| Held-out partition | 40 tasks (20%) |
| Dimensions | 5 |
| Difficulty levels | 3 (easy / medium / hard) |
| Source modes | 4 |

### Source mode distribution

| Mode | Count | Share |
|---|---|---|
| trace-derived | 60 | 30% |
| programmatic | 60 | 30% |
| multi-LLM synthesis | 50 | 25% |
| adversarial (hand-authored) | 30 | 15% |

### Dimension distribution

| Dimension | Count |
|---|---|
| signal_grounding | 50 |
| tone_compliance | 40 |
| cta_quality | 40 |
| personalization | 35 |
| bench_fit_alignment | 35 |

### Difficulty distribution

| Difficulty | Count |
|---|---|
| easy | 80 |
| medium | 80 |
| hard | 40 |

### What does each task contain?
Each task is a JSON object with: `task_id`, `dimension`, `difficulty`, `source_mode`, `input` (company_signal, bench_summary, prior_thread), `expected_features` (must_reference_signal, must_include_cta, banned_phrases, optional required_hedges, optional max_headcount_commitment), and `rubric` (five 0–3 dimension scores).

### Does the dataset contain all possible instances, or is it a sample?
A sample. The full space of Tenacious prospect scenarios is much larger. This benchmark covers the highest-ROI failure modes from the Week 10 probe library (P-005 through P-035) and is designed to be extended with new probes as the Conversion Engine evolves.

### Is there a label or target associated with each instance?
Each task includes a rubric that specifies expected scoring criteria. Trace-derived tasks include a `gold_output` field. Generative tasks use the rubric as ground truth; the scorer returns a numerical score.

### Are there recommended data splits?
Yes. The three partitions (train/dev/held_out) are the recommended splits. Do not train on dev or held_out. The held_out partition is sealed; its contents are not exposed in training scripts.

---

## 3. Collection Process

### Microscopic Detail — Authoring Modes

**Trace-derived (60 tasks):**
Source: Week 10 `trace_log.jsonl`, specifically the 28 simulation trials with `reward=0.0` (failed trials). Each failed trace was reviewed, the failure mode classified against the probe taxonomy, and the input reconstructed into a task format. Signal data was replaced with synthetic analogs to remove any private prospect information. Three traces provided the seed for 60 tasks by varying signal confidence, bench state, and prior thread.

Key trace sources: `simulation_id=0c380837` (task_id=104, bench over-commitment), `simulation_id=f50f1801` (task_id=105, AI maturity over-claim), `simulation_id=ac397276` (task_id=22, tone drift under pricing pressure).

**Programmatic (60 tasks):**
A parameter sweep over: company_size (seed/A/B), signal_confidence (low/medium/high), bench_state (full/partial/empty for required skill), prior_thread (cold/warm/hostile). Each combination produces a template-filled task. 60 tasks drawn from the cross-product (out of ~120 total combinations), filtered by judge score ≥ 3/5 on coherence.

**Multi-LLM synthesis (50 tasks):**
Seed generation: Claude Sonnet 4.6 prompted with Week 10 failure taxonomy + 3 example tasks, generating 30 hard-case seeds. Bulk variation: Qwen3-Next-80B-A3B generated 40 variations per seed (1,200 candidates), deduplicated by 8-gram overlap, judge-filtered (threshold: coherence ≥ 3, verifiability ≥ 3, rubric_clarity ≥ 3). 50 tasks passed all filters.

Preference-leakage prevention: Claude generated task seeds; Qwen judged them. No model both generated and judged the same task.

**Adversarial (30 tasks):**
Hand-authored by the dataset creator to specifically trigger the Week 10 baseline's weakest failure modes: bench over-commitment with partial-skill bench (10 tasks), hostile in-house-objection tone scenarios (10 tasks), AI-maturity confidence conflation (10 tasks). These carry the highest originality weight and were not passed through the LLM quality filter (hand-authored tasks are author-validated).

### Who collected the data?
Kidus Gashaw (TRP1 trainee, 10 Academy). Adversarial tasks are author-original. Trace-derived tasks are reconstructed from Week 10 outputs.

### Over what time-frame?
2026-04-26 through 2026-04-28 (Days 2–3 of Week 11).

### Were any ethical review processes conducted?
The dataset contains no personal information. Prospect signals are synthetic analogs of public firmographic data (Crunchbase ODM sample). No real company names, contact names, or email content are present.

---

## 4. Preprocessing / Cleaning / Labeling

### Was any preprocessing done?
- Trace data: company names replaced with synthetic analogs. Signal values (funding amount, employee count) replaced with plausible but non-identifying values within the same tier.
- Programmatic data: template-filled tasks passed a coherence filter (Qwen3-Next judge, score ≥ 3/5).
- Multi-LLM synthesis: 8-gram deduplication, then embedding similarity filter (cosine < 0.85), then three-dimension judge filter.

### Was the data labeled?
Rubric criteria are embedded in each task's `expected_features` and `rubric` fields. A 30-task subset was hand-labeled twice (inter-rater agreement protocol). See `inter_rater_agreement.md` for results.

### Is the software used for preprocessing available?
Yes. `generation_scripts/generate_trace_tasks.py`, `generate_programmatic_tasks.py`, `contamination_check.py`, and `judge_filter.py` are all committed to the repository.

---

## 5. Uses

### Has the dataset been used for any tasks already?
It is used in Week 11 as the training data and evaluation benchmark for a Path B judge trained on Tenacious-specific rubric dimensions.

### What tasks could the dataset be used for?
- Evaluating cold-email generation agents on Tenacious-style B2B sales tasks.
- Training preference judges for signal-constrained outreach domains.
- Studying the effect of signal confidence on generation quality in agents with access to firmographic data.
- Benchmark comparison: assessing whether a generic LLM scores as well as a domain-fine-tuned judge on structured rubrics.

### What tasks should the dataset NOT be used for?
- Evaluating general-purpose assistants (it is domain-specific and rubric-dependent).
- Tasks involving real prospect data (the dataset contains no real contact or company information).
- As a sole basis for production deployment decisions without validation on real prospect feedback.

---

## Limitations and Known Biases

### Failure Risks

**Evaluator gaming:** `scoring_evaluator.py` uses deterministic pattern matching — banned-phrase lists, CTA patterns, hedge patterns. A model that memorizes the exact pattern lists can score highly without internalizing the underlying style principles. This is a construct validity risk for `tone_compliance` and `cta_quality`: high scores may reflect pattern avoidance rather than genuine compliance.

**Bench-state resolution gap:** 15% of tasks involve partial bench state (two of three required skill slots committed). No tasks cover dynamic bench-state changes mid-thread (e.g., a slot that opens during a conversation). Agents or judges trained on Tenacious-Bench v0.1 may underperform on real-time inventory scenarios.

**Length–quality confound:** The `personalization_depth` dimension checks for at least two distinct signal references but does not penalize length inflation. An agent that repeats the same signal token twice scores identically to one that references two genuinely distinct signals. This confound is most acute for emails under 120 words.

**Signal-confidence conflation:** Tasks assume internally consistent prospect signals. Real prospects produce contradictory signals (e.g., AI maturity score of 3 but active implementation language in the prior thread). The benchmark does not cover signal-conflict resolution, so a judge trained here may be miscalibrated for contradiction scenarios.

### Population Skews

**Domain specificity:** All 200 tasks are B2B outreach for an AI talent staffing platform. Zero coverage of other sales domains, other agent modalities (voice, in-app chat), or non-technical staffing categories (executive search, ops roles). Benchmark performance does not generalize across domains.

**Company segment imbalance:** Programmatic tasks cover `company_size ∈ {seed, Series A, Series B}`. No tasks represent enterprise accounts (Series C+), public companies, or bootstrapped SMBs, which involve different tone registers and bench-commitment thresholds.

**Cold outreach absent:** Every task supplies a `bench_summary` and an optional `prior_thread`. Cold outreach — where no prior engagement or bench context exists — is unrepresented. Agent performance on cold outreach is untested by this benchmark.

**Single-turn only:** All tasks are single-turn email generation. Multi-turn conversation sequences (follow-up, objection handling, re-engagement) are not covered in v0.1. Benchmark scores do not predict multi-turn dialogue quality.

**Synthetic signal distribution:** All company signals are synthetic analogs drawn from a Crunchbase ODM sample. The signal distribution (headcount ranges, funding tiers, AI maturity scores) reflects the ODM sample, not the actual Tenacious prospect population. Calibration of trained models may not match the live prospect distribution.

### Misuse Scenarios

**Reporting as a general email-quality benchmark:** High scores on Tenacious-Bench do not imply general email quality. Reporting benchmark performance without domain qualification — "our agent scores 0.78 on Tenacious-Bench" as a proxy for general sales quality — misrepresents scope.

**Training or fine-tuning on the held-out partition:** The 40 held-out tasks are sealed for evaluation only. Including them in any training or preference-pair construction pipeline invalidates the ablation comparisons and inflates reported performance.

**Deploying the trained judge without production validation:** The SimPO judge adapter was trained on 69 rubric-derived preference pairs from 200 synthetic tasks. Deploying it as a production quality gate without validation on real Tenacious prospect interactions and re-annotation by domain experts risks systematic misclassification of edge cases the synthetic distribution does not cover.

**Interpreting per-dimension scores as independent:** The five dimensions are correlated in practice (a signal-grounded email tends to have higher personalization depth). Optimizing individual dimension scores in isolation — e.g., training only on `tone_compliance` pairs — may produce dimension-level gains that do not translate to overall rubric compliance.

---

## 6. Distribution

**License:** CC-BY-4.0. Attribution required; commercial use permitted.

**Access:** Will be published on HuggingFace Hub under the dataset creator's handle. Held-out partition will remain sealed until the Week 11 leaderboard is published (target: 2026-05-05).

**Is the dataset self-contained or does it link to external resources?**
Self-contained. No external URLs are required to use the benchmark. Signal data is embedded in each task's `company_signal` field.

**Will the dataset be updated?**
Planned: Tenacious-Bench v0.2 (target Week 13) will add trace-derived tasks from the trained agent's held-out inference runs and extend adversarial coverage to multi-turn trajectories.

---

## 7. Maintenance

**Who is maintaining this dataset?**
Kidus Gashaw (kidus@10academy.org) for the duration of TRP1. Long-term maintenance: community-maintained via the HuggingFace dataset repository.

**How can errors be reported?**
Via GitHub Issues in the dataset repository. Include: task_id, observed error, and proposed correction.

**Will older versions be supported or deprecated?**
v0.1 will remain available as a versioned snapshot on HuggingFace. Future versions will not delete v0.1 data.

**Contamination safeguards:**
The contamination check script (`generation_scripts/contamination_check.py`) must be re-run before any additions to the held-out partition. The contamination report (`tenacious_bench_v0.1/contamination_check.json`) must be updated with each dataset version.
