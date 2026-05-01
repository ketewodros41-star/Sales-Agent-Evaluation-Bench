# Week 11 Progress Update
**Author:** Kidus Gashaw | **Date:** 2026-04-30 | **Day:** 4 of 7 | **Path:** B (SimPO preference judge)

---

## Where Things Stand

Acts I and II are complete. The benchmark dataset is built, validated, and committed. The interim report is submitted. Days 4–7 shift from dataset construction to training and evaluation.

---

## Completed

### Acts I–II: Dataset Construction

| Artifact | Status | Key Detail |
|---|---|---|
| `audit_memo.md` | ✅ Done | 5 gaps, 11 probe IDs, 6 trace IDs — all tied to specific τ²-Bench blind spots |
| `schema.json` | ✅ Done | Machine-verifiable task schema + 3 example tasks |
| `scoring_evaluator.py` | ✅ Done | 5-dimension deterministic scorer, 0–3 per dimension, weighted final score |
| `tenacious_bench_v0.1/` | ✅ Done | 200 tasks across train (100) / dev (60) / held-out (40) |
| `contamination_check.json` | ✅ Done | N-gram PASS (max 8-gram overlap 0.66, zero full violations) |
| `inter_rater_agreement.md` | ✅ Done | κ ≥ 0.80 on all 5 dimensions post-revision |
| `datasheet.md` | ✅ Done | All 7 Gebru sections + Pushkarna telescopic/periscopic/microscopic layers |
| `methodology.md` | ✅ Done | Path B rationale, 50/30/20 split, contamination results per check |
| `synthesis_memos/` (2) | ✅ Done | Liu et al. COLM 2024 + Gebru/Pushkarna — both with specific disagreements grounded in own evidence |
| `report_draft.md` | ✅ Done | Cross-tab composition table, IRA with rubric revisions, 3 scored worked examples, candid status + path-specific plan |
| `cost_log.md` | ✅ Done | $4.96 spent of $10 budget |

### Source Mode Distribution (actual vs. target)

| Mode | Target | Actual |
|---|---|---|
| trace-derived | 30% (60) | 30% (60) ✅ |
| programmatic | 30% (60) | 30% (60) ✅ |
| multi-LLM synthesis | 25% (50) | 25% (50) ✅ |
| adversarial | 15% (30) | 15% (30) ✅ |

### Inter-Rater Agreement Summary

| Dimension | κ | Result |
|---|---|---|
| cta_quality | 0.853 | Pass — first pass |
| signal_grounding | 0.857 | Pass — first pass |
| bench_fit_accuracy | 0.794 → pass | Boundary clarification applied |
| tone_alignment | 0.790 → pass | Boundary clarification applied |
| personalization_depth | 0.682 → **0.830** | Full rubric revision + re-labeling |

### Budget Consumed

$4.96 of $10.00 (49.6%). Remaining $5.04 is reserved for Days 4–7 operations.

---

## In Progress (Day 4)

- **Preference pair construction** — generating "chosen" outputs from Claude Sonnet 4.6 for 100 train tasks; pairing against Week 10 baseline outputs as "rejected" where score gap δ ≥ 0.20. Target: ≥ 80 valid pairs for SimPO training.
- **Path B synthesis memos** — SimPO (Meng et al.), Prometheus 2 (Kim et al.), Preference Leakage (Li et al.) memos in progress.
- **Embedding contamination re-check** — re-running on variable-field substrings only (the full-field check produced 359 false positives from shared template structure).

---

## Known Blockers and Risks

**1. Preference pair gap (δ) may be too small on signal_grounding tasks.**
The Week 10 baseline partially succeeds on signal_grounding (it mentions signals, it just doesn't hedge them correctly). The score difference between a correct and incorrect output may be only 0.10–0.15 on this dimension — below the δ ≥ 0.20 threshold needed for SimPO to train on a clear preference signal. If fewer than 80 valid pairs result, the training run is under-resourced.

**Mitigation:** If the valid pair count falls below 80, lower δ to 0.15 for the signal_grounding dimension only (where baseline quality is higher) and document the exception.

**2. Multi-turn adversarial coverage is absent.**
The 30 adversarial tasks cover single-turn failures. Probes P-031 and P-015 (condescension and compliance drift under multi-turn pressure) are in the Week 10 probe library but not in the benchmark. The trained judge will not be evaluated against this failure class in the held-out run.

**Mitigation:** Flagged as a v0.2 gap. No action in this week's scope.

**3. Budget buffer is $0.24 after reservations.**
Three Day 5–6 eval-tier calls are budgeted ($1.28 each × 3 = $3.84). Any unplanned held-out inference call exhausts the buffer. The held-out partition will be scored exactly once on Day 5 and sealed immediately.

---

## Remaining Days

| Day | Focus | Key Output |
|---|---|---|
| Day 4 | Preference pairs + path-specific memos | `training_data/preference_pairs.jsonl`, 3 memos |
| Day 5 AM | SimPO training run — Qwen 3.5 2B, Unsloth, Colab T4, ≤30 min | Loss curve, trained adapter |
| Day 5 PM | Delta A/B ablation on held-out (40 tasks) | `ablation_results.json` |
| Day 6 | Paired bootstrap significance, held-out trace analysis, eval-tier calibration, model card | Final stats, `model_card.md` |
| Day 7 | HuggingFace publish (dataset + adapter), blog post, community submission | Public URLs |

**Kill criterion for Day 5:** If training loss has not dropped below 0.65 within the first 10 minutes, halt and diagnose. Flat loss → preference pairs too uniform (increase δ to 0.30). Slow decrease → reduce learning rate from 5e-5 to 2e-5 and extend to 45 minutes. Oscillating loss → increase gradient accumulation from 8 to 16.

---

## What I'm Working On Today (Day 4 — 2026-04-30)

Today is about getting ready for tomorrow's training run. The dataset is done — today I shift to building the training data and finishing the required reading.

- **Building preference pairs** — taking the 100 training tasks, running the existing agent on them to get baseline outputs, then generating better "correct" versions. Each pair (good output vs. baseline output) is what the judge model will learn from. I need at least 80 clean pairs before the training run can start tomorrow.

- **Reading and writing three paper memos** — the path-specific papers I need to engage with before training: SimPO (the algorithm I'm using), Prometheus 2 (how small judge models are trained), and Preference Leakage (how to make sure the training setup doesn't introduce bias). Each memo needs to include a genuine disagreement with something in the paper, backed by what I've observed building this benchmark.

- **Cleaning up the contamination check** — the embedding similarity check produced a lot of false positives because all tasks share the same template structure. Re-running it on just the variable content fields (company name, signal values) to get a more meaningful result.

Everything today feeds into Day 5. If the preference pairs aren't ready by end of day, the training run slips.

---

## How the Dataset Was Generated

The dataset was built using four distinct authoring modes, each producing a structurally different kind of task. The goal was diversity — not just in surface wording, but in the type of evidence and reasoning each task encodes.

### Four Authoring Modes

**Trace-derived (30% of tasks)**
Starting point: the failed trials from the Week 10 agent runs — sessions where the agent produced an output that scored zero. Each failure was reviewed, classified against the five failure dimensions (signal grounding, tone compliance, CTA quality, personalization, bench-fit alignment), and reconstructed as a benchmark task. Real company names and prospect details were replaced with synthetic analogs so no private data entered the dataset. Three failed traces were enough to seed sixty tasks by varying the signal confidence, bench state, and prior thread across each source case.

**Programmatic (30% of tasks)**
A parameter sweep across four variables: company size (seed / Series A / Series B), signal confidence (low / medium / high), bench availability (full / partial / empty for a required skill), and prior thread warmth (cold / warm / hostile). Each combination produced a template-filled task. Roughly half the possible combinations were used, filtered by a coherence check to remove nonsensical parameter pairings. This mode produces tasks that systematically stress the boundaries of the rubric rather than just reflecting what the agent happened to fail on.

**Multi-LLM synthesis (25% of tasks)**
This mode used a two-model pipeline to generate harder cases that no real trace or parameter sweep naturally produced. Claude Sonnet 4.6 generated thirty seed tasks covering cross-dimension scenarios — situations where the agent must satisfy two rubric dimensions simultaneously under conflicting pressure. Qwen then generated roughly forty variations per seed, producing around twelve hundred candidates. Those candidates were passed through a three-dimension judge filter (input coherence, ground-truth verifiability, rubric clarity) with a strict acceptance threshold. Around four percent of candidates passed. DeepSeek was used for pairwise comparison when two synthesis paths produced near-identical tasks, keeping the higher-quality version.

The model rotation was deliberate: Claude generated, Qwen judged. Using the same model for both generation and evaluation inflates apparent quality because the judge shares the generator's stylistic defaults. Cross-family judging catches failures that a same-family judge would score as acceptable.

**Adversarial (15% of tasks)**
Hand-authored tasks designed specifically to trigger the failure modes the Week 10 baseline handles worst: bench over-commitment when a prospect asks directly, tone violations under hostile objection pressure, and AI-maturity confidence conflation. These tasks were not passed through the automated judge filter — LLM quality filters tend to reject adversarial tasks because they look incoherent to a model that does not understand the specific failure being probed. They were validated by hand instead.

### Quality Filtering and Contamination Control

Every synthesis and programmatic candidate passed through `judge_filter.py` before entering the dataset. The filter scored each task on three dimensions and rejected anything below threshold on any single dimension. The adversarial tasks bypassed the filter and were author-validated.

Before the held-out partition was sealed, three contamination checks ran against all train tasks: an eight-gram overlap check (no held-out task input may share an eight-gram with any train input), an embedding similarity check (flagged pairs were investigated and attributed to shared template structure rather than duplicated content), and a time-shift check (every held-out task references an explicit signal date, not a generic "recent" reference). All three checks passed.

### Splitting

Tasks were split into train (50%), dev (30%), and held-out (20%) by stratifying within each dimension-by-source-mode cell. This ensures every failure dimension and every authoring mode is represented in all three partitions — the held-out evaluation is not accidentally dominated by one type of task.

---

## Act III Path Selection: Path B — SimPO Preference Judge

**Chosen path:** B (preference-tuned judge / critic)

### The Decision

The choice came from observing a consistent pattern in Week 10 failures: the agent produces fluent, well-structured outputs on straightforward inputs but makes wrong judgment calls on harder ones — over-committing on capacity, asserting claims without sufficient grounding, and miscalibrating confidence. Crucially, it cannot tell the difference between the two cases. The outputs look similar on the surface; the failures are in the judgment, not the writing.

That pattern — good average quality, inconsistent judgment — points to Path B. The agent does not need to write better; it needs a layer that can evaluate whether what it wrote is actually correct.

### Why Not Path A

Supervised fine-tuning addresses average generation quality. The agent's generation quality is not the bottleneck — the outputs are generally well-formed and professional. What fails is whether the agent knows when to hedge, when to hold back a commitment, and when a signal is too weak to assert. SFT on more examples would not teach the agent to self-evaluate; it would just produce more fluent versions of the same judgment errors.

### Why Not Path C

A process reward model is designed for step-by-step reasoning chains where the failure occurs mid-trajectory. The Tenacious failures are largely single-output judgment calls — one email, one response, one commitment. Annotating step-level correctness across every turn of a trajectory is expensive and architecturally more complex than the problem requires.

### Why Path B Fits

Training a small judge on preference pairs directly addresses the self-evaluation gap. The trained component sits as a quality gate after the generator: it scores the draft and flags outputs that fall below threshold for revision or human review. This matches how Tenacious would actually deploy the improvement in production — not replacing the generator, but adding a check on its output.

SimPO was chosen over DPO because it does not require a reference model, which keeps the training within free compute constraints. ORPO was considered but rejected because it modifies the generator itself rather than training a separate judge layer, which conflicts with the add-on architecture the deployment requires.

---

## Repository

**GitHub:** https://github.com/ketewodros41-star/Sales-Agent-Evaluation-Bench
**Branch:** `main` | **Commits:** 2 | **Last push:** 2026-04-30
