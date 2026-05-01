# Methodology Rationale: Path B — SimPO Preference Judge
**Author:** Kidus Gashaw | **Date:** 2026-04-30

---

## Why Path B

The choice to train a preference judge (Path B) rather than a supervised fine-tuned generator (Path A) or a process reward model (Path C) came from a specific pattern in the Week 10 failure data, not from a general preference for the paradigm.

### The Evidence Behind the Decision

The Week 10 agent runs produced a 38.7% pass@1 on τ²-Bench's retail slice [34.1%, 43.3%]. That number is not the full story. Three failed traces drove the path selection:

**Trace 0c380837** (task_id 104, cost $0.044, duration 219s): The agent produced a fluent, well-structured email that over-committed on bench capacity. The prospect asked directly whether a senior engineer was available the following week. The agent confirmed availability without hedging. The bench was partially filled in the task context — two of three required skill slots were committed, leaving availability contingent. The output read like a correct answer. The scoring evaluator flagged it as a bench_fit_alignment failure (score 0 on that dimension). The agent had no mechanism to recognize that its output looked correct but violated a capacity constraint.

**Trace f50f1801** (task_id 105, cost $0.023, duration 144s): The agent produced an outreach email that conflated AI maturity signal with AI readiness. The prospect's context indicated a "researching AI vendors" stage. The agent wrote copy positioned for a prospect who had already committed to implementation. The tone_compliance dimension scored 0 (assertive language on low-confidence qualification). Again: fluent, well-structured, plausibly professional. Wrong judgment about what the signal warranted.

**Trace 0857ba6e** (task_id 76, cost $0.036, duration 229s): Personalization failure. The agent referenced two generic signal tokens ("recently visited pricing page," "attended webinar") without grounding them in the specific signal values provided. The signal_grounding dimension scored 0. The output contained 312 words. Zero of them were specific to the prospect's actual signals.

### The Common Pattern

All three failures share the same structure: the output is well-formed by surface criteria and wrong by rubric criteria. The agent produces text that a human skimming for quality would pass. The evaluator, applying the rubric mechanically, catches the violation.

This pattern — good average form, inconsistent judgment — rules out Path A. Supervised fine-tuning would train the agent to produce more fluent emails. The problem is not fluency. Path C requires step-level correctness annotations across trajectories; the failures here are single-output judgment errors, not multi-step reasoning failures. Path B directly addresses the gap: train a component that can evaluate whether a generated output satisfies the rubric before it is sent, so the judgment failures can be caught and revised.

---

## Why SimPO Over DPO

### The Compute Constraint

Training a DPO judge requires running two models simultaneously: the policy being trained and a frozen reference model. On Colab T4 (15GB VRAM), running two forward passes of Qwen 3.5 2B at standard batch sizes exceeds memory capacity. The practical ceiling for T4 with two models is approximately 1.5B parameters per model at bf16 — below the 2B target. SimPO eliminates the reference model: one forward pass, full 2B parameters, within budget.

This is not a theoretical argument. DPO was implemented first in the training script design and abandoned after memory profiling showed the combined forward pass would require gradient checkpointing aggressive enough to make training time exceed the 30-minute wall target.

### The Data Constraint

DPO's reward formulation depends on the reference model having seen the training distribution — the reference log-probability term is only informative if the reference model can assign meaningful probability to the training examples. For Tenacious-Bench training pairs, the reference model is Qwen 3.5 2B (the same backbone as the judge). It has not been exposed to the Tenacious domain during pretraining — it has no prior for what a correctly-grounded B2B sales email looks like. The reference log-probability term would be approximately uniform across all training pairs, contributing noise rather than signal to the reward. SimPO avoids this problem by using length-normalized log-probability instead, which captures the model's own fluency signal rather than comparing to an uninformed reference.

### ORPO Was Considered and Rejected

ORPO modifies the generator's generation loss rather than training a separate judge. The deployment architecture requires a judgment component that runs after the generator, not a modified generator. ORPO would improve average output quality on the training distribution; it would not produce a component that can evaluate novel outputs at inference time. The deployment requirement — a quality gate the agent can query before sending an email — requires a separate judge, not a modified generator.

---

## Split Protocol Rationale

The 50/30/20 split (train/dev/held-out) was chosen to maximize training data while preserving a meaningful held-out evaluation.

The held-out size of 40 tasks was set at the minimum sufficient for statistical power in the ablation: a paired bootstrap test across 40 binary outcomes has approximately 80% power to detect a 15-percentage-point improvement at α=0.05. Below 30 tasks, the test is underpowered for realistic improvement magnitudes. Above 50 tasks, the held-out partition would require drawing from the train partition, which has higher value as training data for a 80-pair preference dataset.

Stratification was done within each (dimension × source_mode) cell — the 20 cells formed by 5 dimensions × 4 source modes. This ensures every failure dimension and every authoring mode is represented in the held-out evaluation, so no dimension is accidentally absent from the ablation result.

---

## Contamination Protocol Rationale

### Why 8-Gram Threshold

The 8-gram threshold follows Chen et al.'s (EMNLP 2025) finding that 8-gram overlap is the minimum granularity at which overlap reliably signals memorization rather than coincidental shared phrasing. For email-domain text, shorter n-grams (4–6) frequently appear in both training and held-out tasks simply because professional email language reuses common phrases ("I wanted to reach out," "based on your recent activity"). The 8-gram threshold suppresses these false positives while catching genuine duplication.

### Why Embedding Similarity Was Scoped to Variable Fields

The full-text embedding similarity check on Tenacious-Bench tasks produced 359 flagged pairs — a number that required individual review. Investigation showed that 357 of 359 flagged pairs were attributed to shared template structure: all tasks share the same JSON schema, the same section headers, and the same instruction preamble. Embedding the full task text made these structural similarities dominate the cosine distance. Re-running the check on variable-content fields only (company_name, prospect_name, signal_values, task_instruction) produced 2 flagged pairs, both confirmed as near-duplicates and resolved before the held-out partition was sealed.

The lesson is general: template-based benchmarks require scoping embedding similarity checks to variable fields, not full task text. The Chen et al. threshold of cosine < 0.85 was adopted as-is; the scope adjustment was improvised from the false-positive rate.

---

## Preference Pair Construction Rationale

### Score Gap Threshold δ ≥ 0.20

The 0.20 threshold on normalized score gap was chosen to produce preference pairs where the quality difference is detectable by the scoring evaluator but not trivially large. Pairs with δ < 0.10 are too close for SimPO's margin γ=0.5 to create a useful gradient — the model is being asked to distinguish outputs that are nearly equivalent. Pairs with δ > 0.50 are likely to be dimension-zero failures (output contains a banned phrase or forced booking CTA) where the failure is so obvious that even the Week 10 baseline would avoid it with minor prompting, making the pair low-information for training.

The δ ≥ 0.20 threshold targets the middle range: outputs where the Tenacious agent currently makes wrong calls and where a trained judge would need genuine evaluation capacity to distinguish correct from incorrect.

### Signal Grounding Exception

If the valid pair count falls below 80 after applying δ ≥ 0.20 globally, the threshold for signal_grounding pairs only will be lowered to δ ≥ 0.15. The rationale: signal_grounding failures in the Week 10 data are consistently in the 0.12–0.22 gap range — the agent hedges sometimes but not consistently, producing outputs that are nearly correct. These near-misses are the most informative training signal for a judge (distinguishing "mentions a signal" from "correctly hedges a weak signal") but they fall below the global threshold. The exception is documented in `training_data/build_pairs.py` and reported in the training log.

---

## Papers That Grounded Each Design Decision

| Decision | Grounding Paper |
|---|---|
| Cross-family model routing (Claude generates, Qwen judges) | Li et al. 2025 (Preference Leakage) — cross-family selection eliminates score inflation |
| SimPO over DPO | Meng et al. NeurIPS 2024 (SimPO) — reference-free, memory-efficient, competitive performance |
| γ set from score distribution not grid search | Meng et al. NeurIPS 2024 — grid search requires large dev set; δ distribution is the reliable signal |
| LLM judge confined to dataset construction | Gu et al. 2024–2025 (LLM-as-Judge Survey) — deterministic rubrics don't benefit from LLM judgment |
| 8-gram contamination threshold | Chen et al. EMNLP 2025 — 8-gram minimum for reliable memorization signal |
| Pointwise over pairwise for quality filter | Gu et al. 2024–2025 — per-dimension independence means pairwise collapses signal |
| Rubric as ground truth (no model-generated gold standard) | Li et al. 2025 — task-definition leakage via model-generated gold standards |
| Small judge model (7B class) is sufficient | Kim et al. 2024 (Prometheus 2) — 7B judge matches GPT-4 on rubric correlation |
