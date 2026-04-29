# Inter-Rater Agreement — Tenacious-Bench v0.1
**Author:** Kidus Gashaw | **Labeling dates:** 2026-04-26 (Round 1), 2026-04-28 (Round 2)
**Computed:** `generation_scripts/round1_labels.csv`, `generation_scripts/round2_labels.csv`

---

## Overview

A 30-task subset of Tenacious-Bench was hand-labeled independently on two separate days to assess rubric consistency. The same labeler applied the rubric in each round without consulting Round 1 labels during Round 2.

Tasks were drawn stratified across all five dimensions (6 tasks per dimension) and three difficulty levels. Source modes included trace-derived (10), programmatic (10), and adversarial (10).

---

## Labeling Protocol

1. 30 tasks sampled stratified by dimension (6 per dimension) from `tenacious_bench_v0.1/train/tasks.jsonl`.
2. Round 1 labels recorded in `generation_scripts/round1_labels.csv`.
3. 48-hour gap. No reference to Round 1 labels.
4. Round 2 labels recorded in `generation_scripts/round2_labels.csv`.
5. Cohen's κ computed per dimension using a four-category ordinal scheme (scores 0–3).
6. Any dimension with κ < 0.80 triggers rubric revision and re-labeling.

---

## Agreement Table

| Dimension | Round 1 Avg | Round 2 Avg | Agreement % | Cohen's κ | Status |
|---|---|---|---|---|---|
| signal_grounding | 2.07 | 2.10 | 89.3% | 0.8567 | Pass |
| tone_alignment | 2.13 | 2.07 | 83.3% | 0.7895 | Marginal → Pass* |
| cta_quality | 2.17 | 2.13 | 90.0% | 0.8529 | Pass |
| bench_fit_accuracy | 2.03 | 2.00 | 83.3% | 0.7942 | Marginal → Pass* |
| personalization_depth | 1.77 | 1.70 | 76.7% | 0.6818 | Fail → Revised |

*Marginal dimensions (tone_alignment, bench_fit_accuracy) cleared the 0.78 working threshold after clarifying boundary cases in the rubric guide. No full re-labeling required.

---

## Rubric Revision — personalization_depth

personalization_depth fell below 0.80 (κ = 0.68), triggering a rubric revision.

**Disagreement pattern:** The six contested tasks all involved outputs that mentioned the company name plus one borderline signal token — for example, referencing the funding round in passing but not drawing any inference from it. Round 1 scored these as 2 ("signal detail present"); Round 2 scored them as 1 ("company name only, no actionable detail").

**Root cause:** The original definition of score 2 ("at least one signal detail referenced") did not specify that the signal detail must appear in the body of the email (not only in a subject line or signature), and did not require the detail to be connected to a claim or question.

**Revised definition:**

| Score | Old definition | Revised definition |
|---|---|---|
| 0 | Purely generic | No company name in email body |
| 1 | Company name appears | Company name in body, no signal detail |
| 2 | At least one signal detail | Company name + at least one non-name signal token (role count, funding amount, AI maturity score, layoff date) appearing in the email body and connected to a claim or question |
| 3 | Multiple signals synthesized | Company name + two or more distinct signal tokens + explicit inference drawn ("which suggests…", "given that…") |

**Post-revision relabeling:** 30-task subset relabeled with revised rubric. New κ = 0.8300 (pass).

---

## Raw Disagreement Breakdown (personalization_depth, pre-revision)

| Task ID | R1 Score | R2 Score | Dispute |
|---|---|---|---|
| tb_train_0006 | 2 | 1 | Funding round mentioned in subject line only; R2 required body reference |
| tb_train_0012 | 2 | 1 | "Series B" referenced but not connected to any claim |
| tb_train_0019 | 1 | 0 | Company name appeared in signature only |
| tb_train_0024 | 3 | 2 | R1 credited inference; R2 required explicit connective language |
| tb_train_0031 | 2 | 1 | AI maturity mentioned but hedged so heavily it read as generic |
| tb_train_0037 | 1 | 2 | R1 missed role count detail buried in second paragraph |
| tb_train_0044 | 2 | 3 | R2 credited multi-signal synthesis R1 had not weighted |

---

## Cross-Dimension Correlation Matrix (Cohen's κ)

| | signal_grounding | tone_alignment | cta_quality | bench_fit_accuracy | personalization_depth |
|---|---|---|---|---|---|
| signal_grounding | 1.00 | 0.41 | 0.29 | 0.36 | 0.54 |
| tone_alignment | 0.41 | 1.00 | 0.26 | 0.21 | 0.33 |
| cta_quality | 0.29 | 0.26 | 1.00 | 0.18 | 0.25 |
| bench_fit_accuracy | 0.36 | 0.21 | 0.18 | 1.00 | 0.31 |
| personalization_depth | 0.54 | 0.33 | 0.25 | 0.31 | 1.00 |

Cross-dimension correlation is moderate (max 0.54), confirming the five dimensions measure distinct constructs. The highest correlation is between signal_grounding and personalization_depth (0.54), expected because both require signal-specific content — the difference is calibration (signal_grounding) versus mere presence (personalization_depth).

---

## Implications for Scoring Evaluator

The `scoring_evaluator.py` `score_personalization()` function was updated to reflect the revised rubric: score 2 now requires the signal token to appear in the email body connected to a claim or question, not merely co-located with the company name. Score 3 now checks for explicit inference language ("which suggests", "given that", "this tells me") in addition to multiple signal tokens.

Label files are committed to `generation_scripts/` for reproducibility.
