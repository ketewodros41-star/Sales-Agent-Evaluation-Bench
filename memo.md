# Executive Memo: Tenacious-Bench v0.1 — Findings and Deployment Recommendation
**To:** Tenacious Engineering and GTM Leadership
**From:** Kidus Gashaw, AI Evaluation Research
**Date:** 2026-04-30
**Re:** Evaluation of SimPO Preference Judge for Sales Outreach Quality Gate

---

## Page 1: Findings

### The Problem

The Tenacious AI sales agent achieves **38.7% pass@1** [34.1%, 43.3%] on the τ²-Bench retail evaluation slice. This is the actual baseline: slightly more than one in three outreach emails fully satisfies the quality rubric. The three most common failure classes are:

| Failure class | Share of failures | Representative trace |
|---|---|---|
| Bench over-commitment | 38.9% | Trace 0c380837: confirmed availability for a partially-filled bench slot |
| Assertive tone on low-confidence signal | 22.2% | Trace f50f1801: AI-implementation framing for a prospect at "researching" stage |
| Signal reference without hedging | 22.2% | Trace 0857ba6e: mentioned two signals generically, neither grounded in provided values |

The agent's failure pattern is not poor writing quality. The outputs are professionally structured and fluent. The failures are judgment errors: over-commitment, under-hedging, and weak personalization.

### What Was Built

**Tenacious-Bench v0.1** — a 200-task evaluation benchmark designed specifically for these failure modes, with four authoring methods (trace-derived, programmatic, multi-LLM synthesis, adversarial) and a five-dimension deterministic scoring evaluator (inter-rater agreement κ ≥ 0.80 on all dimensions post-calibration).

A **SimPO preference judge** was trained on top of Qwen2.5-1.5B-Instruct using **69 preference pairs** (44 original + 25 excellence-tier pairs with chosen scores 0.90–1.00) using a pure SimPO loss implemented directly in PyTorch. The judge was trained to score outreach emails against the five-dimension rubric and flag outputs that fall below the passing threshold before they reach a prospect.

### Key Results

| Metric | Value | Notes |
|---|---|---|
| Delta A (trained judge vs. corrected deterministic evaluator) | **+0.0641** lift | 95% CI: [0.022, 0.108], p < 0.0001 |
| Delta B (trained judge vs. prompt-only backbone) | **+0.3204** lift | Training made judge more generous than raw backbone (see p.2) |
| p-value (Delta A, one-tailed bootstrap) | **< 0.0001** | Significant at α=0.05: **YES** |
| p-value (Delta B, one-tailed bootstrap) | **< 0.0001** | Significant at α=0.05: **YES** |
| Win rate (Delta A, 40 held-out tasks) | **23/40 (57.5%)** | |
| Win rate (Delta B, 40 held-out tasks) | **37/40 (92.5%)** | |
| Training pairs | **69** | 44 original + 25 excellence-tier (0.90–1.00) |
| Training wall time | **~4.5 min (269 s)** | Colab T4, Qwen2.5-1.5B-Instruct |
| Final training loss | **0.52** | SimPO (pure PyTorch); final margin = 3.98 (converged) |
| Cost per held-out eval with trained judge | **~$0.0004/task** | vs. ~$0.0002/task without |

**Deployment recommendation:** The trained Qwen judge produces statistically significant separation from both the deterministic evaluator (Delta A, p < 0.0001) and the untuned backbone (Delta B, +0.32 lift, 37/40 wins, p < 0.0001). The adapter demonstrably changed the model's scoring behavior. However, Delta B being positive means the trained judge is more generous than the raw backbone — the excellence-tier training shifted the judge's internal scale upward rather than making it stricter. Deploy with a calibrated threshold (≥ 0.65 recommended) and monitor the override rate in the first 30 days before treating the judge as a hard gate.

---

## Page 2: Limitations, Failure Modes, and Kill Switch

### Four Failure Modes Not Covered in v0.1

1. **Multi-turn adversarial compliance drift** — Probes P-031 and P-015 (condescension and compliance drift under multi-turn objection pressure) are in the Week 10 probe library but not in this benchmark. The trained judge was not evaluated against outputs that appear compliant in turn 1 but drift toward policy violations by turn 3. This is a v0.2 gap.

2. **Cross-signal hedging under partial information** — All 200 tasks provide at least one usable signal token. Real prospects frequently have zero confirmed signals (cold outreach). The benchmark does not include tasks with empty signal fields, so the judge has not been calibrated on the "no signal → no assertion" case.

3. **Bench state change mid-thread** — The benchmark treats bench availability as a fixed parameter at task time. In production, a skill slot can become unavailable between the initial email and a follow-up. The judge was not trained on tasks where the bench state changes between turns.

4. **Length-quality confound** — The judge's SimPO training used length-normalized log-probability. For very short emails (under 120 words), the normalization may over-penalize brevity. The held-out partition contains tasks where a short email is the correct response (declined objection handling); whether the trained judge correctly scores these was not separately analyzed.

### Honest Assessment of Training Results

The ablation results (Delta A, Delta B) were computed using scoring_evaluator.py v2, which enforces the full Tenacious Style Guide v2 banned-phrase list (27 phrases), word-count constraints (cold ≤ 120, warm ≤ 200 words), and the "bench" word prohibition. All numbers reported reflect the corrected evaluator.

The final training used a pure SimPO loss loop implemented directly in PyTorch (no TRL dependency), avoiding earlier CPO fallback issues. Convergence was confirmed by the SimPO margin trajectory: the model started with a negative margin and converged to +3.98 at the final step, with `converged=True` in run_log.json.

**Delta A (+0.064):** The trained judge scores held-out emails marginally higher than the strict deterministic evaluator. This is a statistically significant but modest lift. It reflects better calibration: the v3 judge, having been trained on excellence-tier pairs, sees the mediocre Week 10 baseline emails for what they are and scores them close to where the deterministic evaluator places them. The large Delta A in the prior version (+0.41) was an artifact of training only on mediocre-vs-bad pairs, which produced a judge with a looser internal scale than the deterministic evaluator.

**Delta B (+0.320):** The trained judge scores emails substantially higher than the untuned backbone (37/40 wins, p<0.0001). This is the primary evidence that SimPO training changed the model's behavior. The direction — trained judge more generous than backbone — reflects the excellence-tier training: the judge has been exposed to what a 0.90–1.00 email looks like and gives more credit to emails that partially satisfy quality signals. The raw backbone, without this calibration, defaults to conservative mid-range scoring.

### Training Data Quality Fix (v3)

Post-training analysis identified a structural weakness in preference_pairs_v2.jsonl: all 44 "chosen" outputs had final scores of 0.38–0.57 (mediocre range). The judge trained exclusively on mediocre-vs-bad pairs and was never shown what excellent looks like. This explains why Delta A is large (the judge scores 0.41 higher than the corrected deterministic evaluator) but may under-penalize near-miss outputs that are merely adequate rather than poor.

**Fix:** `training_data/preference_pairs_v3.jsonl` (69 pairs = 44 original + 25 excellence-tier pairs). The 25 new pairs use:

- **Chosen** = hand-engineered ideal email scoring 0.90–1.00 on the corrected evaluator (sg=2, ta=3, cq=3, bf=3, pd=3 for most; sg=3 for tasks with required_hedges)
- **Rejected** = the former v2 "chosen" (0.35–0.57), which becomes the new rejected tier

Score gaps on the new pairs: +0.33 to +0.65 (mean ≈ +0.44), all above the δ ≥ 0.20 threshold. The training data now spans three tiers: bad (v2 rejected), mediocre (v2 chosen = v3 rejected for new pairs), and excellent (v3 chosen).

The v3 training run completed successfully (run_log.json: n_pairs=69, final_loss=0.52, margin=3.98, converged=True, elapsed=269s). The ablation on the full 40-task held-out partition is complete — results are in `ablations/ablation_results.json` and `ablations/significance.json`.

### Kill Switch Condition

The trained judge should be disabled from the pre-send quality gate and returned to manual review if:

- **Judge override rate exceeds 30%** — if the judge flags more than 30% of outputs as below threshold in a 30-day production window, the threshold is miscalibrated for the live distribution. Investigate before continuing automated flagging.
- **IRA drops below κ = 0.75** — if re-annotation of 20 judge-scored outputs shows human-judge agreement below κ = 0.75, the judge has drifted from the rubric. Retrain on updated preference pairs.
- **Any bench over-commitment passes undetected for 5+ consecutive days** — the bench_fit_alignment dimension is the highest-consequence failure class. If the judge fails to catch a confirmed over-commitment in production for 5 days, halt automated use immediately and investigate.

### Dataset and Artifacts

- Tenacious-Bench v0.1: https://huggingface.co/datasets/ketewodros41/tenacious-bench-v0.1
- SimPO LoRA adapter: https://huggingface.co/ketewodros41/qwen2.5-1.5b-simpo-tenacious-judge
- Ablation results and bootstrap significance: `ablations/ablation_results.json`, `ablations/significance.json`
- Every numeric claim in this memo traces to a task ID, a training log line, or an ablation row in `evidence_graph.json`

---

*This memo was prepared as part of the 10 Academy TRP Week 11 final submission. All numeric claims trace to ablations/ablation_results.json, ablations/significance.json, and training/qwen_simpo_judge/run_log.json.*
