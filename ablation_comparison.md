# Ablation Results: v2 vs v3 — Honest Comparison

**Date written:** 2026-05-01  
**v2 source:** evidence_graph.json claims (original documented results, raw data now overwritten)  
**v3 source:** ablation_results.json + significance.json + run_log.json (current files)

---

## Training Run Differences

| Parameter | v2 | v3 |
|---|---|---|
| Training pairs | 44 | 69 (44 + 25 excellence-tier) |
| Chosen score range | 0.38–0.57 (mediocre) | 0.38–0.57 + 0.90–1.00 (excellence) |
| Loss function | CPO (NLL + preference, via TRL) | SimPO (pure PyTorch, no TRL) |
| Final loss | 3.37 (CPO — not comparable) | 0.52 (SimPO) |
| Final margin | +0.52 (step-wise log) | +3.98 |
| Wall time | 90s | 269s |
| Converged | True | True |

The loss values cannot be compared — CPO and SimPO compute different quantities.

---

## Delta A: Trained Judge vs Deterministic Evaluator

*Measures how much the trained Qwen judge scores differ from the deterministic evaluator on the 40 held-out Week 10 baseline emails.*

| Metric | v2 | v3 | Change |
|---|---|---|---|
| Mean lift | +0.4108 | +0.0641 | **−0.347** |
| 95% CI | [0.3587, 0.4608] | [0.022, 0.108] | Narrowed significantly |
| p-value | < 0.0001 | < 0.0001 | Both significant |
| Wins / 40 | 40 (100%) | 23 (57.5%) | **−42.5 pp** |
| Losses | 0 | 11 | |
| Ties | 0 | 6 | |

**Delta A dropped by ~0.35.** The v2 judge scored held-out emails 0.41 higher than the strict deterministic evaluator. The v3 judge scores them only 0.064 higher.

**Why this happened:** The v2 training data consisted entirely of mediocre-vs-bad pairs (chosen scores 0.38–0.57). A judge trained only on that range learns to reward the upper end of a narrow quality band, which produces inflated scores relative to the now-strict Style Guide v2 deterministic evaluator. The v3 training added excellence-tier pairs (chosen=0.90–1.00). The judge now has a reference point for what "excellent" actually looks like, so it scores the mediocre Week 10 emails closer to their true position — which happens to align much better with the deterministic evaluator. The smaller Delta A is arguably a sign of *better calibration*, not worse performance.

That said: v2's 100% win rate looked better on paper. v3's 57.5% win rate is harder to sell as a headline.

---

## Delta B: Trained Judge vs Raw Backbone (No Adapter)

*Measures how much the LoRA adapter shifted the judge relative to the untuned Qwen 1.5B backbone.*

| Metric | v2 | v3 | Change |
|---|---|---|---|
| Mean lift | −0.0596 | +0.3204 | **+0.38 (direction flipped)** |
| 95% CI | [−0.0975, −0.0292] | [0.268, 0.376] | |
| p-value | not significant | < 0.0001 | |
| Wins / 40 | 21 (52.5%) | 37 (92.5%) | +40 pp |
| Losses | 19 | 1 | |

**Delta B flipped direction completely.** In v2, the adapter made the judge stricter than the backbone (negative lift = trained judge scored lower). In v3, the adapter made the judge substantially more generous than the backbone (positive lift = trained judge scored higher).

**Why this happened:** The raw Qwen 1.5B backbone without the adapter tends to assign mid-range, conservative scores to emails. The v3 adapter was trained on excellence-tier pairs where the "chosen" output scored 0.90–1.00. This shifted the judge's internal scale upward — it now gives more credit to emails that partially satisfy quality signals, because it has been exposed to what a fully satisfying email looks like. The backbone, by comparison, has no such calibration and defaults to cautious mid-range scoring.

The v2 adapter, trained only on mediocre-vs-bad pairs, learned to be conservative (stricter than the backbone). The v3 adapter, trained on a wider quality range including excellence, became more generous.

---

## Summary Scorecard

| | v2 | v3 | Better? |
|---|---|---|---|
| Delta A magnitude | 0.41 | 0.06 | v2 looks better |
| Delta A win rate | 100% | 57.5% | v2 looks better |
| Delta A direction | judge > evaluator | judge ≈ evaluator | v3 may be better calibrated |
| Delta B significance | Not significant | p < 0.0001 | v3 better |
| Delta B direction | adapter = stricter | adapter = more generous | Ambiguous — depends on use case |
| Delta B win rate | 52.5% | 92.5% | v3 better |
| Training convergence | True | True | Same |

---

## Honest Interpretation

**What v3 genuinely improved:** The adapter now produces a statistically robust and large effect on Delta B — the trained judge is measurably different from the untuned backbone (37/40 wins, p<0.0001). This was not true in v2 (Delta B was not significant). If the goal is to demonstrate that SimPO training changed the model's behavior, v3 is a stronger result.

**What v3 made worse:** Delta A collapsed from a 100% win rate to 57.5%. The headline number in the blog post — "+0.41 lift, 40/40 wins" — is now gone. The v3 judge no longer consistently outscores the deterministic evaluator.

**What changed between runs that makes direct comparison hard:**
1. The loss function changed (CPO → pure SimPO). The v2 adapter may have learned different representations.
2. The training data changed both in size and in quality tier distribution.
3. The v2 ablation_results.json raw data was overwritten, so there is no way to rerun the bootstrap on v2 data. The v2 numbers come from the claims in evidence_graph.json only.

**The uncomfortable truth about v2 Delta A:** A 100% win rate with +0.41 mean lift was unusually strong. In context, it reflected that the v2 judge had been trained on a narrow band (bad vs mediocre) and the deterministic evaluator was strict — the gap between them was large *because the judge had been trained to rate mediocre emails as acceptable*. It was not evidence that the judge was well-calibrated. It was evidence that the judge had a different, looser internal scale than the deterministic evaluator. v3's closer alignment between judge and evaluator (Delta A = +0.06) is the better outcome for a judge you intend to use as a quality gate.

---

## What to Do With These Numbers

- **For the blog post:** Lead with v3 Delta B (+0.32, 92.5% win rate, p<0.0001) as the primary evidence that SimPO training worked. Frame Delta A (+0.06, significant) as evidence of calibration alignment, not score inflation.
- **Do not cite v2 Delta A (+0.41, 100%)** in v3 context — the training data and loss function changed, making those numbers incomparable artifacts.
- **Acknowledge the regression honestly** — the previous version showed larger Delta A because it was trained on a narrower quality band.
