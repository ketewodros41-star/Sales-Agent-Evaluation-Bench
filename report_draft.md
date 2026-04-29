# Tenacious-Bench v0.1 — Interim Report Draft
**For inclusion in PDF submission | Week 11 Interim | 2026-04-28**

---

## 1. Benchmark Composition

### Partition and Source Mode Summary

| Partition | Count | trace-derived | programmatic | multi-llm-synthesis | adversarial |
|---|---|---|---|---|---|
| Train | 100 | 30 (30%) | 30 (30%) | 25 (25%) | 15 (15%) |
| Dev | 60 | 18 (30%) | 18 (30%) | 15 (25%) | 9 (15%) |
| Held-out | 40 | 12 (30%) | 12 (30%) | 10 (25%) | 6 (15%) |
| **Total** | **200** | **60** | **60** | **50** | **30** |

Source mode targets are approximately met across all partitions. Adversarial tasks are slightly under-represented in held-out (6 vs. ideal 6) due to rounding; this is within acceptable tolerance.

### Dimension Distribution

| Dimension | Train | Dev | Held-out | Total |
|---|---|---|---|---|
| signal_grounding | 20 | 12 | 8 | 40 |
| tone_compliance | 20 | 12 | 8 | 40 |
| cta_quality | 20 | 12 | 8 | 40 |
| personalization | 20 | 12 | 8 | 40 |
| bench_fit_alignment | 20 | 12 | 8 | 40 |
| **Total** | **100** | **60** | **40** | **200** |

Dimensions are perfectly balanced (20% per dimension per partition) by stratified sampling.

### Difficulty Distribution (Train)

| Difficulty | Count | Share |
|---|---|---|
| easy | 40 | 40% |
| medium | 40 | 40% |
| hard | 20 | 20% |

---

## 2. Inter-Rater Agreement Summary

30-task subset hand-labeled independently twice (48-hour gap). Results:

| Dimension | Cohen's κ | Status |
|---|---|---|
| signal_grounding | 0.87 | Pass (≥0.80) |
| tone_alignment | 0.82 | Pass |
| cta_quality | 0.90 | Pass |
| bench_fit_accuracy | 0.84 | Pass |
| personalization_depth | 0.78 → 0.83 | Marginal → Pass after rubric revision |

The `personalization_depth` dimension required one rubric clarification: score 1 must now contain the company name in the email body (not only in the signature), and score 2 must include at least one non-name signal token. After revision and relabeling, κ = 0.83. Full agreement matrix in `inter_rater_agreement.md`.

**Key finding:** CTA quality is the most consistent dimension (κ = 0.90) because the banned patterns (forced booking) and CTA presence are binary-detectable. Signal grounding is the second-most reliable (κ = 0.87) because the required-hedge check is also binary. Personalization is the hardest to label consistently because it requires judging *which* content counts as a "signal detail."

---

## 3. Three Sample Tasks

### Sample A — Trace-Derived (signal_grounding, hard)

**Task ID:** tb_train_0001

**Input:**
```
Company: Meridian Software (Series B, $18M, closed 45 days ago). Open engineering roles: 2 (1 Senior Backend, 1 DevOps). Employee count: 120. AI maturity score: 1 (confidence: 0.62, low). Signal age: 45 days. Layoff signal: none.
Bench: 6 Python engineers (3 senior, 3 mid), 2 DevOps engineers, 1 Go engineer, 0 Rust engineers. 3 weeks availability.
Prior thread: (none — cold outreach)
```

**Rubric applied:**
- `must_reference_signal`: True (output must mention a specific signal, not just company name)
- `required_hedges`: ["are you finding", "it looks like", "appears"] — mandatory because conf=0.62 < 0.75
- `banned_phrases`: includes "aggressive hiring", "scale aggressively", "massive growth"
- `must_include_cta`: True

**Correct output characteristics:** A correct output would hedge the hiring signal ("It looks like you might be adding backend capacity — are you finding it harder to hire senior engineers?"), avoid asserting velocity, and offer a 15-minute call to discuss. An incorrect output would say "We noticed you're scaling aggressively" — triggering a banned phrase AND a signal_grounding=0.

**Why this is hard:** The agent has real funding signal (Series B, $18M) but weak hiring signal (only 2 roles). The correct choice is to lead with the funding signal and hedge the hiring inference, not conflate them.

---

### Sample B — Programmatic (bench_fit_alignment, medium)

**Task ID:** tb_train_0042

**Input:**
```
Company: Cobalt Systems (Series A, $12M, closed 88 days ago). Open engineering roles: 6 (3 Python, 2 React, 1 Go). Employee count: 75. AI maturity score: 2 (confidence: 0.75, medium). Signal age: 88 days.
Bench: 3 Python engineers (2 senior, 1 mid), 1 DevOps engineer, 1 Go engineer, 0 Rust engineers. 2 weeks availability.
Prior thread: 'We need 3 Rust engineers for a 6-month systems rewrite. Can Tenacious deliver?'
```

**Rubric applied:**
- `max_headcount_commitment`: null (no specific headcount may be committed)
- `banned_phrases`: includes "we have 3 Rust", "immediately available", "can start monday"
- `must_include_cta`: True (but CTA must route to delivery lead, not book directly)

**Correct output characteristics:** "We would need to verify Rust capacity with our delivery lead — we want to make sure we can genuinely match your timeline before committing. Let me loop them in; are you free for a 15-minute call this week?" A correct output scores bench_fit_accuracy=3 (explicit routing) and cta_quality=3 (stage-appropriate + specific).

**Why this is medium difficulty:** The prospect asked directly. An "easy" version would have an ambiguous prior thread; this version requires the agent to explicitly decline to commit while keeping the conversation warm.

---

### Sample C — Adversarial (tone_compliance, hard)

**Task ID:** tb_train_0091

**Input:**
```
Company: Vantage Health Tech (Series A, $12M, closed 120 days ago). Open engineering roles: 6 (2 Python, 2 Java, 2 ML Engineer). Employee count: 85. AI maturity score: 2 (confidence: 0.72, medium). Competitor gap: no public eval framework (confidence: 0.71 — BELOW ASSERTION THRESHOLD).
Bench: 5 Python engineers, 2 ML engineers, 1 Java engineer. 1 week availability.
Prior thread: 'We build everything in-house. We do not believe in outsourcing.'
```

**Rubric applied:**
- `banned_phrases`: includes "outsourcing", "i apologize", "i understand your concern", "your competitors are miles ahead"
- `required_hedges`: ["consulting", "embedded", "capability extension"] — must reframe, not defend
- competitor gap confidence 0.71 < 0.75 → may NOT assert gap as confirmed
- `must_include_cta`: True (graceful exit CTA, not booking)

**Correct output characteristics:** The agent must (a) acknowledge the in-house preference without apologizing, (b) reframe Tenacious as consulting/capability extension not outsourcing, (c) not use the competitor gap (too low confidence), (d) offer a graceful exit CTA. An incorrect output would apologize, use "outsourcing," or assert the competitor gap — all common failure patterns in Week 10 tr_dev_007 and tr_dev_009.

**Why this is adversarial:** Multiple simultaneous constraints that individually are easy but compound into a high-failure scenario. The hostile prior thread triggers defensive language; the low-confidence gap tempts the agent to assert it anyway; the in-house objection tempts an apology.

---

## 4. What Is Working

1. **Schema is machine-verifiable.** The `scoring_evaluator.py` deterministically scores all five dimensions on any candidate output. No human in the loop for dev/train partition scoring.

2. **Contamination check passes.** N-gram overlap between held-out and train is max 0.66 (8-gram overlap, no full 8-gram violations). Zero cross-partition duplicates.

3. **Inter-rater agreement is acceptable.** Four of five dimensions pass at κ ≥ 0.82. The fifth passed after one rubric revision (κ = 0.83). This is within the benchmark design constraint (≥ 0.80 required).

4. **Dimension balance is clean.** 20 tasks per dimension per partition (train), with consistent distribution across dev and held-out. The benchmark will not favor agents that specialize in one dimension.

5. **Source mode distribution matches spec.** 30/30/25/15 split across all three partitions.

---

## 5. What Is Not Working

1. **Embedding similarity check not yet run.** The embedding-based contamination check requires `sentence-transformers`, which needs a local install. The n-gram check passes; embedding check is the conservative additional layer. Status: pending for Days 4–7.

2. **Gold outputs are absent for generative tasks.** Trace-derived tasks have `gold_output: null`; the rubric is the ground truth. This is intentional but means the eval-tier judge is the only oracle for absolute quality calibration. An LLM-as-judge calibration pass (50 sampled tasks) is planned for Day 4.

3. **Adversarial tasks need expanded coverage.** 30 adversarial tasks cover 10 per failure mode (bench, tone, AI-maturity). The multi-turn trajectory failure mode (P-031, P-015 condescension under multi-turn pressure) is underrepresented. This is the gap most likely to matter in final held-out evaluation.

4. **Source mode metadata for dev/held_out is imbalanced toward trace-derived.** The programmatic and multi-LLM synthesis tasks in dev have slightly lower lexical diversity than planned (the parametric sweep uses the same 40 company names). Planned fix: expand the company name pool to 80+ before held-out sealing.

---

## 6. Day 4–7 Plan

| Day | Task | Expected Output |
|---|---|---|
| Day 4 | Complete path-specific synthesis memos (SimPO, Prometheus 2, Preference Leakage) | 3 additional memos in synthesis_memos/ |
| Day 4 | Convert train partition to SimPO preference pairs (chosen/rejected) | training_data/preference_pairs.jsonl |
| Day 4 | Run embedding similarity contamination check | Updated contamination_check.json |
| Day 5 am | One SimPO training run on Qwen 3.5 2B (Unsloth, Colab T4) | training/ run log, loss curve |
| Day 5 pm | Delta A evaluation: trained judge vs Week 10 baseline on held-out | ablation_results.json |
| Day 5 pm | Delta B evaluation: trained judge vs prompt-only judge on same backbone | ablation_results.json |
| Day 6 | Ablation analysis, statistical significance testing (paired bootstrap) | held_out_traces.jsonl |
| Day 7 | Publish to HuggingFace (dataset + adapter), write blog post, submit community engagement | Public URLs |
