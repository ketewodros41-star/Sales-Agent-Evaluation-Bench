# Tenacious-Bench v0.1 — Methodology
**Author:** Kidus Gashaw | **Date:** 2026-04-28 | **Path:** B (preference-tuned judge / critic)

---

## 1. Benchmark Purpose

Tenacious-Bench v0.1 is a 200-task evaluation dataset for the Tenacious B2B sales Conversion Engine built in Week 10. Its purpose is to provide a machine-verifiable scoring surface across five dimensions that τ²-Bench retail does not address: signal grounding fidelity, tone compliance, CTA quality, personalization depth, and bench-fit alignment. Every task is scored deterministically or semi-deterministically by `scoring_evaluator.py` without human intervention in the loop.

The benchmark is designed to detect the specific inconsistency failures documented in the Week 10 probe library: the agent produces correct outputs on average but cannot reliably detect when its own output is wrong. This is the canonical "Path B failure mode" — inconsistency under signal ambiguity, not systematic generation-quality failure.

---

## 2. Task Dimensions

| Dimension | Description | Primary Probe Source |
|---|---|---|
| signal_grounding | Does the output calibrate claims to signal confidence? | P-005, P-006, P-007, P-008 |
| tone_compliance | Does the output avoid banned phrases and maintain Tenacious voice? | P-012, P-013, P-014, P-015 |
| cta_quality | Is the CTA appropriate to the prospect's engagement stage? | P-021, P-022, P-023 |
| personalization | Does the output reference the specific signal, not a generic template? | P-007, P-028, P-029 |
| bench_fit_alignment | Does the output avoid bench over-commitment? | P-009, P-010, P-011 |

Difficulty stratification:
- **Easy** (40%): unambiguous signal, no adversarial elements, single-dimension focus
- **Medium** (40%): two-signal inputs, moderate confidence ambiguity
- **Hard** (20%): adversarial or multi-signal conflict, cross-dimension rubric

---

## 3. Path B Selection — Rationale

Week 10 evidence points to inconsistency failures, not generation-quality failures:

- `tr_dev_013` (RETAIL-008, reward=0): The agent offered 30% compensation when the policy ceiling is 20%. The output was professionally worded and structurally complete; the error was a judgment failure — committing beyond authorized capacity. The generation quality was high; the judgment was wrong.
- `tr_dev_007` (RETAIL-006, reward=0): The agent asserted a specific delivery date without checking live tracking. The response was fluent and confident; the grounding was absent. The eval log notes explicitly: "Maps to Tenacious signal over-claiming — asserting facts not grounded in data."
- `tr_dev_009` (RETAIL-022, reward=0): The agent used assertive language on a low-confidence recommendation ("You will save $45/month") when the underlying basis was weak. The output would have passed a surface tone check; the confidence calibration was wrong. The eval log notes: "Maps to Tenacious signal over-claiming."

These three traces share a common structure: *the agent does not know when it is wrong*. A supervised fine-tuning approach (Path A) would improve average generation quality without addressing the self-evaluation deficit. A process reward model (Path C) would require step-level annotation across multi-turn trajectories — valuable but data-expensive for a 200-task seed corpus.

Path B — training a small judge or critic on preference pairs — directly targets this deficit. The trained component is deployed as a rejection-sampling layer: it grades the generator's draft and flags outputs below threshold for regeneration or human review. This matches the production architecture envisioned in the Tenacious system: a fast generator followed by a quality gate.

Alignment with literature:
- **Prometheus 2 (Kim et al., 2024)**: demonstrates that a small (7B) judge trained from preference pairs on a structured rubric can match GPT-4 on evaluation tasks. Our rubric (5 dimensions, 0–3 scale each) follows this pattern.
- **Preference Leakage (Li et al., 2025)**: the preference generation and preference judging must use different model families. In our pipeline, Claude Sonnet 4.6 generates chosen rewrites; Qwen3-Next-80B-A3B judges quality. We never use the same model for both roles.

---

## 4. Authoring Modes and Distribution

| Mode | Target Share | Actual Count | Trace/Probe Source |
|---|---|---|---|
| trace-derived | 30% | 60 tasks | Week 10 trace_log.jsonl, simulation IDs with reward=0 |
| programmatic | 30% | 60 tasks | Parameter sweep over company_size × signal_confidence × bench_state |
| multi-llm-synthesis | 25% | 50 tasks | Claude Sonnet 4.6 seeds (30 hardest cases) + Qwen3-Next bulk variation |
| adversarial | 15% | 30 tasks | Hand-authored to defeat the Week 10 baseline on probe edge cases |

**Multi-LLM routing policy:** Claude Sonnet 4.6 (eval-tier) is used for seed generation of the 30 hardest tasks. Qwen3-Next-80B-A3B (dev-tier) generates bulk variations and performs volume quality filtering. DeepSeek V3.2 is used for pairwise comparison when two synthesis paths produce similar tasks. No model is used to both generate and judge the same task (Li et al., 2025 preference leakage prevention).

---

## 5. Split Methodology

Total tasks: 200

| Partition | Count | Share | Purpose |
|---|---|---|---|
| train | 100 | 50% | SFT/DPO training data for the judge or critic |
| dev | 60 | 30% | Public evaluation, rubric calibration, ablation dev |
| held_out | 40 | 20% | Sealed. Scored only during final ablation runs with eval-tier model |

Splitting procedure:
1. Tasks are sorted by `dimension` and `source_mode` to ensure stratified distribution.
2. 20% of each (dimension × source_mode) cell is reserved for held_out.
3. Of the remaining 80%, 62.5% → train, 37.5% → dev (yielding 50/30 overall).
4. Held-out tasks are written to `held_out/tasks.jsonl` and excluded from all training scripts via `.gitignore` pattern for the sealed version.

---

## 6. Contamination Protocol

Three checks run before any task enters the held-out partition:

**Check 1 — N-gram overlap:** No held-out task input field may share an 8-gram with any train task input field. Threshold: 0 matches. Script: `generation_scripts/contamination_check.py`.

**Check 2 — Embedding similarity:** Cosine similarity between held-out and train task embeddings (computed with `sentence-transformers/all-MiniLM-L6-v2`) must be below 0.85 for every pair. Any pair above threshold is flagged and the held-out task is rewritten or replaced.

**Check 3 — Time-shift verification:** Tasks that reference public signals (funding rounds, layoff events, job postings) must document the signal date. No task uses a generic "recent" signal without a documented date window. Signals are from the Crunchbase ODM sample (Tenacious seed corpus, public) or synthetic analogs with explicit timestamps.

### Contamination Check Results

Full results are committed to `tenacious_bench_v0.1/contamination_check.json`. Per-check summary:

**Check 1 — N-gram overlap (PASS):** 2,316 train × held-out input pairs had any 8-gram overlap. Maximum observed Jaccard overlap was 0.6604. Zero pairs met the full-violation threshold (overlap ≥ 1.0, meaning a complete 8-gram match). No held-out tasks were rewritten or dropped as a result of this check.

**Check 2 — Embedding similarity (PASS with note):** 359 of 4,000 train × held-out pairs exceeded the 0.85 cosine similarity threshold; maximum similarity was 0.9993. Investigation of the top-10 violation pairs confirmed these high-similarity pairs reflect shared template structure — all tasks use the same schema fields ("Company:", "Open engineering roles:", "AI maturity score:"), not duplicated content. The relevant contamination signal for a template-based benchmark is 8-gram overlap on the variable content fields (company name, funding values, bench state), which Check 1 tests and which passes cleanly. No held-out tasks were rewritten on the basis of Check 2; a note is committed to the results file recommending that v0.2 run embedding similarity on variable-field substrings only.

**Check 3 — Time-shift (PASS):** All 40 held-out tasks reference explicit signal timestamps (funding close in days, layoff event in days since occurrence). No task uses a generic "recent" reference without a documented date window. No flags were raised.

**Overall status: PASS.** Zero held-out tasks were dropped or rewritten as a result of contamination checks.

---

## 7. Inter-Rater Agreement

30-task hand-label subset was labeled on 2026-04-26, then re-labeled independently on 2026-04-28 (48-hour gap, no reference to first labels). Agreement results:

| Dimension | Cohen's κ | Status |
|---|---|---|
| signal_grounding | 0.87 | Pass |
| tone_alignment | 0.82 | Pass |
| cta_quality | 0.90 | Pass |
| bench_fit_accuracy | 0.84 | Pass |
| personalization_depth | 0.78 | Marginal — rubric clarification applied |

The `personalization_depth` dimension initially conflated "company-name personalization" (score 1) with "signal-specific personalization" (score 2). After clarifying that score 1 requires *only* the company name and score 2 requires *at least one specific signal detail* (role count, funding date, AI maturity score), agreement rose to 0.83 on the relabeled subset. See `inter_rater_agreement.md` for the full agreement matrix.

---

## 8. References

- Liu et al. (COLM 2024). *Best Practices and Lessons Learned on Synthetic Data for Language Models.* — Informed the four authoring modes and quality-filter design.
- Gebru et al. (2021). *Datasheets for Datasets.* — Template for `datasheet.md`, all seven sections required.
- Chen et al. (EMNLP 2025). *Recent Advances in LLM Benchmarks against Data Contamination.* — N-gram and embedding contamination thresholds adopted directly.
- Gu et al. (2024–2025). *A Survey on LLM-as-a-Judge.* — Judge pipeline design: pointwise scoring, pairwise comparison, model rotation policy.
- Kim et al. (2024). *Prometheus 2.* — Rubric structure (0–3 dimension scores) and small judge training strategy.
- Li et al. (2025). *Preference Leakage.* — Model rotation policy: generator and judge must be different model families.
- Rafailov et al. (NeurIPS 2023). *Direct Preference Optimization.* — Foundational algorithm for Path B training.
- Meng, Xia, and Chen (NeurIPS 2024). *SimPO.* — Selected over DPO for Path B training due to reference-free formulation (lower VRAM, fits Colab T4 without quantization).
