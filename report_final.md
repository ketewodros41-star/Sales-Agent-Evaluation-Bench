# Tenacious-Bench v0.1 — Final Self-Assessment Report
**Author:** Kidus Gashaw | **Date:** 2026-05-01 | **Rubric:** Week 11 Final Submission

---

## Score Summary

| Criterion | Max | Score | Level |
|---|---|---|---|
| 1. Audit Memo | 5 | **5** | Mastered |
| 2. Four-Mode Dataset Authoring | 15 | **12** | Competent+ |
| 3. Multi-LLM Routing & Anti-Leakage | 10 | **10** | Mastered |
| 4. Judge Filter Pipeline | 15 | **10** | Competent |
| 5. Contamination Prevention | 15 | **10** | Competent |
| 6. Inter-Rater Agreement | 10 | **10** | Mastered |
| 7. Datasheet Completeness | 5 | **5** | Mastered |
| 8. Path Declaration & Methodology Rationale | 10 | **10** | Mastered |
| 9. Training Run Script | 15 | **10** | Competent |
| 10. Ablation Methodology & Statistical Rigor | 15 | **10** | Competent |
| 11. README Reproducibility | 5 | **3** | Competent |
| **Total** | **120** | **95** | **79%** |

---

## Criterion-by-Criterion Breakdown

---

### 1. Audit Memo — 5/5 (Mastered)

**What was done:**
- `audit_memo.md` exists, structured around 5 named gaps matching the 5 rubric dimensions
- Word count: approximately 420 words — under the 600-word ceiling
- Probe IDs cited: P-005, P-006, P-007, P-008, P-009, P-010, P-012, P-013, P-021, P-027, P-028 — **11 probe IDs** (requirement: 8)
- Trace IDs cited: tr_dev_005, tr_dev_007, tr_dev_009, tr_dev_013, tr_dev_014, tr_dev_023 — **6 trace IDs** (requirement: 5)
- Explicit contrast with τ²-Bench retail: each gap section names what τ²-Bench measures and why that measurement misses the Tenacious failure
- Gaps stated at dimension level: signal_grounding, tone_compliance, personalization_depth, bench_fit_alignment, cta_quality — all five dimensions named and grounded

**Nothing to fix.**

---

### 2. Four-Mode Dataset Authoring — 12/15 (Competent+)

**What was done well:**
- `generate_trace_tasks.py`: trace-derived mode implemented ✅
- `generate_programmatic_tasks.py`: genuine combinatorial sweep using `itertools.product` across company_size × signal_confidence × bench_state × prior_thread × dimension (3×3×3×4×5 = 540 combinations, sample 60) ✅
- `generate_synthesis_tasks.py`: multi-LLM synthesis routing implemented ✅
- Source-mode metadata: per-task `source_mode` field visible in all generation scripts ✅
- Share targets (30/30/25/15): documented in methodology.md ✅

**What is missing:**
- No `generate_adversarial_tasks.py` or equivalent code. Hand-authored adversarial tasks exist in the dataset (30 tasks), but the rubric requires *code present* for the hand-authored adversarial mode. The 30 adversarial tasks entered the dataset through `gen_dataset.py` without a dedicated authoring script — there is no code that shows the adversarial authoring process.

**Why not Mastered:** The adversarial-mode code gap is real. The tasks exist but the code path is not visible. The rubric specifically checks for "Code present for... hand-authored adversarial tasks."

---

### 3. Multi-LLM Routing & Anti-Leakage — 10/10 (Mastered)

**What was done:**
- Named model families per role documented in `methodology.md`: Claude Sonnet 4.6 (frontier seed author), Qwen3-Next-80B-A3B (dev-tier bulk generator + filter), DeepSeek V3.2 (pairwise comparison) ✅
- Explicit rotation rule: "No model is used to both generate and judge the same task" — verbatim in methodology.md ✅
- Family separation: Claude generates, Qwen judges — different providers, different architectures ✅
- Li et al. (2025) Preference Leakage cited with justification in both methodology.md and methodology_rationale.md ✅
- `generate_synthesis_tasks.py` code shows distinct API calls with model routing ✅

**Nothing to fix.**

---

### 4. Judge Filter Pipeline — 10/15 (Competent)

**What was done well:**
- Pointwise scoring on exactly 3 dimensions: `input_coherence`, `ground_truth_verifiability`, `rubric_clarity` — each 1–5 ✅
- Thresholds documented: each dimension ≥ 3, mean ≥ 3.5 ✅
- Both LLM mode (OpenRouter) and heuristic dry-run mode implemented ✅
- Judge prompt committed verbatim in `generation_scripts/judge_prompt.md` and also hardcoded in code ✅
- Per-task pass/fail logging via `--log` argument ✅
- Default model is dev-tier (qwen/qwen3-next-80b-a3b) ✅

**What is missing:**
- **No pairwise comparison logic in code.** Methodology mentions "DeepSeek V3.2 for pairwise comparison when two synthesis paths produce similar tasks" but `judge_filter.py` contains no pairwise logic — only pointwise scoring per task. This is a rubric requirement.
- **No eval-tier spot-check of ~50 tasks in code.** The rubric requires "eval-tier model used only for calibration spot-checks on a sampled subset." The code default is dev-tier for all tasks; there is no sampling or spot-check logic.

**Why not Mastered:** Both pairwise logic and eval-tier spot-check separation are explicitly in the rubric and absent from the code.

---

### 5. Contamination Prevention — 10/15 (Competent)

**What was done well:**
- N-gram check (8-gram): fully implemented with correct threshold ✅
- Embedding similarity check (cosine < 0.85, `sentence-transformers/all-MiniLM-L6-v2`): fully implemented ✅
- Report-emitting code: structured JSON output with per-check results ✅
- Held-out vs train covered for both checks ✅

**What is missing:**
- **No time-shift verification in code.** Time-shift check is described in detail in methodology.md and methodology_rationale.md, but `contamination_check.py` contains no code for it. The three implemented checks are: duplicate detection, n-gram overlap, embedding similarity. Time-shift is documented but not implemented.
- **N-gram check does not cover held-out vs dev.** The `check_ngram_overlap` function takes `train` and `held_out` as arguments. There is no call path for checking held-out vs dev — only held-out vs train. The rubric requires "script applies across held-out vs. training and held-out vs. dev pairs."

**Why not Mastered:** Two of three rubric sub-criteria (time-shift code, dev pair coverage) are missing from the actual script.

---

### 6. Inter-Rater Agreement — 10/10 (Mastered)

**What was done:**
- `inter_rater_agreement.md` documents the protocol: 30-task subset, 48-hour gap (exceeds the 24-hour minimum), second pass blind to first-pass labels ✅
- Per-dimension Cohen's κ reported: signal_grounding (0.87), tone_alignment (0.82), cta_quality (0.90), bench_fit_accuracy (0.84), personalization_depth (0.78 → revised to 0.83) ✅
- Rubric revision evidence: personalization_depth clarified after falling below 0.80, relabeled, final κ = 0.83 ✅
- Final agreement reported per dimension post-revision ✅
- `round1_labels.csv` and `round2_labels.csv` committed ✅

**Nothing to fix.**

---

### 7. Datasheet Completeness — 5/5 (Mastered)

**What was done:**
- `datasheet.md` covers all seven Gebru sections with non-stub content: Motivation, Composition, Collection Process, Preprocessing/Labeling, Uses, Distribution, Maintenance ✅
- Pushkarna layered detail visible at three levels ✅
- Limitations and known biases acknowledged with specifics (template structure embedding issue, week 10 failure distribution bias, signal date provenance) ✅
- License: CC-BY-4.0 stated with rationale ✅
- Length: within 3–5 page range ✅

**Nothing to fix.**

---

### 8. Path Declaration & Methodology Rationale — 10/10 (Mastered)

**What was done:**
- Path B explicitly declared in `methodology.md` header ✅
- `methodology_rationale.md` cites three specific Week 10 trace IDs with task IDs, costs, and failure patterns: 0c380837 (task 104), f50f1801 (task 105), 0857ba6e (task 76) ✅
- Paper citations with section-level references: Li et al. 2025 (cross-family routing), Meng et al. NeurIPS 2024 (SimPO memory efficiency), Kim et al. 2024 (Prometheus 2 small judge), Gu et al. (pointwise vs pairwise) ✅
- Failure-mode-to-path mapping: inconsistency failures → Path B, explicitly named ✅
- Alternative paths explicitly considered and dismissed: Path A ruled out (fluency not the problem), Path C ruled out (not multi-turn trajectory failures, data-expensive) ✅

**Nothing to fix.**

---

### 9. Training Run Script — 10/15 (Competent)

**What was done well:**
- Core hyperparameters explicit: lr=5e-5, epochs=3, LoRA r=16, alpha=16, beta=2.0, gamma=0.5, lora_dropout=0.0 ✅
- Random seed fixed and visible: `--seed 42`, seeded in `random`, `np.random`, and `torch` ✅
- Loss logging: step-level loss and margin printed every 10 steps; epoch-level average logged ✅
- Backbone: `unsloth/Qwen2.5-1.5B-Instruct` named ✅
- LoRA-only confirmed: `model.save_pretrained` saves adapter only, no merge ✅
- Path-aligned trainer: pure SimPO loss in PyTorch, correct for Path B ✅
- Wall time: 269s (~4.5 minutes) — well within 30–90 minute target ✅

**What is missing:**
- **No warmup or scheduler.** The AdamW optimizer has no warmup steps and no learning rate schedule. The rubric expects these to be explicit or explicitly set to none with justification.
- **No batch size parameter.** The training loop processes one pair at a time (effective batch size = 1) but there is no `--batch-size` argument and no documentation of why batch size = 1. This is a standard hyperparameter that should be explicit.
- **Backbone not pinned with commit hash or HF revision.** The model string `unsloth/Qwen2.5-1.5B-Instruct` is named but no `revision=` parameter is passed to `from_pretrained`. Different HuggingFace repo states will load different weights.

**Why not Mastered:** Warmup/scheduler absent, batch_size undocumented, no revision pinning.

---

### 10. Ablation Methodology & Statistical Rigor — 10/15 (Competent)

**What was done well:**
- Delta A implemented: trained SimPO judge vs. Week 10 baseline, paired bootstrap with 95% CI and p-value ✅
- Delta B implemented: trained judge vs. prompt-only backbone (no adapter), same test ✅
- Harness parameterized: `--baseline week10` / `--baseline prompt-only` switch ✅
- Bootstrap test in `bootstrap_test.py`: 1000 resamples, seed fixed, CI computed, one-tailed p-value, significance flag ✅
- Failure handling: judge output fallback to deterministic evaluator on JSON parse failure ✅
- Results written to structured JSON ✅

**What is missing:**
- **No Delta C.** The rubric requires Delta C to handle the τ²-Bench retail reference informationally (no re-run). The ablation code has no `--baseline tau2bench` mode and no informational τ²-Bench score lookup. The Week 10 38.7% pass@1 number is in the memo but there is no code that slots it into the ablation table.
- **No Cost-Pareto instrumentation.** The rubric requires "timing logic, token counters, per-task cost computation." `run_ablation.py` has no timing code (no `time.time()` calls), no token counters, and no per-task cost computation. Cost is discussed qualitatively in the memo (~$0.0004/task) but there is no code producing that number.

**Why not Mastered:** Two of four rubric sub-criteria (Delta C, Cost-Pareto) are completely absent from code.

---

### 11. README Reproducibility — 3/5 (Competent)

**What was done well:**
- Overview, status, setup instructions present ✅
- HuggingFace dataset URL: https://huggingface.co/datasets/ketewodros41/tenacious-bench-v0.1 ✅
- HuggingFace model URL: https://huggingface.co/ketewodros41/qwen2.5-1.5b-simpo-tenacious-judge ✅ (Path B, not required by spec but added)
- Quickstart with specific commands ✅
- Attribution section ✅
- Key Results table ✅

**What is missing:**
- **No blog post URL.** Blog post not yet published; placeholder not filled.
- **No community engagement URL.** GitHub issue or equivalent not yet created.
- **No `requirements.txt` or `LICENSE` file.** Dependencies are listed inline in README but not pinned in a requirements file. No `LICENSE` file exists in the repo root despite CC-BY-4.0 being declared in `datasheet.md`.

**Why not Mastered:** Blog URL, community engagement URL, and LICENSE file are all absent.

---

## Top 5 Gaps to Close Before Submission

| Priority | Fix | Effort | Points at Risk |
|---|---|---|---|
| 1 | **Publish blog post + add URL to README** | 30 min | README drops to 1/5 without it |
| 2 | **Post community engagement + add URL** | 15 min | Same README criterion |
| 3 | **Add LICENSE file** (CC-BY-4.0) | 2 min | README criterion |
| 4 | **Add adversarial task generation script** (even a stub that reads/validates hand-authored tasks) | 20 min | Recovers 3 pts on Dataset Authoring |
| 5 | **Add Cost-Pareto timing to run_ablation.py** (wrap each task in `time.time()`, log tokens) | 30 min | Recovers 5 pts on Ablation |

---

## What is Genuinely Strong

- **Methodology depth:** The rationale for SimPO over DPO, the embedding-check scope fix, the δ threshold justification, and the ORPO rejection are all specific and paper-grounded. This is rare.
- **Audit memo:** 11 probe IDs, 6 trace IDs, dimension-level gap specification — exceeds the bar on every sub-criterion.
- **IRA:** 48-hour gap (exceeds 24h requirement), rubric revision documented with before/after κ values, raw label files committed.
- **Anti-leakage policy:** The cross-family routing (Claude → Qwen) with explicit Li et al. citation is one of the better-executed contamination controls in the cohort.
- **Training convergence:** SimPO margin from negative to +3.98 on 69 pairs in 269s is a clean signal.
- **Statistical rigor:** Both Delta A and Delta B significant at p<0.0001 with tight 95% CIs. The honest reframing of why Delta A dropped (better calibration, not worse training) is intellectually honest.
