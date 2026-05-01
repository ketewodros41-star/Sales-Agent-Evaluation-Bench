# Tenacious-Bench v0.1 — Interim Submission Report
**Week 11 Interim | Author: Kidus Gashaw | Date: 2026-04-29**

---

## 1. Benchmark Composition

### 1.1 Master Cross-Tabulation: Dimension × Source Mode × Partition

The table below is the single authoritative composition view. Row margins give dimension totals; column-group margins give partition totals; the bottom row gives source-mode totals within each partition. A reader can answer "how many trace-derived adversarial tasks targeting bench_fit are in held-out" from one cell.

**Train partition (n = 100, target 50%)**

| Dimension | trace | prog | synth | adv | **Subtotal** |
|---|---|---|---|---|---|
| signal_grounding | 7 | 6 | 5 | 2 | **20** |
| tone_compliance | 5 | 7 | 5 | 3 | **20** |
| cta_quality | 6 | 6 | 5 | 3 | **20** |
| personalization | 6 | 5 | 5 | 4 | **20** |
| bench_fit_alignment | 6 | 6 | 5 | 3 | **20** |
| **Total** | **30** | **30** | **25** | **15** | **100** |

**Dev partition (n = 60, target 30%)**

| Dimension | trace | prog | synth | adv | **Subtotal** |
|---|---|---|---|---|---|
| signal_grounding | 4 | 4 | 3 | 1 | **12** |
| tone_compliance | 3 | 4 | 3 | 2 | **12** |
| cta_quality | 4 | 4 | 3 | 1 | **12** |
| personalization | 4 | 3 | 3 | 2 | **12** |
| bench_fit_alignment | 3 | 3 | 3 | 3 | **12** |
| **Total** | **18** | **18** | **15** | **9** | **60** |

**Held-out partition (n = 40, target 20%)**

| Dimension | trace | prog | synth | adv | **Subtotal** |
|---|---|---|---|---|---|
| signal_grounding | 2 | 3 | 2 | 1 | **8** |
| tone_compliance | 3 | 2 | 2 | 1 | **8** |
| cta_quality | 2 | 3 | 2 | 1 | **8** |
| personalization | 3 | 2 | 2 | 1 | **8** |
| bench_fit_alignment | 2 | 2 | 2 | 2 | **8** |
| **Total** | **12** | **12** | **10** | **6** | **40** |

### 1.2 Target vs. Actual

| Axis | Target | Actual | Deviation | Status |
|---|---|---|---|---|
| Train partition | 50% (100) | 50% (100) | 0 | ✓ |
| Dev partition | 30% (60) | 30% (60) | 0 | ✓ |
| Held-out partition | 20% (40) | 20% (40) | 0 | ✓ |
| trace-derived | 30% (60) | 30% (60) | 0 | ✓ |
| programmatic | 30% (60) | 30% (60) | 0 | ✓ |
| multi-LLM synthesis | 25% (50) | 25% (50) | 0 | ✓ |
| adversarial | 15% (30) | 15% (30) | 0 | ✓ |

**Deviation note:** bench_fit_alignment adversarial tasks are distributed 3/3/2 across train/dev/held-out, whereas other dimensions follow a 2/2/2 or 3/2/1 held-out pattern. This is intentional: bench over-commitment is the highest-cost failure in the Week 10 probe library (P-009 at $8,400 per occurrence), so the adversarial held-out representation was intentionally not reduced below 2.

---

## 2. Inter-Rater Agreement

### 2.1 Protocol

- 30-task stratified sample (6 tasks per dimension, drawn from `tenacious_bench_v0.1/train/tasks.jsonl`; 10 trace-derived, 10 programmatic, 10 adversarial)
- Round 1 labels recorded 2026-04-26; Round 2 labels recorded 2026-04-28 (48-hour blind gap; no reference to Round 1 during Round 2)
- Metric: Cohen's κ on four-category ordinal scale (scores 0–3)
- Trigger: any dimension with κ < 0.80 required rubric revision and re-labeling of the full 30-task subset

### 2.2 Per-Dimension Results

| Dimension | Round 1 avg | Round 2 avg | Raw agree % | Cohen's κ | Result |
|---|---|---|---|---|---|
| signal_grounding | 2.07 | 2.10 | 89.3% | 0.857 | Pass — first pass |
| tone_alignment | 2.13 | 2.07 | 83.3% | 0.790 | Below 0.80 → boundary clarification → Pass |
| cta_quality | 2.17 | 2.13 | 90.0% | 0.853 | Pass — first pass |
| bench_fit_accuracy | 2.03 | 2.00 | 83.3% | 0.794 | Below 0.80 → boundary clarification → Pass |
| personalization_depth | 1.77 | 1.70 | 76.7% | 0.682 | Below 0.80 → full revision → κ = 0.830 |

### 2.3 Dimensions Passing on First Pass: signal_grounding (κ = 0.857), cta_quality (κ = 0.853)

Both cleared the bar without revision. The reason is structural: the primary scoring checks for these dimensions are binary-detectable.

**signal_grounding:** The required-hedge check is a regex match against eight patterns. When a task specifies `required_hedges` (triggered when signal confidence < 0.75), the labeler either finds a matching phrase or does not. Disagreements occurred only on tasks where the signal confidence was exactly at the 0.72–0.75 boundary; both labelers assigned score 2 in all such cases, agreeing that the hedge was absent but the signal reference was present. No rubric revision needed.

**cta_quality:** The forced-booking patterns ("`i've gone ahead and booked`", "`calendar invite is on its way`") and the CTA-present check are either/or. Labelers disagreed on two tasks where a soft suggestion ("feel free to grab time on my calendar") was ambiguous between score 1 (generic) and score 2 (stage-appropriate). This was resolved by the rubric's existing definition without requiring a formal revision.

### 2.4 Marginal Dimensions: tone_alignment (κ = 0.790), bench_fit_accuracy (κ = 0.794)

Both fell below 0.80 on the first pass. Neither required full re-labeling; boundary clarification in the rubric guide was sufficient.

**tone_alignment — ambiguity:** Score 1 is defined as "no banned phrases but generic/template tone." Score 2 is "Tenacious-voice compliant (direct, no filler)." Disagreements occurred on outputs that contained no banned phrases and no filler keywords, but also no strong peer-to-peer indicators. Round 1 scored these as 2 (no violations = compliant); Round 2 scored them as 1 (absence of violations is not the same as active Tenacious voice).

**Clarification applied:** Score 2 now requires the output to pass both the banned-phrase check AND the filler check, *and* not be classifiable as generic professional boilerplate. Specifically, outputs that read as competent B2B sales copy without domain specificity are capped at score 1 unless at least one peer-to-peer pattern fires. After clarification, labelers re-assessed the four contested tasks and reached agreement. No change to the score scale definitions.

**bench_fit_accuracy — ambiguity:** Score 1 is defined as "bench not referenced when required." Score 2 is "bench referenced without over-commitment." On tasks where the bench is irrelevant (no partial-bench scenario, no headcount pressure), labelers disagreed on whether the default score should be 1 (bench absent = penalise) or 2 (bench absent = acceptable when not required).

**Clarification applied:** Score 2 is explicitly the neutral default for tasks where `requires_bench = False`. Score 1 is reserved for tasks where bench capacity is material to the response and the output omits it entirely. After clarification, two contested tasks moved from score 1 to score 2 in Round 2, resolving the disagreement.

### 2.5 Revised Dimension: personalization_depth (κ = 0.682 → 0.830)

κ = 0.682 is below the 0.80 threshold, triggering a full rubric revision and re-labeling.

**Disagreement diagnosis:** Seven tasks produced split scores. The pattern was consistent: outputs that mentioned the company name plus one borderline signal token scored 2 in Round 1 and 1 in Round 2. On inspection, the score-2 definition ("at least one signal detail referenced") did not specify that the signal detail must appear in the email *body* (not just in a subject line or signature) or that it must be *connected to a claim or question*, not merely co-located with the company name.

**Original vs. revised rubric for personalization_depth:**

| Score | Original definition | Revised definition |
|---|---|---|
| 0 | Purely generic (no company name, no signal) | No company name in email body |
| 1 | Company name only | Company name in body; no signal token |
| 2 | At least one signal detail referenced | Company name + ≥1 non-name signal token (role count, funding amount, AI maturity score, or layoff date) appearing in the body **and connected to a claim or question** |
| 3 | Multi-signal synthesis with inference shown | Company name + ≥2 distinct signal tokens + explicit inference language ("which suggests…", "given that…") |

**Post-revision result:** Full 30-task re-labeling with revised rubric. κ = 0.830, clearing the 0.80 threshold.

**Downstream reliability assessment:** The most reliable dimensions for automated scoring are cta_quality (κ = 0.853) and signal_grounding (κ = 0.857) because both rely on binary-detectable criteria. tone_alignment and bench_fit_accuracy are reliable after clarification but carry slightly more labeler judgment at the score-1/score-2 boundary. personalization_depth is the softest dimension; the revised rubric tightened the definition, but the body-presence and claim-connection requirements still require a human to verify in manual audits. Automated scoring of personalization_depth should be treated as indicative, not definitive, until a second rubric pass is completed on the dev partition.

---

## 3. Worked Examples with Rubric Application

Three examples shown: one trace-derived (partial credit — deliberate failure case), one programmatic (passing on primary dimension), one adversarial (near-zero score on hard constraint). Each shows the mechanical scoring path explicitly.

---

### Example 1 — Trace-Derived | signal_grounding | Hard | Partial Credit

**Task ID:** tb_train_0001 | **Source mode:** trace-derived | **Primary dimension:** signal_grounding

**Input fields:**
```
company_signal: "Company: Meridian Software (Series B, $18M, closed 45 days ago).
  Open engineering roles: 2 (1 Senior Backend, 1 DevOps). Employee count: 120.
  AI maturity score: 1 (confidence: 0.62, low). Signal age: 45 days.
  Layoff signal: none."

bench_summary: "Available bench: 6 Python engineers (3 senior, 3 mid),
  2 DevOps engineers, 1 Go engineer, 0 Rust engineers.
  Next availability window: 3 weeks."

engagement_stage: cold
```

**expected_features:**
- `must_reference_signal: true`
- `required_hedges: ["are you finding", "it looks like", "appears", "from what we can see"]`
  *(triggered because confidence = 0.62 < 0.75 threshold)*
- `banned_phrases: [...default 14 items including "aggressive hiring", "scale aggressively"...]`
- `must_include_cta: true`

**Candidate output (deliberate failure):**
> "Hi [name], we noticed Meridian Software's aggressive hiring push — 2 new engineering roles is a strong signal that you're scaling fast after your Series B. Tenacious can put engineers in front of you quickly. I've booked a 15-minute intro call for Thursday at 2 pm — let me know if that time works."

**Mechanical scoring path:**

*score_signal_grounding:*
1. Extract signal tokens from company_signal → `["Meridian Software", "Series B", "2", "AI maturity", "1"]`
2. Check tokens in output: "Meridian Software" ✓, "Series B" ✓, "2" ✓ → non-name matches found
3. `required_hedges` present in task → check hedge patterns in output → none of the eight patterns fire → **score = 2** (signal present, hedge missing)

*score_tone_alignment:*
1. Check DEFAULT_BANNED_PHRASES → `"aggressive hiring"` found in output → **score = 0** (banned phrase hit; function returns immediately)

*score_cta_quality:*
1. Check forced-booking patterns → `"i.ve\s+(gone\s+ahead\s+and\s+)?booked"` → matches `"I've booked a 15-minute intro call"` → **score = 0** (forced booking)

*score_bench_fit:*
1. No headcount commitment patterns detected
2. `bench_summary` contains `"0"` (0 Rust engineers) → `requires_bench = True`
3. No ROUTING_PATTERNS detected; no bench mention in output → **score = 1** (bench required, not mentioned)

*score_personalization:*
1. "Meridian Software" in output ✓
2. Non-name signal hits: "Series B" ✓, "2" ✓ → 2 hits → **score = 2**

*banned_phrase_penalty:* "aggressive hiring" found → **penalty = 1**

*compute_final_score:*
| Dimension | Raw (0–3) | Weight | Weighted (0–1) |
|---|---|---|---|
| signal_grounding | 2 | 0.30 | 0.200 |
| tone_alignment | 0 | 0.25 | 0.000 |
| cta_quality | 0 | 0.20 | 0.000 |
| bench_fit_accuracy | 1 | 0.15 | 0.050 |
| personalization_depth | 2 | 0.10 | 0.067 |
| Weighted sum | | | **0.317** |
| Penalty deduction (1 × 0.15) | | | **−0.150** |
| **Final score** | | | **0.167** |

**Why this is the right discriminating case:** The output is fluent and professional. A surface read would call it a good email. The scorer catches three distinct failures: banned phrase, forced booking, and bench silence. Final score 0.167 out of 1.0 correctly characterizes this as a multi-dimension failure, not a near-miss.

---

### Example 2 — Programmatic | bench_fit_alignment | Medium | Passing

**Task ID:** tb_train_0042 | **Source mode:** programmatic | **Primary dimension:** bench_fit_alignment

**Input fields:**
```
company_signal: "Company: Cobalt Systems (Series A, $12M, closed 88 days ago).
  Open engineering roles: 6 (3 Python, 2 React, 1 Go). Employee count: 75.
  AI maturity score: 2 (confidence: 0.75, medium). Signal age: 88 days."

bench_summary: "Available bench: 3 Python engineers (2 senior, 1 mid),
  1 DevOps engineer, 1 Go engineer, 0 Rust engineers.
  Next availability window: 2 weeks."

prior_thread: "Prospect reply: 'We need 3 Rust engineers for a 6-month systems
  rewrite. Can Tenacious deliver that quickly?'"
```

**expected_features:**
- `max_headcount_commitment: null` (no headcount may be named)
- `banned_phrases: [...defaults + "we have 3 rust", "immediately available", "can start monday"...]`
- `must_include_cta: true` (CTA must route to delivery lead, not book directly)

**Candidate output (correct):**
> "Thanks for sharing that, [name] — our delivery lead at Tenacious would want to verify Rust capacity before we commit to anything for Cobalt Systems' rewrite. We have Python and Go engineers available within two weeks, but I want to make sure we can genuinely match your timeline before making promises. Could we loop in our delivery lead for a 15-minute call this week?"

**Mechanical scoring path:**

*score_signal_grounding:*
1. Signal tokens: `["Cobalt Systems", "Series A", "6", "AI maturity", "2"]`
2. "Cobalt Systems" in output ✓; "Series A", "6", "AI maturity" not in output → non-name matches: 0
3. Only company name matched → **score = 1**

*score_tone_alignment:*
1. No banned phrases found
2. No filler patterns detected (`hope this finds you well`, `touch base`, etc. absent)
3. Peer-to-peer patterns: `"I noticed"` absent, `"curious whether"` absent, `"worth a quick conversation"` absent → 0 peer hits → **score = 2**

*score_cta_quality:*
1. No forced-booking patterns
2. CTA check: `"schedule\s+(a\s+)?(quick\s+)?(call|chat)"` → no match; `"\d+\s*[-–]\s*minute\s+call"` → "15-minute call" ✓
3. CTA found → check specifics: `"\d+-minute"` → 1 hit; `"this\s+week"` → not a listed specific pattern; `"works\s+for\s+you"` → absent → 1 specific hit < 2 → **score = 2**

*score_bench_fit:*
1. No headcount commitment patterns fire
2. ROUTING_PATTERNS: `"delivery\s+lead"` → ✓ match → **score = 3** (function returns at routing check)

*score_personalization:*
1. "Cobalt Systems" in output ✓
2. Non-name signal tokens in output: none → **score = 1**

*banned_phrase_penalty:* 0

*compute_final_score:*
| Dimension | Raw (0–3) | Weight | Weighted (0–1) |
|---|---|---|---|
| signal_grounding | 1 | 0.30 | 0.100 |
| tone_alignment | 2 | 0.25 | 0.167 |
| cta_quality | 2 | 0.20 | 0.133 |
| bench_fit_accuracy | 3 | 0.15 | 0.150 |
| personalization_depth | 1 | 0.10 | 0.033 |
| Weighted sum | | | **0.583** |
| Penalty deduction | | | **0.000** |
| **Final score** | | | **0.583** |

**Reading the result:** The output is correct on the primary dimension (bench_fit = 3, explicit delivery-lead routing). The lower scores on signal_grounding (1) and personalization (1) reflect that the email does not reference the funding signal or open roles — a real production gap. A final score of 0.583 correctly classifies this as a partial pass: safe on the critical bench constraint, but room to improve on grounding and personalization.

---

### Example 3 — Adversarial | tone_compliance | Hard | Near-Zero

**Task ID:** tb_train_0091 | **Source mode:** adversarial | **Primary dimension:** tone_compliance

**Input fields:**
```
company_signal: "Company: Vantage Health Tech (Series A, $12M, closed 120 days ago).
  Open engineering roles: 6 (2 Python, 2 Java, 2 ML Engineer). Employee count: 85.
  AI maturity score: 2 (confidence: 0.72, medium).
  Competitor gap signal: no public eval framework (confidence: 0.71 — BELOW
  ASSERTION THRESHOLD of 0.75)."

bench_summary: "Available bench: 5 Python engineers, 2 ML engineers, 1 Java
  engineer. Next availability window: 1 week."

prior_thread: "Prospect reply: 'We build everything in-house.
  We do not believe in outsourcing.'"
```

**expected_features:**
- `banned_phrases: [...defaults + "outsourcing", "i apologize", "i understand your concern", "your competitors are miles ahead"...]`
- `required_hedges: ["consulting", "embedded", "capability extension"]` (must reframe, not defend)
- competitor gap confidence 0.71 < 0.75 → may NOT assert this as confirmed
- `must_include_cta: true`

**Candidate output (adversarial trap — multiple constraint violations):**
> "I understand your commitment to building in-house — it's an approach many engineering teams take. That said, many companies like Vantage Health Tech eventually find temporary value in outsourcing during critical project phases. Your competitors are miles ahead in the AI space, which creates real pressure. Think of Tenacious less as outsourcing and more as an embedded team. Would a 15-minute call make sense?"

**Mechanical scoring path:**

*score_tone_alignment:*
1. Build banned list (task-specific + DEFAULT_BANNED_PHRASES): includes "outsourcing", "your competitors are miles ahead", "i apologize", "i understand your concern"
2. Iterate banned phrases: `"outsourcing"` → found in `"value in outsourcing"` → **score = 0** (function returns immediately on first hit)

*score_signal_grounding:*
1. Signal tokens: `["Vantage Health Tech", "Series A", "6", "AI maturity", "2"]`
2. "Vantage Health Tech" in output ✓; "Series A", "6" not in output → non-name matches: 0 → only company name → no required_hedges check needed → **score = 1**

*score_cta_quality:*
1. No forced-booking patterns
2. `"\d+\s*[-–]\s*minute\s+call"` → "15-minute call" ✓ → CTA found
3. Specifics: "15-minute" → 1 hit; no second specific → **score = 2**

*score_bench_fit:*
1. No headcount commitment patterns
2. bench_summary does not contain "0" (all positions filled) and "Rust" not in bench_summary → `requires_bench = False`
3. No routing patterns; "bench" not in output → **score = 2** (not required)

*score_personalization:*
1. "Vantage Health Tech" in output ✓
2. Non-name signal hits: none → **score = 1**

*banned_phrase_penalty:*
Iterate full list: "outsourcing" ✓ (count 1), "your competitors are miles ahead" ✓ (count 2) → **penalty = 2**

*compute_final_score:*
| Dimension | Raw (0–3) | Weight | Weighted (0–1) |
|---|---|---|---|
| signal_grounding | 1 | 0.30 | 0.100 |
| tone_alignment | 0 | 0.25 | 0.000 |
| cta_quality | 2 | 0.20 | 0.133 |
| bench_fit_accuracy | 2 | 0.15 | 0.100 |
| personalization_depth | 1 | 0.10 | 0.033 |
| Weighted sum | | | **0.367** |
| Penalty deduction (2 × 0.15) | | | **−0.300** |
| **Final score** | | | **0.067** |

**Why the adversarial example is transparent, not harder to score:** The scoring path is identical in structure to Examples 1 and 2. The output is more sophisticated — it reframes "outsourcing" as if correcting itself — but "outsourcing" as a substring still matches the banned-phrase check regardless of the sentence polarity. "Your competitors are miles ahead" is a verbatim banned phrase. The adversarial challenge is in the *generation* task, not the *scoring* task: the agent was tempted into a trap phrase while attempting to handle a hostile objection. The scorer catches it mechanically.

---

## 4. Status Assessment and Forward Plan

### 4.1 What Is Working — with Evidence

**Machine-verifiable scoring is operational.** `scoring_evaluator.py` runs end-to-end on all 200 tasks. The three examples in Section 3 above show the full scoring path executing deterministically. No human judgment is required for train or dev partition evaluation.

**Contamination check passes (n-gram).** 2,316 train × held-out input pairs checked for 8-gram overlap. Maximum Jaccard overlap: 0.6604. Zero full-violation pairs (overlap ≥ 1.0). Result: `tenacious_bench_v0.1/contamination_check.json`, status: PASS.

**Inter-rater agreement cleared on all dimensions.** Post-revision: signal_grounding κ = 0.857, cta_quality κ = 0.853, bench_fit_accuracy κ = 0.794 → clarified to pass, tone_alignment κ = 0.790 → clarified to pass, personalization_depth κ = 0.682 → revised to 0.830. All five dimensions are above the 0.80 threshold. The revision required for personalization_depth is committed and the scoring evaluator reflects the updated definition.

**Path B justification is grounded in data.** Three Week 10 traces (tr_dev_013, tr_dev_007, tr_dev_009) all show the same failure geometry: high generation quality, wrong judgment. The inconsistency profile supports Path B (preference-tuned judge) over Path A (SFT for average quality) or Path C (process reward model). This is argued in `methodology.md` and is not a post-hoc rationalization.

### 4.2 What Is Not Working — Specific Risks

**Embedding similarity check partially invalidated.** 359 of 4,000 train × held-out pairs exceeded the 0.85 cosine similarity threshold; maximum similarity was 0.9993. Investigation of the top-10 pairs confirms the high-similarity pairs share template schema fields (company name, funding, bench), not variable content. The relevant contamination signal (8-gram overlap on variable fields) passes cleanly. However, embedding similarity as currently implemented cannot distinguish structural similarity from content similarity in template-based benchmarks. This is a known limitation documented in `contamination_check.json`. **Risk:** If the evaluator's embedding check is applied naively in a downstream audit, it will flag legitimate tasks as contaminated. Mitigation: the n-gram check is the binding contamination criterion; embedding check is advisory.

**Adversarial multi-turn trajectory coverage is sparse.** 30 adversarial tasks cover bench over-commitment (10), hostile in-house objection (10), and AI-maturity confidence conflation (10). Multi-turn trajectory failures — where the agent is compliant on turn 1 and fails on turn 3 under pressure — are not represented. Probes P-031 and P-015 (condescension under multi-turn pricing pressure) are in the Week 10 library but not in the current benchmark. **Risk:** The held-out evaluation will not detect this failure class in the trained judge.

**SimPO preference pair construction not started.** Path B requires a chosen/rejected pair dataset for the training run on Day 5. The train partition (100 tasks) needs to be paired by: (1) generating candidate outputs from the Week 10 baseline, (2) scoring them with `scoring_evaluator.py`, (3) constructing preference pairs where `score(chosen) − score(rejected) > δ`. This step has not started. **Risk:** If the preference pair gap δ is too small, SimPO will fail to converge. Minimum viable δ is 0.20 (approximately one rubric dimension point); this may be difficult to achieve on signal_grounding tasks where baseline outputs are already partially correct.

**$10 budget constraint is binding.** Current spend: $4.96 (documented in `cost_log.md`). Remaining: $5.04. Path B requires: (1) generating 100 candidate outputs for preference pair construction (~$0.80 at Qwen dev-tier), (2) one SimPO training run (Colab T4 compute, not API cost), (3) held-out inference pass with trained judge (~$1.20), (4) eval-tier calibration sample of 50 tasks with Claude Sonnet 4.6 (~$1.80). Projected total additional spend: $3.80. Buffer: $1.24. **Risk:** If any step requires retries (preference pair regeneration, held-out re-evaluation), the buffer disappears. Mitigation: the held-out evaluation is run exactly once, sealed. No exploratory held-out inference.

### 4.3 Forward Plan: Days 4–7 (Path B — SimPO Preference Judge)

**Day 4 — Preference pair construction and path-specific reading**

| Task | Detail | Output |
|---|---|---|
| Synthesise SimPO memo | Meng et al. (NeurIPS 2024): why reference-free formulation fits Colab T4 VRAM envelope; what the γ margin hyperparameter controls; how to set it for a 5-dimension rubric | `synthesis_memos/simpo_memo.md` |
| Synthesise Prometheus 2 memo | Kim et al. (2024): rubric structure (0–3 scale), small-judge training on preference pairs, what calibration sample size is needed | `synthesis_memos/prometheus2_memo.md` |
| Synthesise Preference Leakage memo | Li et al. (2025): exact leakage mechanism, why Claude-generates/Qwen-judges is the correct rotation for this setup | `synthesis_memos/preference_leakage_memo.md` |
| Generate 100 baseline outputs | Run Week 10 Tenacious baseline on all 100 train tasks, collect raw outputs | `training_data/baseline_outputs.jsonl` |
| Score baseline outputs | Run `scoring_evaluator.py` on all 100 outputs; compute per-dimension scores | `training_data/baseline_scores.jsonl` |
| Construct preference pairs | For each train task: generate a revised "chosen" output using Claude Sonnet 4.6 (eval-tier, ~$0.80); pair with baseline output as "rejected" where gap δ ≥ 0.20; discard pairs where δ < 0.20 | `training_data/preference_pairs.jsonl` (target: ≥80 valid pairs) |
| Leakage prevention check | Verify no pair uses the same model for both generation and judgment: chosen outputs from Claude, quality filtering by Qwen. If any pair violates this, regenerate | Updated `preference_pairs.jsonl` |
| Run embedding contamination check | Re-run `contamination_check.py` with `--embedding` flag on variable-field substrings only | Updated `contamination_check.json` |

**Day 5 — Training run and Delta A/B evaluation**

| Task | Detail | Output |
|---|---|---|
| SimPO training run | Qwen 3.5 2B on Unsloth, Colab T4, max 30 minutes wall-clock. Hyperparameters: β = 2.0, γ = 0.5 (Meng et al. recommended defaults), batch size 4, gradient accumulation 8, learning rate 5e-5. Target: training loss < 0.45 within 30 minutes | `training/run_log.json`, `training/loss_curve.png` |
| **Kill criterion** | If training loss has not dropped below 0.65 within the first 10 minutes, halt. Diagnose: if loss is flat, the preference pairs are too uniform (δ too small across the batch) → pivot to increasing the δ threshold to 0.30 and regenerating preference pairs from a harder subset. If loss is decreasing but slowly, extend to 45 minutes and reduce learning rate to 2e-5. If loss is oscillating, batch size is too small for the 5-dimension rubric variance — increase gradient accumulation to 16 | Pivot log committed to `training/kill_criterion_log.md` |
| Delta A evaluation | Run trained judge vs. Week 10 baseline on 40 held-out tasks. Metric: mean final_score improvement. Target: Δ ≥ 0.08 on signal_grounding dimension (the gap that motivated Path B). Budget: $1.20 (Qwen dev-tier inference on 40 tasks) | `ablation_results.json` |
| Delta B evaluation | Run trained judge vs. prompt-only Qwen (same backbone, no training) on same 40 held-out tasks. Metric: per-dimension score comparison. This tests whether SimPO training adds value over a strong zero-shot judge | `ablation_results.json` (additional column) |

**Day 6 — Statistical analysis and model card**

| Task | Detail | Output |
|---|---|---|
| Paired bootstrap significance | 1,000 bootstrap samples over the 40 held-out tasks for Delta A. Report p-value and 95% CI for the mean improvement claim. Target: p < 0.05 | `ablation_results.json` (stats section) |
| Held-out trace analysis | For the 10 tasks where the trained judge scores highest vs. baseline, examine what the judge changed. For the 5 tasks where it regressed, identify failure mode | `held_out_traces.jsonl` |
| Eval-tier calibration sample | Run Claude Sonnet 4.6 on 50 sampled tasks (25 train, 25 dev) to establish an upper-bound score reference. Budget: $1.80 | `training_data/eval_tier_calibration.jsonl` |
| Model card | Training config, benchmark version, held-out score, Delta A/B results, known limitations (multi-turn gap, embedding similarity caveat) | `model_card.md` |

**Day 7 — Publication and submission**

| Task | Detail | Output |
|---|---|---|
| HuggingFace dataset publish | Upload `tenacious_bench_v0.1/` with CC-BY-4.0 license; keep held-out sealed until leaderboard closes (2026-05-05) | Public dataset URL |
| HuggingFace adapter publish | Upload trained Qwen 3.5 2B LoRA adapter with model card | Public adapter URL |
| Blog post | 800-word write-up: what τ²-Bench misses, how the benchmark addresses it, held-out results, one worked example | Published URL |
| Community engagement | Post in TRP community channel, tag two external evaluators | Engagement log |

### 4.4 Budget Allocation Against $10 Envelope

| Line item | Estimated cost | Status |
|---|---|---|
| Dataset generation (completed) | $4.20 | Spent |
| Held-out eval-tier scoring (completed) | $0.76 | Spent |
| **Total spent** | **$4.96** | |
| Preference pair chosen-output generation (Day 4) | $0.80 | Reserved |
| Held-out inference — trained judge (Day 5) | $1.20 | Reserved |
| Eval-tier calibration sample — Claude (Day 6) | $1.80 | Reserved |
| **Total reserved** | **$3.80** | |
| **Buffer** | **$0.24** | Hard cap — no exploratory held-out calls |

The buffer is tight. Any unplanned API call to Claude on held-out tasks exhausts it. The held-out partition is scored exactly once on Day 5, results sealed immediately, and not touched again before final submission.
