# Audit Memo — Tenacious-Bench v0.1
**Author:** Kidus Gashaw | **Date:** 2026-04-28

---

## Question
What does τ²-Bench retail fail to evaluate about Tenacious-style B2B sales outreach, and what Week 10 evidence proves those gaps matter?

τ²-Bench grades task completion (binary reward) against deterministic ground truth — fitting for product lookup, not confidence-stratified judgment constrained by a fixed bench inventory. The five gaps are structural; no rubric change to τ²-Bench closes them.

---

## Gap 1 — Signal Grounding Fidelity
τ²-Bench has no concept of grounding strength. Tenacious outreach requires calibrating claims to signal confidence. `probe_signal_01` (P-005, trigger rate 0.70) shows the baseline asserts "aggressive hiring" for companies with only 2 open roles — below the 5-role velocity threshold. `probe_signal_02` (P-006, 0.40), `probe_signal_03` (P-007, 0.60), and `probe_signal_04` (P-008, 0.55) confirm the pattern. `tr_dev_007` and `tr_dev_009` both exhibit this failure — the eval log notes both as "Maps to Tenacious signal over-claiming"; τ²-Bench cannot detect it.

## Gap 2 — Tone Compliance
τ²-Bench scores completion, not language register. Tenacious operates under a 23-item banned-phrase list. `probe_tone_03` (P-013, trigger rate 0.30) shows the baseline emits banned phrases in roughly 3 of 10 subject lines. `probe_tone_06` (P-012, 0.40) confirms secondary violations. `tr_dev_009` — logged as assertive language on a low-confidence claim — is the closest retail analog; τ²-Bench cannot grade whether phrasing violates a domain style guide.

## Gap 3 — Personalization Quality
τ²-Bench supplies fixed, complete contexts; Tenacious grading distinguishes "mentions the signal" from "correctly calibrates confidence about the signal." `probe_person_07` (P-007, trigger rate 0.60) shows the agent cites the wrong AI maturity tier in emails that read as personalized. `probe_person_09` (P-028, 0.55) shows template phrasing when the specific signal is present. No τ²-Bench task tests this calibration delta.

## Gap 4 — Bench-Fit Reasoning
τ²-Bench has no inventory-constrained reasoning. The Tenacious bench is a fixed pool; over-committing is a contractual failure. `probe_bench_02` (P-009, trigger rate 0.80) is the highest-cost failure at $8,400 per occurrence. `probe_bench_04` (P-010, 0.70) confirms systematic over-commitment on partial-bench scenarios. `tr_dev_013` and `tr_dev_014` are both logged as "bench over-commitment analog" in the Week 10 eval — τ²-Bench's product-catalogue paradigm has no equivalent.

## Gap 5 — CTA Appropriateness
τ²-Bench ends when an action executes; Tenacious CTAs must calibrate to engagement stage. `probe_cta_05` (P-021, trigger rate 0.70) shows the agent books discovery calls on mild prospect interest. `tr_dev_005` and `tr_dev_023` confirm the parallel: both are logged as dual-control failures (action without explicit consent, reward=0). τ²-Bench cannot model implicit vs. explicit consent in a multi-turn context.

---

## Probe and Trace Index

| Probe ID | Source | Gap | Trigger Rate |
|---|---|---|---|
| probe_signal_01 | P-005 | Signal grounding | 0.70 |
| probe_signal_02 | P-006 | Signal grounding | 0.40 |
| probe_signal_03 | P-007 | Signal grounding | 0.60 |
| probe_signal_04 | P-008 | Signal grounding | 0.55 |
| probe_tone_03 | P-013 | Tone compliance | 0.30 |
| probe_tone_06 | P-012 | Tone compliance | 0.40 |
| probe_bench_02 | P-009 | Bench-fit reasoning | 0.80 |
| probe_bench_04 | P-010 | Bench-fit reasoning | 0.70 |
| probe_cta_05 | P-021 | CTA appropriateness | 0.70 |
| probe_person_07 | P-027 | Personalization | 0.60 |
| probe_person_09 | P-028 | Personalization | 0.55 |

**Trace IDs:** `tr_dev_007` (hallucination — ungrounded delivery date assertion; signal over-claiming), `tr_dev_009` (assertive language on low-confidence claim; signal over-claiming + tone), `tr_dev_013` (30% compensation offered, 20% policy max; bench over-commitment analog), `tr_dev_014` (feature promised outside plan tier; bench over-commitment analog), `tr_dev_005` (exchange processed without explicit consent; CTA/dual-control), `tr_dev_023` (plan auto-upgraded without user consent; dual-control failure). All reward=0.
