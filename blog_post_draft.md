# Tenacious-Bench v0.1: What Happens When You Build a Benchmark for Your Own Agent's Failures
**Author:** Kidus Gashaw | **Date:** [FILL IN publish date]
**Published at:** [FILL IN: HuggingFace Community / Substack / personal site]

---

Most benchmark papers start with a general problem. This one starts with a specific embarrassment.

The Tenacious AI sales agent — an LLM-powered system that writes B2B outreach emails for a talent platform — achieves **38.7% pass@1** on the τ²-Bench retail evaluation slice. That means roughly one in three emails it produces fully satisfies the quality rubric. For the other two, something is wrong: the agent over-commits on bench capacity, uses assertive tone when the prospect is clearly still researching, or references prospect signals so generically that they might as well not be there.

That gap — from 38.7% to something worth deploying — is what this project is about. But before I could close the gap, I had to understand exactly where it was. That meant building a benchmark.

---

## The Gap τ²-Bench Couldn't See

τ²-Bench is a retail-domain evaluation benchmark for sales agents. It is well-constructed and widely used. It is also the wrong benchmark for Tenacious.

Tenacious operates in B2B enterprise sales for an AI talent platform. The failure modes are domain-specific: the agent must know that "let's get you booked in this week" is a banned phrase in the Tenacious style guide; it must know that confirming availability for a bench role without hedging on partial bench state is a policy violation; it must know that writing "AI-native company" copy for a prospect who is still at the "exploring AI options" stage is a tone mismatch. τ²-Bench retail does not test any of these things. Its rubric covers general outreach quality — relevance, structure, call-to-action appropriateness. The things it cannot grade are the exact things that fail in production.

I ran the Tenacious agent against τ²-Bench's retail slice and saw 38.7% pass@1. Then I read the failure traces. Three stood out:

- **Trace 0c380837** (task 104): The agent confirmed availability for a senior engineer the following week. The bench context showed two of three required skill slots were already committed. The confirmation was a policy violation. τ²-Bench didn't catch it because it has no bench capacity model.

- **Trace f50f1801** (task 105): The agent wrote implementation-stage copy for a prospect who was clearly still at the "researching AI vendors" stage. The copy was confident and well-structured. It was also misaligned with the prospect's actual position. τ²-Bench scored it acceptable because it looked like good outreach.

- **Trace 0857ba6e** (task 76): The agent referenced signals — "recently visited pricing page," "attended a webinar" — without using the specific values from the task context. The output mentioned signals in the abstract. The task required grounding in the specific signals provided. τ²-Bench does not test grounding fidelity.

These are not edge cases. They are the majority of Tenacious's failure distribution.

---

## Building a Benchmark From Failure Traces

Tenacious-Bench v0.1 is a 200-task benchmark built specifically to detect these failure modes. It uses four task-generation methods:

**Trace-derived tasks (60 tasks):** I took the failed agent outputs from Week 10's evaluation runs, extracted the failure pattern (what went wrong and why), and reconstructed them as benchmark tasks. Real company and prospect details were replaced with synthetic analogs. Three failed traces seeded 60 tasks by varying the signal confidence, bench state, and prior thread context.

**Programmatic tasks (60 tasks):** A parameter sweep across company size, signal confidence, bench availability, and prior thread warmth produced systematic coverage of the rubric's boundary conditions. Rather than reflecting what the agent happened to fail on, these tasks stress-test every combination.

**Multi-LLM synthesis tasks (50 tasks):** A two-model pipeline — Claude Sonnet 4.6 generates seed tasks, Qwen judges them against the rubric — produced tasks covering cross-dimension scenarios. The key design decision: the generator and judge are different model families. In dry-run testing, Claude judging Claude-generated tasks accepted 78% of candidates. Qwen judging the same candidates accepted 4.2%. That gap is preference leakage — the same phenomenon Li et al. (2025) name and characterize. The cross-family routing eliminates it.

**Adversarial tasks (30 tasks):** Hand-authored tasks designed to trigger the failure modes the automated pipeline wouldn't generate: bench over-commitment under direct questioning, tone violations under hostile objection pressure, confidence conflation between AI maturity signal and AI readiness. These bypassed the automated quality filter and were validated by hand.

---

## The Scoring Architecture

The benchmark uses a five-dimension deterministic scoring evaluator with a key design choice: four of the five dimensions require no LLM judge at all.

- **cta_quality**: regex-based detection of forcing booking CTAs, urgency language, and commitment patterns
- **tone_compliance**: banned phrase detection against a 23-item list plus assertiveness scoring
- **bench_fit_alignment**: capacity commitment pattern matching against the task's bench state
- **signal_grounding**: token-level match between the output and the signal field values in the task input

Only **personalization_depth** requires any interpretation beyond pattern matching, and after rubric revision, it was reformulated as "does the output reference at least two specific tokens from the prospect context?" — effectively a lookup.

The inter-rater agreement results confirm this is the right design. The two most mechanically-implemented dimensions — cta_quality (κ = 0.853) and signal_grounding (κ = 0.857) — are the most reliably scored. The dimension that came closest to requiring judgment — personalization_depth — had κ = 0.682 on first pass, which triggered a rubric revision specifically to make it more mechanical. After revision, κ improved to 0.830.

This design disagrees with the standard recommendation in the LLM-as-Judge literature (Gu et al. 2024–2025) that reference-based LLM judging is the gold standard. For tasks where the quality criterion is a conjunction of binary-checkable properties — phrase presence, pattern matching, token grounding — a deterministic evaluator is faster, cheaper, and more reliable. LLM judging is appropriate for holistic quality assessment. Tenacious-Bench is not that.

---

## Act III: Training a Judge

The benchmark reveals the failure geometry. The trained component closes the gap.

I chose Path B: train a preference judge using SimPO (Meng et al. NeurIPS 2024). The judge sits after the generator — it scores the draft and flags outputs that fall below threshold for revision or human review. This matches how Tenacious would actually deploy an improvement: not a retrained generator, but a quality gate on the output.

SimPO was chosen over DPO because it eliminates the reference model, making training feasible on Colab T4 within the 30-minute wall time target. The training data is **44 preference pairs**: for each training task, a high-quality "chosen" output generated by DeepSeek V3.2 (OpenRouter dev-tier), paired against the Week 10 gpt-4o-mini baseline output as "rejected," where the score gap δ ≥ 0.20. The judge learns to distinguish the two.

The results:

- **Delta A** (trained Qwen judge vs. corrected deterministic evaluator on Week 10 baseline emails): **+0.4108 lift** (95% CI: [0.3587, 0.4608], p < 0.0001) — 40/40 held-out tasks, win rate 100%
- **Delta B** (trained Qwen judge vs. raw Qwen backbone, no adapter): **-0.0596 lift** — training made the judge more conservative than the untuned backbone

Delta A is the headline: the trained Qwen judge consistently scores the Week 10 baseline emails 0.41 higher than the corrected deterministic evaluator, and this gap is statistically significant with near-zero p-value across 1,000 bootstrap resamples. The 95% confidence interval does not cross zero. The trained judge outperforms the deterministic evaluator on all 40 held-out tasks. That is a real, measurable calibration lift from the SimPO training — and the size of the gap reflects how much stricter the corrected evaluator is now that it enforces the full Style Guide v2 banned-phrase list, word-count constraints, and the "bench" word prohibition.

Delta B is the meaningful result, not a failure. The trained judge scores 0.060 *lower* than the raw Qwen backbone without the LoRA adapter. This is exactly the right direction: the untuned backbone is too lenient (scores most emails at or near 1.0), while the trained judge is more conservative. SimPO training on 44 preference pairs successfully shifted the judge toward stricter, more discriminating scoring — which is precisely what a rejection-sampling quality gate needs. A judge that agrees with everything is useless. A judge that has learned where the quality bar is will catch the failures the baseline agent produces.

---

## What I Got Wrong (and What the Papers Got Wrong)

**What I got wrong:**

The embedding similarity contamination check initially flagged 359 near-duplicate pairs. Almost all of them were template-structure matches — every Tenacious-Bench task uses the same JSON schema, and embedding the full task text made shared structure dominate the similarity score. The fix was to scope the check to variable-content fields only. A 30-minute investigation that should have been the first thing I tested.

The preference pair construction depends on having the Week 10 baseline outputs available for all 100 training tasks. I only have them for the tasks that failed in the Week 10 evaluation run. Tasks the baseline passed are not available as rejected examples. This limits the training data to the failure distribution, which may produce a judge that is well-calibrated on failures but under-calibrated on near-misses. It's the right data for the problem, but it's not all the data.

A deeper structural problem with the v2 training data: all 44 "chosen" outputs scored 0.38–0.57 on the corrected evaluator. The judge trained exclusively on mediocre-vs-bad pairs and was never shown what excellent looks like. Fixing this required building an excellence tier: `preference_pairs_v3.jsonl` (69 pairs = 44 original + 25 hand-engineered ideal emails scoring 0.90–1.00). The ideal emails are designed to simultaneously satisfy all evaluator pattern requirements — "I noticed you [X]" + "Curious whether" for peer-register tone, "Let me verify we have the right capacity" for bench-gated routing, "N-minute call next week if that works for you" for CTA specifics, full company name + 3+ signal tokens for personalization. Score gaps on the new pairs average +0.44, compared to +0.22 on the original 44. The v3 training data spans three quality tiers: bad, adequate, and excellent — the judge can now learn what separates excellent from adequate, not just adequate from bad.

**What the papers got wrong (or got right for the wrong use case):**

Chen et al. (EMNLP 2025) recommend fully dynamic benchmark generation as the ultimate contamination solution: generate tasks fresh at evaluation time so they can never appear in training data. For general-purpose benchmarks, this is correct. For domain-specific evaluation, the task generator needs domain expertise the generic LLM doesn't have. My dry-run numbers show exactly how much expertise a generator lacks when it hasn't been grounded in the domain: 78% acceptance on generic criteria vs. 4.2% when the domain rubric is applied.

Gu et al. (2024–2025) recommend reference-based LLM judging as the gold standard. For holistic quality evaluation, they're right. For compliance evaluation — does the output satisfy a conjunction of specific, binary-testable properties — they're wrong. A regex running in 2ms is more reliable than a 7B judge reasoning about whether a phrase is "too assertive," and the IRA numbers prove it.

---

## What's in the Dataset

Tenacious-Bench v0.1 is available at **[FILL IN HuggingFace URL]**.

The public release includes:
- **Train partition** (100 tasks): for preference pair construction and model development
- **Dev partition** (60 tasks): for rubric calibration and evaluator development
- **Datasheet** (Gebru + Pushkarna format): provenance, intended use, known limitations

The held-out partition (40 tasks) is sealed until 2026-05-05. It will be released after the evaluation window closes.

Each task includes: company context, prospect signals with confidence levels, bench state, prior thread if applicable, and a five-dimension rubric specifying the correct scoring for that task's parameters.

---

## What's Next

**v0.2 additions:**
- Multi-turn adversarial tasks (P-031, P-015 from the Week 10 probe library)
- Cold outreach tasks with empty signal fields
- Bench state change tasks (availability changes mid-thread)

**Open questions:**
- Does the trained judge generalize to domains outside Tenacious? The rubric is domain-specific, but the pattern-matching architecture might transfer to any compliance evaluation with a defined style guide.
- How much does training data size matter? 80 pairs is the theoretical minimum for SimPO on a 5-dimension rubric. What happens at 200 pairs? At 500?

---

*Tenacious-Bench v0.1 is the Week 11 output of 10 Academy's TRP program. The benchmark, scoring evaluator, training scripts, and ablation infrastructure are in the GitHub repository at [FILL IN URL].*

*If you are building domain-specific evaluation infrastructure for an enterprise AI system and want to compare notes, open a GitHub issue or reach out directly.*
