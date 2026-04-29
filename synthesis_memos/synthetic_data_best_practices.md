# Synthesis Memo: Best Practices and Lessons Learned on Synthetic Data for Language Models
**Source:** Liu et al., COLM 2024
**Author:** Kidus Gashaw | **Date:** 2026-04-26

---

## Core Idea

Liu et al. survey the synthetic data landscape across four paradigms: data augmentation, knowledge distillation from stronger models, self-improvement via model-generated feedback, and multi-agent collaborative generation. The central thesis is that the quality of synthetic data depends more on *generation diversity* and *quality-filter design* than on raw volume. The paper distinguishes between "quantity-scaling" approaches (generate many, filter loosely) and "quality-first" approaches (generate fewer, filter strictly), arguing that for alignment and evaluation tasks, quality-first consistently dominates once you have roughly 1,000 representative seed examples. Their contamination section is the most operationally relevant finding for benchmark construction: static synthetic benchmarks generated from the same model family as the agent being evaluated are prone to spurious capability inflation — the agent is essentially tested against a slightly different draw from its own training distribution.

---

## Design Choice I Disagree With

Section 4.2 of the paper recommends using a **single high-quality teacher model** to generate synthetic evaluation data, then filtering with a separate judge from the same model family. The justification is consistency: a single model produces stylistically coherent data that is easier to judge, and same-family judging reduces noise from style disagreements between generator and evaluator.

I disagree with this recommendation, and my disagreement is grounded in what I observed while building Tenacious-Bench — not primarily in theory.

During the multi-LLM synthesis pass, Claude Sonnet 4.6 generated seed tasks and Qwen3-Next-80B-A3B judged them. When I ran the heuristic judge filter in dry-run mode first (no API calls, using the rule-based approximation in `judge_filter.py`), acceptance was near-universal — roughly 78% of candidates passed. When I switched to the Qwen LLM judge, acceptance fell to 4.2% (50 of 1,200). The gap is not noise. Inspecting rejected tasks, the pattern was consistent: Claude-generated candidates used hedge language and professional register that looks correct to a model trained on similar data — phrases like "I understand you build in-house, but..." — which the Tenacious style guide explicitly bans ("`i apologize`", "`i understand your concern`" are on the banned-phrase list). A same-family judge would score these as stylistically appropriate because they are, by general standards. A Qwen judge, applying the rubric without stylistic bias toward Claude's defaults, flagged them correctly.

Week 10 corroborates this at the trace level. `tr_dev_009` (RETAIL-022, reward=0) is logged as "assertive language on a low-confidence recommendation" — a policy_violation in τ²-Bench's taxonomy, but not distinguishable from a tone violation by any generic judge. The eval log note explicitly maps this to "Tenacious signal over-claiming." Under evaluation run `run_dev_baseline_30x3_20260423` (21.6% pass@1, τ²-Bench binary reward), the failure was recorded but the reason — confident phrasing on a weak evidential basis — was invisible to the reward function. A same-family Claude judge scoring Claude-generated candidates with similar confident phrasing would make the identical error: it would rate assertive professional register as appropriate, because by general LLM standards it is. A Qwen judge applying the Tenacious rubric literally would catch it. The cross-family routing is a direct response to this observed failure pattern.

Liu et al.'s same-family consistency argument is correct for general-purpose quality — coherence, fluency, relevance. It fails specifically when the rubric includes domain-specific hard-binary criteria that conflict with the generator's stylistic defaults. Li et al. (2025) *Preference Leakage* later formalised this as a systematic effect, but the observation preceded the paper in my own pipeline.

My practice: **Claude Sonnet 4.6 generates seeds; Qwen3-Next-80B-A3B judges them.** This adds latency and cost but produced a judge filter that catches the exact failure modes the Week 10 probe library documented.

---

## How This Informed Tenacious-Bench

Three specific design decisions follow directly from Liu et al.:

**1. Four authoring modes simultaneously** (trace-derived, programmatic, multi-LLM synthesis, adversarial) rather than a single synthesis pipeline. Liu et al.'s Table 3 shows that diversity across generation strategies improves held-out transfer more than scaling within a single strategy. Each of the four modes in Tenacious-Bench produces structurally different inputs: trace-derived tasks encode real failure geometries from Week 10; programmatic tasks stress parameter boundaries; synthesis tasks explore hard-case combinations no real trace hit; adversarial tasks are hand-crafted to defeat specific probe weaknesses. The structural diversity is by design, not by accident.

**2. Quality-first filtering for the multi-LLM synthesis partition.** We generated roughly 1,200 candidate tasks and filtered to 50 — a 4.2% acceptance rate. This is aggressive by industry standards, but consistent with Liu et al.'s finding that loose filters produce task-level dataset contamination: tasks where the rubric and the input don't actually align, making the rubric unapplicable or trivially satisfied. The three-dimension judge filter (coherence ≥ 3, verifiability ≥ 3, rubric clarity ≥ 3, mean ≥ 3.5) enforces that every accepted task is scoreable by a human without additional context.

**3. Adversarial tasks excluded from the judge filter.** Liu et al. note that LLM quality-filters tend to reject adversarial tasks because they look "incoherent" to a model that does not understand the specific failure mode being probed. Our 30 adversarial tasks are hand-validated and intentionally excluded from `judge_filter.py`, consistent with Liu et al.'s recommendation to trust human authorship for edge-case inputs.

---

## Limitation

Liu et al. focus primarily on *training* synthetic data, not *evaluation* synthetic data. The contamination risks differ: for training data, contamination inflates benchmark performance; for evaluation data, contamination inflates the benchmark's apparent coverage of failure modes — a subtler and more dangerous problem. The distinction is not developed in Section 4.2, and the recommendation to filter for quality applies differently in the two settings. For evaluation data, a task that "looks coherent" to a general-purpose judge may still fail to discriminate between agents that genuinely differ on the target dimension. Tenacious-Bench addresses this by adding rubric-clarity as a third filter dimension, but Liu et al. provide no direct guidance for this case.
