# Synthesis Memo: A Survey on LLM-as-a-Judge
**Source:** Gu et al. (2024–2025) — *A Survey on LLM-as-a-Judge*
**Author:** Kidus Gashaw | **Date:** 2026-04-30

---

## Core Idea

Gu et al. survey the emerging practice of using LLMs to evaluate other LLMs, covering four evaluation paradigms (pointwise scoring, pairwise comparison, reference-based, and reference-free), common failure modes (position bias, verbosity bias, self-preference), calibration strategies, and mitigation techniques. The central argument is that LLM judges are practical proxies for human evaluation at scale, provided the judge is calibrated against a known ground truth and the evaluation protocol is designed to minimize systematic bias. The paper recommends reference-based evaluation — providing gold-standard outputs to the judge — as the highest-fidelity paradigm, because it anchors the judge's scoring to a concrete target rather than requiring the judge to reconstruct the evaluation standard from a rubric alone.

---

## Design Choice I Disagree With

Section 3.2 recommends **reference-based evaluation** as the primary quality signal, arguing that judges achieve higher agreement with human annotators when given a reference output to compare against. The paper presents this as the gold standard toward which practitioners should default.

I disagree that reference-based evaluation is appropriate for the Tenacious-Bench scoring evaluator, and the disagreement is structural, not a matter of preference.

For reference-based judging to work, you need reference outputs — examples of what a correct answer looks like. For the majority of Tenacious-Bench tasks, no reference output exists. The benchmark was built from four authoring modes: trace-derived tasks where the ground truth is the rubric, not a corrected email; programmatic tasks defined by parameter constraints, not by a canonical correct response; synthesis tasks where "correct" means satisfying five independent rubric dimensions simultaneously; and adversarial tasks specifically designed so that an LLM could not reliably generate a gold standard without already knowing the failure mode being probed. In this setting, the rubric IS the reference — and the judge's job is to apply the rubric mechanically, not compare to a template.

The practical implication is visible in `scoring_evaluator.py`. Four of the five dimensions are fully deterministic: banned phrase detection, CTA pattern matching, headcount commitment patterns, and routing language detection require no judge call at all. The fifth — personalization depth — uses token matching against extracted signal fields. None require a reference output. The IRA results confirm this is the right design: the two most mechanically-implemented dimensions (cta_quality κ=0.853, signal_grounding κ=0.857) are the most reliably scored. The one dimension that came closest to requiring reference-style judgment (personalization_depth κ=0.682) needed a rubric revision specifically to make it more mechanical — to reduce the judgment load, not increase it.

Gu et al.'s recommendation is correct for tasks where the quality criterion is holistic (is this a good translation? is this a well-reasoned argument?). It is wrong for tasks where the quality criterion is a conjunction of specific, binary-testable properties. The Tenacious rubric is the latter: an email either contains a banned phrase or it does not; it either references a specific signal token or it does not; it either has a forced-booking CTA or it does not.

---

## How This Informed Tenacious-Bench

**Judge design for dataset construction (not for evaluation):**
The LLM judge role in Tenacious-Bench is confined to `judge_filter.py` during *dataset construction*, not during *evaluation*. Gu et al.'s survey is the reference for this use case — pointwise scoring of candidate tasks on coherence, verifiability, and rubric clarity before they enter the dataset. Here, reference-free judging is appropriate because we are assessing task quality (is this a well-formed task?), not output quality (is this a correct response?). The judge applies a rubric to the task structure, not to a candidate output.

**Model rotation policy:**
Gu et al.'s discussion of self-preference bias (Section 4.3) directly informed the model rotation policy: Claude Sonnet 4.6 generates candidate tasks; Qwen judges them. A Claude judge evaluating Claude-generated tasks would inflate acceptance rates — exactly what the 78% dry-run heuristic acceptance vs. 4.2% Qwen LLM acceptance demonstrates.

**Pointwise over pairwise for quality filtering:**
Gu et al. find that pairwise comparison is more consistent than pointwise scoring for holistic quality judgments. For the quality filter — scoring coherence, verifiability, and rubric clarity each 1–5 with documented thresholds — pointwise is more appropriate because the evaluation criteria are independent. A task can be highly coherent but have low rubric clarity. Collapsing this into a pairwise preference would lose the per-dimension signal.

---

## Limitation

Gu et al. focus almost entirely on evaluating natural language quality — fluency, coherence, helpfulness, accuracy. The survey does not address the case where the evaluation criterion is a conjunction of domain-specific hard constraints (presence/absence of specific phrases, compliance with a documented style guide, adherence to a capacity policy). This is the Tenacious use case, and it is not a minor variation. When the rubric consists of binary-checkable constraints rather than holistic quality assessments, LLM judging is largely unnecessary — a deterministic evaluator is both faster and more reliable. The survey would benefit from a section distinguishing "quality evaluation" (where LLM judges add value) from "compliance evaluation" (where deterministic checks dominate). That distinction is central to the design of Tenacious-Bench's scoring architecture and is absent from the paper's framework.
