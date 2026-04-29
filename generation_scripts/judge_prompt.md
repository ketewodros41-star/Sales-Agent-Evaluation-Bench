# Tenacious-Bench Judge Prompt

**Version:** 1.0
**Used by:** `judge_filter.py` (`llm_score` function, variable `JUDGE_PROMPT_TEMPLATE`)
**Model:** Qwen3-Next-80B-A3B (via OpenRouter)
**Purpose:** Pointwise quality filter for multi-LLM synthesis candidate tasks before inclusion in Tenacious-Bench.

---

## Prompt (verbatim)

```
You are a benchmark quality judge for a B2B sales outreach evaluation dataset called Tenacious-Bench.

Score the following benchmark task on three dimensions, each 1–5:
1. Input coherence: Does the company_signal make logical sense? Is the bench_summary internally consistent with the company signal?
2. Ground-truth verifiability: Can a human scorer deterministically apply the rubric from the inputs alone (without additional context)?
3. Rubric clarity: Are the expected_features (banned_phrases, must_reference_signal, required_hedges) specific enough to produce consistent labels across labelers?

Respond with ONLY a JSON object in this exact format:
{"input_coherence": <int 1-5>, "ground_truth_verifiability": <int 1-5>, "rubric_clarity": <int 1-5>, "one_line_note": "<short note>"}

Task to score:
{task_json}
```

---

## Scoring Thresholds

| Dimension | Minimum Score |
|---|---|
| input_coherence | ≥ 3 |
| ground_truth_verifiability | ≥ 3 |
| rubric_clarity | ≥ 3 |
| mean (all three) | ≥ 3.5 |

A task must clear **all four thresholds** to be included in the dataset.

---

## Dimension Calibration

**input_coherence (1–5):**
- 1: Company signal is internally contradictory (e.g., "Seed, $50M" or bench claims engineers of a role not mentioned in the signal).
- 2: Signal is plausible but missing key fields (no role count, no AI maturity score).
- 3: All required fields present; no internal contradictions.
- 4: All fields present and logically consistent (funding amount matches stage, employee count matches stage).
- 5: All fields present, consistent, and the company_signal + bench_summary together create a well-formed decision scenario.

**ground_truth_verifiability (1–5):**
- 1: The rubric cannot be applied without external information not present in the task.
- 2: The rubric is partially applicable but requires interpretation of missing context.
- 3: A trained scorer can apply all rubric dimensions from the task alone.
- 4: Rubric application is unambiguous; at least one dimension has a deterministic binary check (banned phrase, headcount commitment).
- 5: All five rubric dimensions are mechanically scoreable from the task inputs; no judgment calls required.

**rubric_clarity (1–5):**
- 1: expected_features are empty or trivially non-specific ("avoid being rude").
- 2: Some banned phrases present, but required_hedges or must_reference_signal are missing where needed.
- 3: banned_phrases and must_include_cta are present; task is scoreable.
- 4: All expected_features match the dimension; required_hedges are present for signal_grounding tasks; bench constraints for bench_fit tasks.
- 5: expected_features are tight enough that two independent scorers would reach the same label without discussion.

---

## Notes on Usage

This prompt is applied only to the **multi-LLM synthesis** partition (50 tasks, ~1,200 candidates generated). Trace-derived and programmatic tasks pass through the heuristic scorer (`heuristic_score` in `judge_filter.py`). Adversarial tasks are hand-validated and bypass this filter entirely.

The judge model (Qwen3-Next-80B-A3B) is a **different model family** from the seed generator (Claude Sonnet 4.6), implementing the preference-leakage prevention policy from Li et al. (2025).
