# Synthesis Memo: Recent Advances in LLM Benchmarks against Data Contamination
**Source:** Chen et al., EMNLP 2025 — *Recent Advances in Large Language Model Benchmarks against Data Contamination: From Static to Dynamic Evaluation*
**Author:** Kidus Gashaw | **Date:** 2026-04-30

---

## Core Idea

Chen et al. survey the contamination problem in LLM evaluation: static benchmarks are increasingly included in model pretraining corpora, causing inflated scores that measure memorization rather than capability. The paper organizes detection methods into three families — n-gram overlap (exact substring matching between benchmark and training data), embedding similarity (semantic proximity using dense retrieval), and time-shift verification (comparing model performance on pre-cutoff vs. post-cutoff data) — and argues that the field must move from static, fixed benchmarks toward dynamic evaluation where new tasks are generated at test time, preventing contamination by construction.

The key operational recommendation is the contamination-prevention design rules applied to held-out partitions: n-gram overlap below 8-gram threshold, cosine similarity below 0.85 for embedding pairs, and time-shift documentation for any task referencing datable public events. These thresholds were adopted directly in Tenacious-Bench's contamination protocol.

---

## Design Choice I Disagree With

Section 5 of the paper advocates for **fully dynamic benchmark generation** as the ultimate solution to contamination: tasks are generated fresh at evaluation time, making it impossible for training data to include them. The paper presents this as the natural endpoint of contamination-resistant evaluation design.

I disagree that dynamic generation is the right solution for domain-specific evaluation benchmarks, and my disagreement comes directly from building Tenacious-Bench.

Dynamic generation requires a reliable task generator — a system that can produce valid, rubric-consistent, non-trivial tasks on demand. For Tenacious-Bench, generating a valid task requires: knowledge of the Tenacious style guide (23-item banned-phrase list, tone register expectations), familiarity with the bench inventory structure (capacity constraints, skill availability logic), understanding of the five failure dimensions from the Week 10 probe library, and calibration on what constitutes a "hard" vs. "easy" signal confidence scenario. When I ran the synthesis pipeline without this grounding, the LLM quality filter accepted 78% of Claude-generated candidates in dry-run mode — because the heuristic filter had no domain knowledge. When I switched to a Qwen LLM judge applying the actual Tenacious rubric, acceptance dropped to 4.2%. The gap between those two numbers is the domain expertise that a generic dynamic generator does not have.

Dynamic generation at evaluation time would use a generator with no access to the Week 10 probe library, no training on the failure taxonomy, and no knowledge of which scenarios constitute adversarial edge cases. It would generate plausible-looking tasks that pass surface coherence checks but miss the specific failure modes the benchmark is designed to detect. Contamination-free tasks that don't measure what we care about are not an improvement over contaminated tasks that do.

Chen et al.'s argument is correct for general-purpose benchmarks — math reasoning, reading comprehension, coding tasks — where the task generator can be validated against a known ground truth. It does not transfer to domain-specific evaluation where the domain constraints themselves are the difficult part to encode.

---

## How This Informed Tenacious-Bench

**Adopted directly:**
1. **N-gram threshold of 8-gram overlap** — applied in `contamination_check.py` with a zero-violation policy on full 8-gram matches. The 8-gram threshold follows Chen et al.'s finding that 8-gram overlap is the minimum granularity at which contamination reliably signals memorization rather than coincidental phrasing.
2. **Embedding similarity threshold of 0.85** — adopted as the upper bound, with the caveat documented in `contamination_check.json` that template-based benchmarks require scoping to variable-content fields rather than full task text.
3. **Time-shift documentation** — all 40 held-out tasks carry explicit signal timestamps. No task uses a generic "recent" reference.

**Adapted:**
The paper recommends static-to-dynamic migration as a long-term strategy. Tenacious-Bench adopts a "static but versioned" alternative: v0.1 is the current fixed benchmark; v0.2 will add new trace-derived tasks from the trained agent's held-out inference runs, refreshing the benchmark with new failure geometries without requiring dynamic generation at evaluation time. This preserves reproducibility (a stranger can re-run the same benchmark) while addressing the contamination concern through versioned refreshes rather than on-the-fly generation.

---

## Limitation

The paper focuses on contamination from benchmark tasks entering training corpora. Tenacious-Bench faces a different contamination direction: benchmark structure entering generation — knowing which (dimension × difficulty) cells are sparse allows adversarial over-training before leaderboard submission. Chen et al. do not address this evaluation-design contamination vector. The mitigation (withholding held-out cell-level counts from the public datasheet) was improvised from first principles, not derived from any framework the paper provides.
