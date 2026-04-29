# Synthesis Memo: Datasheets for Datasets + Data Cards
**Sources:** Gebru et al. (2021) *Datasheets for Datasets*; Pushkarna et al. (FAccT 2022) *Data Cards: Purposeful and Transparent Dataset Documentation*
**Author:** Kidus Gashaw | **Date:** 2026-04-26

---

## Core Idea

Gebru et al. propose a standardized documentation template for datasets modeled on the component datasheet used in electronics manufacturing: seven mandatory sections (Motivation, Composition, Collection, Preprocessing, Uses, Distribution, Maintenance) that together allow a reader to determine whether a dataset is appropriate for a given use case without running experiments. The key argument is that dataset documentation is currently an afterthought, which causes silent misuse — models trained or evaluated on datasets that were never designed for their use case. Pushkarna et al. extend this with the *data card* framework: a layered documentation structure (telescopic for quick scan, periscopic for practitioner use, microscopic for auditor-level detail) that trades off readability and completeness based on the reader's role. Together the papers argue that dataset transparency is not a nicety but a prerequisite for responsible benchmark comparison.

---

## Design Choice I Disagree With

Gebru et al. recommend that the **Composition** section include "the number of instances of each type, along with counts of missing values, if any" (Section 2, Composition subsection). For most tabular or image datasets, this is straightforward and unambiguously good practice. For synthetic *evaluation* benchmarks like Tenacious-Bench, it creates a perverse incentive: if you document exact counts per (dimension × source\_mode × difficulty) cell in a public datasheet, you expose the benchmark's coverage gaps to the very model developers whose agents you are evaluating. A developer could inspect the datasheet, notice that the hard/signal\_grounding/adversarial cell contains only 8 tasks, and deliberately over-train on adversarial signal-grounding scenarios before submitting to the leaderboard.

I disagree that full cell-level counts belong in the **public** datasheet for the *held-out* partition. My practice: the public `datasheet.md` documents train and dev partition counts in full detail (dimension × source\_mode breakdowns, difficulty distributions); the held-out partition is documented only at the aggregate level (40 tasks, balanced across dimensions and modes) until the leaderboard closes. This creates a deliberate asymmetry with Gebru et al.'s transparency principle — but the principle was articulated for training datasets, not sealed evaluation benchmarks.

My own Week 11 evidence supports this: during adversarial task authoring, I found that knowing which (dimension × difficulty) cells were sparsely populated made it straightforward to craft inputs that would be over-represented at evaluation time. A developer with the same information could exploit this before the leaderboard runs. Pushkarna et al.'s microscopic layer is the correct venue for full cell counts — available to auditors under NDA, not published in the public card.

---

## How This Informed Tenacious-Bench

**From Gebru et al.:**

1. **All seven sections are present** in `datasheet.md`. Non-negotiable per the publication checklist. The absence of any section is a graded failure in this submission and in any responsible dataset release. Gebru et al.'s framing of the seven sections as a minimum rather than a maximum was important: it is a floor, not a ceiling.

2. **Maintenance section is specific**, not generic. Gebru et al. note that most datasheets treat maintenance as an afterthought ("will be updated as needed"). Tenacious-Bench specifies: v0.2 target date (Week 13), contribution protocol (GitHub Issues with task\_id and proposed correction), and the explicit condition under which the held-out partition is released (Week 11 leaderboard publication date).

3. **Uses section includes explicit NOT-for-use cases.** Gebru et al. argue this is the most commonly omitted element of any dataset documentation. Tenacious-Bench explicitly documents that the benchmark should not be used as a sole basis for production deployment decisions, and should not be used to evaluate general-purpose assistants — the rubric is domain-specific and will systematically misevaluate agents not trained on Tenacious-style outreach.

**From Pushkarna et al.:**

1. **Telescopic / periscopic / microscopic layering** is visible in `datasheet.md`. The one-paragraph telescopic summary gives reviewers a 60-second read on what the dataset is and whether it is relevant. The seven-section body is the periscopic practitioner layer, with enough detail to reproduce authoring choices. The cell-level count tables (train and dev only) constitute the microscopic layer — enough for an auditor to verify stratification without exposing held-out structure.

2. **The data card framing influenced the task schema design.** Pushkarna et al. argue that the schema itself is a documentation artifact: a structured format communicates intended use more precisely than prose. The Tenacious-Bench `schema.json` is simultaneously a JSON Schema validation contract and a documentation layer. A practitioner who reads `schema.json` understands exactly what fields are required, what values are permissible, and which features are optional — without reading the datasheet at all.

---

## Limitation

Neither paper addresses synthetic *evaluation* benchmarks specifically. Gebru et al. focus on training datasets; Pushkarna et al. on broad-use datasets from large organizations. The adversarial-partition disclosure problem I describe above has no direct guidance in either framework. The closest analog is the held-out test-set disclosure debate in NLP (SuperGLUE, BIG-Bench), which neither paper cites. Both frameworks would benefit from a dedicated treatment of sealed evaluation partitions: when transparency and evaluation integrity conflict, which takes precedence, and what partial-disclosure mechanisms (aggregate counts, stratification description without cell detail) preserve both values. That guidance does not currently exist in standardized form, which means practitioners building evaluation benchmarks must improvise — as I did here.
