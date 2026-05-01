

## Ground Truth
Building the Sales Evaluation Bench and Aligning the Conversion Engine
## Summary
In Week 10 you built the Conversion Engine for Tenacious — the system that finds prospects, grounds
outreach in public signal, qualifies leads, and books discovery calls. The Tenacious executive team has
now asked a question τ²-Bench retail cannot answer: how do we know this works for our business, our
voice, our segments, our bench?
This week you build the answer. You audit what existing benchmarks miss for Tenacious-style B2B sales
work. You construct a sales-domain evaluation dataset using multi-LLM synthesis and judge filtering on a
small seed corpus. You train a small model component — a generation adapter, a judge, or a process
scorer — that lifts your Week 1 0 agent on a Tenacious-specific failure mode. And you ship the work
publicly: a HuggingFace dataset with full datasheet, a trained adapter or judge with model card, a
technical blog post under your name, and an artifact contributed to the open evaluation community.
The hardest engineering problem of the week is the dataset, not the training run. Tenacious has a small
seed corpus, no historical labeled prospects, and no robust dataset to hand you. You will build the corpus
from limited material using techniques drawn from recent literature on synthetic data, LLM-as-a-judge,
and contamination-resistant evaluation. The training run, by design, is short and cheap.
## The Shift From Week 10
You will not re-run τ²-Bench retail this week. Re-running it costs roughly $5–8 per pass and the score
does not move; that compute funds dataset authoring and training instead. If you produced a τ²-Bench
score in Week 10, reuse it as informational reference. If you did not, your Week 10 agent's score on the
new Tenacious-Bench is your baseline — that is sufficient.
Your Week 10 work is the seed regardless of which deliverables you completed:
Week 10 artifact Week 11 use
trace_log.jsonl Trace-derived task authoring; SFT/DPO training data.
probe_library.md Adversarial task seeds; each probe expands into 3–8 task
variants.
failure_taxonomy.md Schema dimension structure for Tenacious-Bench.

Hiring signal brief and competitor gap
brief outputs
Task input templates that require grounding to score
correctly.
Tenacious style guide adherence
checks
Alignment objective for the trained component.
Trainees who completed more of Week 10 have a slightly larger reference corpus. Trainees with thinner
Week 10 output still have everything they need: the new benchmark is built from your agent's behavior
on the Tenacious workflow, not from anyone else's measurements.
## Required Reading
Read the four common papers before Day 2. Read your path-specific papers before Day 4. Each reading
produces a one-page synthesis memo committed to your repo. The memo is graded on whether you can
disagree with the paper on a specific design choice and justify the disagreement against your own
evidence — not on whether you can summarize.
Common to all paths
- Best Practices and Lessons Learned on Synthetic Data for Language Models (Liu et al., COLM 2024)
— the operational reference for the dataset-authoring decisions you make in Acts I–II.
- Datasheets for Datasets (Gebru et al., 2021) and Data Cards: Purposeful and Transparent Dataset
Documentation (Pushkarna et al., FAccT 2022) — the documentation standard your published
dataset must meet. Read both; Data Cards extends Datasheets with modular layered detail
(telescopic, periscopic, microscopic).
- Recent Advances in Large Language Model Benchmarks against Data Contamination: From Static
to Dynamic Evaluation (Chen et al., EMNLP 2025) — the contamination-prevention design rules
you must apply to your held-out partition.
- A Survey on LLM-as-a-Judge (Gu et al., 2024–2025, latest revision) — the judge-design reference
for your scoring evaluator and your training-data quality filter.
Path A — supervised fine-tuning of a generation component
- Tülu 3: Pushing Frontiers in Open Language Model Post-Training (Lambert et al., 2024) — the
modern open SFT recipe end to end.
- LIMA: Less Is More for Alignment (Zhou et al., NeurIPS 2023) — the empirical case for small
high-quality datasets, which governs your authoring strategy.
- Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing (Xu et
al., 2024) — a 2024 technique for self-generating instruction data that you will adapt for
Tenacious-style outreach drafts.

Path B — preference-tuned judge or critic
- Direct Preference Optimization (Rafailov et al., NeurIPS 2023) — the foundational algorithm.
- SimPO: Simple Preference Optimization with a Reference-Free Reward (Meng, Xia, and Chen,
NeurIPS 2024) and ORPO: Monolithic Preference Optimization without Reference Model (Hong,
Lee, and Thorne, EMNLP 2024) — modern reference-free variants that frequently outperform DPO
at lower cost. Pick one for your training and justify the choice.
- Prometheus 2: An Open-Source Language Model Specialized in Evaluating Other Language Models
(Kim et al., 2024) — the canonical reference for a small open judge model trained from
preferences.
- Preference Leakage: A Contamination Problem in LLM-as-a-Judge (Li et al., 2025) — what to avoid
when one LLM generates data and another grades it.
Path C — process reward model
- Let's Verify Step by Step (Lightman et al., 2023) — the canonical PRM paper, including the
PRM800K data construction pattern you will adapt.
- DeepSeek-Math: Pushing the Limits of Mathematical Reasoning (Shao et al., 2024) —
process-reward-driven inference at small scale, with code.
- Source2Synth: Synthetic Data Generation and Curation Grounded in Real Data Sources (Lupidi et
al., 2024) — directly applicable to converting your Week 10 traces into stepwise process labels.
Optional but useful for any path
- LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021) and the Unsloth Qwen 3.5
fine-tuning guide, the training mechanics every path uses.
- Constitutional AI: Harmlessness from AI Feedback (Bai et al., 2022) — preference-data generation
without humans in the loop.
## Inputs You Have, Inputs You Build
Tenacious does not have a robust dataset. You will build the dataset from a small seed corpus and
engineering effort, not by asking Tenacious for more.
What you have
- Tenacious style guide v2 (with 12 hand-labeled "good" and 12 hand-labeled "bad" outreach
drafts).
- Tenacious sales deck and three redacted case studies.
- Tenacious bench summary v2.
- Tenacious pricing sheet.

- Five synthetic discovery-call transcripts.
- Public Crunchbase ODM 1,001-company sample.
- Public layoffs.fyi CSV.
- Your Week 10 trace_log.jsonl (your Conversion Engine's actual outputs).
- Your Week 10 probe library (your documented failure modes).
What you build from these inputs
- A 200–300 task evaluation dataset across the dimensions of Tenacious-specific failure.
- A training partition (50%) with no overlap with the held-out, a public dev partition (30%), and a
sealed held-out (20%).
- A machine-verifiable scoring evaluator for every task.
- A datasheet documenting motivation, composition, collection, preprocessing, uses, distribution,
and maintenance.
- A trained LoRA adapter or judge model that lifts your agent on a Tenacious-specific failure mode.
- A blog post and a community engagement artifact.
The challenge is converting limited inputs into a publishable benchmark using engineering and creative
use of multiple LLMs, not by waiting for more data.
## Data Construction Approach
Modern benchmark construction at small-data starting points uses a routed multi-LLM pipeline. Reading
the synthesis memos before Day 2 is what equips you to make the design choices below.
The four authoring modes
You author tasks in four modes simultaneously, weighted by the failure modes named in your audit:
## Mode Share How
Trace-derived ≈30% Real Week 10 traces, redacted, restructured into (input,
candidate output) pairs with rubric-graded ground truth.
Highest fidelity because they reflect real distributional
behavior. Cost: free (already in your repo).
Programmatic with
parameter sweeps
≈30% Templates with structured slots — company size, segment,
requested headcount, stack, bench state, AI-maturity score,
signal confidence — populated by combinatorial expansion.
A single "bench over-commitment" probe becomes 20 tasks

by varying inputs. Cost: small dev-tier LLM calls for surface
variation.
Multi-LLM synthesis ≈25% Generate hard cases by routing across LLM families with
different strengths, then quality-filter with a judge. A
frontier model (Claude or GPT-class) authors the 30–50
hardest seeds anchored to your Week 10 failure taxonomy; a
cheap dev-tier model (Qwen3-Next, DeepSeek V3.2)
generates bulk variations. Pools are deduplicated and
judge-filtered. Pattern follows Magpie-style self-instruction
with explicit grounding in your Week 10 evidence.
## Hand-authored
adversarial
≈15% The hardest 30–50, written by you to specifically defeat your
Week 10 system on edge cases the synthesis pipeline
misses. These carry the most originality weight at grading.
Cost: human time only.
LLM-as-a-judge for quality filtering
Every generated task passes a judge filter before entering the dataset. The judge is itself a small pipeline:
- Pointwise scoring on three dimensions: input coherence, ground-truth verifiability, and
rubric-application clarity. Score 1–5 each. Threshold for inclusion is documented per dimension.
- Pairwise comparison when two synthesis paths produce similar tasks — pick the more diagnostic
one.
- A separate cheap-model judge for high-volume filtering, with the eval-tier model used only to
spot-check 50 sampled tasks for calibration.
To avoid preference leakage (Li et al., 2025): never use the same model to generate and judge the same
task. Rotate between model families. The rotation policy is documented in methodology.md.
Contamination prevention
Three checks before any task enters the held-out partition:
- *N-gram overlap* between any held-out task and the training partition: less than 8-gram overlap
on input fields.
- *Embedding similarity* between held-out and training tasks using a cheap embedding model:
cosine similarity below 0.85 for any pair.
- *Time-shift verification* for any task referencing public data — the underlying signal must be from
a window the trainee can document, not a generic placeholder.

The contamination-check script is itself a Week 11 deliverable and runs as part of the dataset publication
pipeline.
Inter-rater agreement
You hand-label a 30-task subset, then re-label it 24 hours later without looking at your first labels.
Agreement under 80% on any rubric dimension triggers a rubric revision. This is the single most
important quality signal for your scoring evaluator. The agreement matrix is committed to
methodology.md.
## The Training Paths
Pick one path on Day 1. Justify the pick against your Week 10 evidence in methodology.md. The
justification is itself an observable: a Path A pick when your Week 10 evidence points to inconsistency
loses methodology credit, because the failure mode dictates the treatment.
Path What you train When to pick Typical training
cost
A — SFT a
generation
component
Small backbone (Qwen 3.5 0.8B, Qwen
3.5 2B, or Qwen 3.5 4B) with LoRA,
replacing one piece of your Week 10
agent: brief-to-email composer,
signal-grounded outreach generator,
or tone-preservation rewriter.
Week 10 failures were
generation-quality
failures — tone drift,
formulaic phrasing,
weak grounding
language.
$0 on free Colab
T4 via Unsloth,
or $1–3 on
RunPod
community
## 4090.
## B —
DPO/SimPO/
ORPO a judge
or critic
Small classifier or preference scorer
trained to grade agent outputs on
Tenacious dimensions. Deployed in
production as a rejection-sampling or
rollback layer in front of your Week 10
generator.
Week 10 failures were
inconsistency failures —
the agent gets it right
most of the time but
cannot tell when it is
wrong.
$0 on Colab T4
(SimPO/ORPO
are
reference-free
and lighter than
DPO), or $2–4
on RunPod.
C — Train a
process
reward model
A scorer that evaluates intermediate
trajectory steps mid-conversation.
Used at inference time for tree-search,
rollback, or human-escalation
decisions.
Week 10 failures were
trajectory failures —
locally reasonable
choices that compound
into bad endings.
$2–5 on RunPod
community
## 4090. Data-prep
is the
bottleneck, not
training.

Path B is the most production-relevant for the FDE engagements TRP1 graduates will see in their first
roles. Path A is the most direct teacher of training mechanics. Path C contributes the most to long-term
FDE depth on agent reliability work but is the hardest for the data-prep step.
Methodology rationale must cite at least three Week 10 trace IDs and at least two of your read papers.
## Production Stack
The stack adds three capabilities to Week 10's: training, model hosting, and dataset hosting. Email,
telephony, CRM, and calendar from Week 10 remain available for trace generation but are not the focus.
## Layer Choice Notes
Training compute
(default, free)
Unsloth on Google Colab
## T4
Free 16 GB VRAM. Qwen 3.5 0.8B, Qwen 3.5 2B,
and Qwen 3.5 4B all fit. SFT, DPO, SimPO, ORPO,
and GRPO supported. Qwen3.5 Fine-tuning
## Guide | Unsloth Documentation.
Training compute
(paid alternative)
RunPod community-cloud
4090 (~$0.34/hr) or A40
## (~$0.39/hr)
Use only if Colab session caps disrupt your run.
Cap at $5.
Training framework Unsloth (preferred) or
HuggingFace TRL
SFT, DPO/SimPO/ORPO, and reward modeling
all supported. Unsloth is approximately 1.5×
faster and uses ~50% less VRAM than vanilla
## TRL.
Adapter library PEFT with LoRA LoRA only. Full fine-tunes will not fit in budget.
Backbone Qwen 3.5 0.8B, Qwen 3.5
2B, or Qwen 3.5 4B
LoRA adapter only; do not merge unless
required for inference. Pin the version in
requirements.txt.
Dataset hosting HuggingFace Hub Dataset published with datasheet. License
CC-BY-4.0 unless you have a specific reason.
Model hosting HuggingFace Hub Model card required. Publish the LoRA adapter
only, not the merged backbone.
Eval-tier judge
model
Claude Sonnet 4.6 or GPT-5
class via OpenRouter
Sealed-slice scoring only.

Dev-tier judge
model
OpenRouter cheap tier
(Qwen3-Next-80B-A3B or
DeepSeek V3.2)
Iteration during dataset authoring and
training-data preparation.
Synthesis-LLM
router
OpenRouter Multi-model generation as described in Data
## Construction Approach.
Observability Langfuse Reused from Week 10. Tag training-run IDs and
ablation conditions.
## Synthetic-prospect
rig
Carried over from Week 10 Use sparingly. Trace generation is supplemental
this week, not the work.
If you have not used Unsloth before, check it here Qwen3.5 Fine-tuning Guide | Unsloth Documentation.
Run it on Day 0 — you will not have time to debug compute on Day 5.
## Cost Discipline
The week's compute envelope is $10 per trainee. Most trainees will spend less. The default training path
is free.
Bucket Budget What it pays for
Dataset authoring
(cheap dev-tier LLM
calls for synthesis,
dedup, judge filtering)
$3–5 OpenRouter cheap tier on Days 2–3. Cap individual model
calls and log every charge.
Training $0–5 Free if you use Unsloth on Colab T4. Use RunPod only if
Colab session limits force you to.
Held-out evaluation $2–3 Eval-tier model on the sealed slice only, three to four
passes maximum.
Reserve $1–2 Bug fixes, re-runs, late-week probe additions.
Two non-negotiable rules:
- *No τ²-Bench retail validation runs.* Your Week 10 score is reused if you have one; the new
Tenacious-Bench is your primary baseline if you don't. Spending on re-running an established
measurement is graded as a cost-discipline failure on the Pareto observable.

- *No eval-tier model on Days 2–3.* Iteration during dataset authoring uses dev-tier models
exclusively. A trainee who burns $5 of eval-tier API on Day 3 dedup will not have budget left for
ablation.
The cost log is itself a graded artifact. Every API and compute charge is recorded with timestamp, bucket,
and purpose.
## Day 0 — Pre-flight Checklist
Roughly four hours. The readiness review on Day 1 morning confirms each item.
Item What done looks like
HuggingFace account and access token Account created, write token generated, environment
variable set in your repo.
Google Colab account Tested. You have connected to a T4 runtime at least once.
RunPod account (optional) Account created with free credits applied. Required only if
you anticipate exceeding Colab session caps.
Unsloth starter notebook runs The provided unsloth notebooks completes a 5-task dummy
LoRA run end to end (fp16 mixed precision on Colab T4, bf16
on RunPod 4090 or Colab Pro L4) and pushes the adapter to
your HuggingFace account. The kernel compile on the first
run takes 6 to 10 minutes on T4 and is expected. QLoRA
4-bit is not used; per the Unsloth Qwen 3.5 guide, 16-bit
LoRA is the recommended path, with the precision following
the GPU's native support.
## .
Local environment Python 3.11+, transformers, peft, trl, datasets, accelerate,
bitsandbytes installed. python -c "import trl, peft;
print(trl.__version__, peft.__version__)" succeeds.
Week 10 artifacts inventoried trace_log.jsonl, probe_library.md, failure_taxonomy.md,
agent source — all confirmed present and parseable.
Schema starter reviewed Tenacious-Bench schema starter cloned. You can validate
one dummy task against the JSON schema.

First common reading complete One of Best Practices on Synthetic Data, Datasheets,
Contamination Survey, or LLM-as-a-Judge Survey finished.
Synthesis memo at draft v0.
Cost tracking set up Spreadsheet or simple log. Every API and compute charge
recorded with timestamp and bucket.
Path declaration filed A, B, or C committed in writing in methodology.md, with one
paragraph of preliminary justification citing two Week 10
trace IDs.
OpenRouter account and key Account created, key set in environment. Tested with one
cheap-tier call.
The Five-Act Loop
The shape carries over from Week 10 for cohort consistency. Each act now centers on dataset and
training rather than agent and probes.
Act I — Audit and Schema Design
## Day 1.
Write a 600-word audit memo answering one question: what does τ²-Bench retail (or any public
benchmark) fail to grade about Tenacious-specific behavior, and what does your Week 10 evidence prove
about that gap? The audit must reference at least eight probe IDs from your Week 10 library and at least
five real trace examples by trace ID.
From the audit, design the Tenacious-Bench v0.1 schema. The schema is machine-verifiable: a script
reads a task plus an agent output and returns a numerical score with no human in the loop. This is the
binding design constraint, and most Day 1 effort goes into making the rubric mechanically gradable.
A rubric that says "the email should sound on-brand" is not yet a benchmark. A rubric that says "the
email contains zero of these 23 banned phrases AND references at least one signal from the supplied
brief AND ends with a calendar link AND scores ≥ 4/5 on the LLM-judge for each of the five Tenacious
tone markers" is.
Schema must include input fields (the prospect's hiring signal brief, bench summary, prior thread),
candidate output, ground-truth fields where applicable, the scoring rubric, and difficulty stratification.
*Deliverables:* audit_memo.md (max 600 words), schema.json with three example tasks,
methodology.md draft including path declaration, scoring_evaluator.py running against three hand-built
dummy tasks.

Act II — Dataset Authoring
Days 2 and 3.
Author 200–300 tasks across the dimensions named in your audit, using all four authoring modes from
the Data Construction Approach. Apply the LLM-as-a-judge filter to every generated task. Record each
task's source mode in metadata (trace-derived, programmatic, multi-LLM synthesis, hand-authored).
Partitioning is sealed held-out (20%), public dev (30%), training partition (50%). Run all three
contamination checks before sealing the held-out. The contamination report is committed alongside the
dataset.
Hand-label 30 tasks against your rubric, then re-label them 24 hours later. If agreement is below 80% on
any dimension, revise the rubric and relabel. Record the agreement matrix in methodology.md.
Write the datasheet. Three to five pages. Cover all seven Gebru sections plus Pushkarna's data-card
layered detail (telescopic, periscopic, microscopic). The datasheet is the artifact that converts a JSON file
into a benchmark.
*Deliverables:* tenacious_bench_v0.1/ with three partitions, datasheet.md, generation_scripts/ (with
seed counts, model routes, and judge-filter logs), contamination_check.json, inter_rater_agreement.md.
Act III — Method Selection and Training Data Preparation
## Day 4.
Read your path-specific papers and complete those synthesis memos. Then convert the training partition
of Tenacious-Bench into the format your path needs:
- *Path A:* input/output pairs in chat-template format, filtered by quality score from your evaluator.
Aim for 1,000–3,000 high-quality pairs after filtering — LIMA shows quality dominates quantity at
this scale.
- *Path B:* preference pairs (chosen, rejected) constructed from probe-triggered failures (rejected)
versus corrected outputs (chosen). Corrections come from your Week 10 hand-fixes plus, where
needed, dev-tier model rewrites that pass your evaluator. Apply preference-leakage prevention by
using a different model family for chosen-rewrites versus your judge.
- *Path C:* step-level annotations on multi-turn trajectories from your Week 10 trace pool. Label
step correctness from final-turn outcome plus rubric scoring at each step. Source2Synth provides
the annotation pattern.
Most of Day 4 goes into data preparation. The training run is short the next day. Bad training data is the
most common reason a LoRA run fails to lift on the held-out, and it is invisible until the ablation table
comes back flat.

*Deliverables:* training_data/ formatted for your path, methodology_rationale.md (one page, citing
Week 10 evidence and at least two of your read papers), contamination check passed against held-out
and dev partitions.
Act IV — Train, Ablate, Measure
Days 5 and 6.
Day 5 morning: one core training run. LoRA on the pinned backbone with the hyperparameters from the
Unsloth starter notebook. Log training loss and validation curves. Total wall time should be 30 to 90
minutes; if it is not converging by 30 minutes, kill it and check your data — do not throw more compute
at it.
Day 5 afternoon and Day 6: ablations and held-out evaluation.
- *Delta A.* Trained model versus your Week 10 baseline on Tenacious-Bench held-out. Must be
positive with 95% CI separation, p < 0.05 on a paired bootstrap.
- *Delta B.* Trained model versus a prompt-engineered version of the same intervention on the
same backbone, no training. Tests whether training actually beat what a careful prompt could do.
Many Week 11 interventions will fail Delta B; that is a legitimate, publishable finding and goes in
the blog honestly.
- *Delta C.* Trained model versus your Week 10 score on published τ²-Bench retail held-out, only if
you have your Week 10 score on file. Tests whether your improvement is Tenacious-specific or
general. Informational only — no re-running of τ²-Bench this week, only reusing existing numbers.
- *Cost-Pareto.* Per-task cost and latency with the trained component versus without. A 3
percentage point lift that triples cost is graded against a 2-point lift that holds cost flat.
Each ablation is one held-out pass at the eval-tier model. Three to four passes total. Sealed-slice scores
are written to ablation_results.json and the raw scoring traces to held_out_traces.jsonl.
*Deliverables:* ablation_results.json, held_out_traces.jsonl, model_card.md if Path A or C,
training_run.log with hyperparameters and loss curves.
Act V — Publish and Engage
## Day 7.
Three artifacts ship publicly. The publication itself is the act.
*HuggingFace dataset.* tenacious_bench_v0.1, with datasheet, license, baseline scores from your Week
10 agent, top-of-leaderboard target, and a quickstart example a stranger can run in ten minutes. Indexed
under your handle.

*HuggingFace model (Path A or C).* LoRA adapter only, with a complete model card: backbone, training
data partition, hyperparameters, intended use, limitations, evaluation results.
*Technical blog post.* 1,200–2,000 words, on the HuggingFace blog (community submission), your
personal site, or Substack. Structure: the gap (what existing benchmarks miss for Tenacious-style sales
work, with evidence), the audit method (how you found it), the dataset (how you built it, with hard
design choices named — multi-LLM routing, judge-filter calibration, contamination protocol), the training
experiment (path, paper foundations, what worked, what didn't, including failed Delta B if applicable),
the honest result (lift with confidence intervals), what is next.
*Community engagement.* One of the following:
- A GitHub issue or discussion on the τ²-Bench repo presenting your Tenacious-specific gap finding
and linking the new dataset.
- A submission to NeurIPS Datasets and Benchmark, ICLR Tiny Papers, or a posting on the EleutherAI
Discord, LMSYS, or Allen AI community boards.
- A pull request to a related open benchmark project (BIRD-Critic, AgentBench, ToolBench) with a
complementary contribution.
Most trainees take the GitHub-issue route, which is achievable and high-signal. The strongest two or
three submissions per cohort target the workshop route.
The two-page memo to the Tenacious CEO and CFO. Page 1: the decision (what was built, headline lift
number with CI, cost per task delta, what should change in the production deployment). Page 2: the
skeptic's appendix (failure modes the new bench still does not capture, public-signal lossiness in your
ground truth, one honest unresolved failure, a kill-switch trigger).
*Deliverables:* HuggingFace dataset URL, HuggingFace model URL if applicable, blog post URL,
community-engagement evidence (issue link, submission confirmation, or merged PR), memo.pdf,
evidence_graph.json, README.md.
Public-Artifact Quality Bar
Public artifacts have your name on them permanently. Before any artifact goes public, the publication
checklist must be passed.
Check Pass condition
Datasheet present All seven Gebru sections have non-stub content; layered detail per
Pushkarna where relevant.

License correct CC-BY-4.0 or another deliberate choice with rationale in
methodology.md.
README runnable A stranger can clone, install, and reproduce the headline number in
under one hour.
Reproducibility seed All training and eval scripts run from a fixed seed. Logs include seed in
filename.
Held-out sealed Held-out is in a separate file, gitignored from training scripts, and not
committed in unencrypted form to the public repo. Sealed-slice tasks
released only after the leaderboard is published.
Contamination report N-gram, embedding, and time-shift checks all run and committed.
Model card complete (if
applicable)
Backbone, training data, hyperparameters, intended use, limitations,
evaluation results, environmental cost.
Attribution clean Every cited paper, dataset, and tool credited. Tenacious named only as
the workflow domain, not with any private detail.
Program staff sign-off A program staff member reviews the publication checklist before the
artifact goes public under your identity.
## Deliverables
Interim Submission: Wednesday, 21hr UTC
Submit: GitHub repo plus PDF report (public Google Drive link).
Covers Acts I and II — the audit, the schema, and the authored dataset (pre-publication, not yet pushed
to HuggingFace). Dataset must be present in the repo with three partitions, datasheet, and
contamination-check output.
GitHub repo
- README.md at root with overview, status, setup, and what is next.
- audit_memo.md, schema.json, scoring_evaluator.py.
- tenacious_bench_v0.1/ with held_out/, dev/, train/ partitions.
- datasheet.md following the Gebru and Pushkarna templates.
- methodology.md with path declaration, justification citing Week 10 evidence, partitioning
protocol, contamination-check results.

- generation_scripts/ with reproducible authoring code (model routes, judge prompts, dedup logic).
- inter_rater_agreement.md.
- synthesis_memos/ with at least two completed common-reading memos.
- Cost log to date.
PDF report
- Bench composition (counts per dimension, partition, source mode).
- Inter-rater agreement results.
- Three example tasks (one programmatic, one trace-derived, one adversarial) with rubric
application shown.
- What is working, what is not, plan for Days 4–7.
Final Submission: Saturday, 21hr UTC
Submit: GitHub repo, two-page PDF memo, demo video, public artifact URLs (all public, no login
required).
GitHub repo (adds to Wednesday's)
- training_data/ with the formatted training partition for your path.
- methodology_rationale.md citing path-specific papers and at least three Week 10 trace IDs.
- training/ with the training run script, hyperparameters, loss logs.
- ablations/ with ablation_results.json, held_out_traces.jsonl, statistical-test output.
- model_card.md if Path A or C.
- evidence_graph.json mapping every numeric claim in the memo to its source.
- All synthesis memos completed (4 common, 2–3 path-specific).
Public artifacts
- HuggingFace dataset URL.
- HuggingFace model URL if Path A or C.
- Blog post URL.
- Community engagement URL (issue, submission, or PR link).
PDF memo (memo.pdf, exactly two pages)
Page 1 — the decision. Three-sentence executive summary. Headline lift on Tenacious-Bench held-out
with 95% CI (Delta A). Delta B reported honestly. Cost per task with and without the trained component.
Recommendation for the production deployment: deploy, deploy with caveat, or do not deploy with
what would need to change.

Page 2 — the skeptic's appendix. Four failure modes Tenacious-Bench v0.1 still does not capture, with
what would need to be added in v0.2. Public-signal lossiness in your ground truth. One honest
unresolved failure from training. The kill-switch trigger condition for the trained component in
production.
Demo video (max 6 minutes, no login)
- Walk through the dataset on HuggingFace (datasheet visible, partitions visible).
- Show one task being scored end to end by your evaluator.
- Show one ablation result with the held-out traces opened and a numeric claim traced back to its
source.
- Show the blog post page.
- Show the community-engagement artifact (issue link, submission, or PR).
Evidence-Graph Grading
Observable Week 11 manifestation
Reproduction fidelity Tenacious-Bench is reproducible — an evaluator clones the repo, runs
the scoring evaluator on a fresh agent, and gets stable scores within
2pp. Held-out partition sealed and uncorrupted.
Probe / task originality Tasks in Tenacious-Bench cannot be solved by a generic τ²-Bench-tuned
agent. The adversarial-task slice carries the most originality weight.
Multi-LLM synthesis routing decisions documented and defensible.
Mechanism attribution Delta A positive on sealed held-out with statistical significance. Delta B
reported honestly even when negative. Ablations isolate the trained
component from prompt and scaffolding effects.
Cost-quality Pareto Per-task cost and latency reported with and without the trained
component. Cost discipline of the week itself is graded — wasteful
re-runs are a Pareto-observable failure.
Evidence-graph integrity Every numeric claim in the memo and the blog post resolves to a
dataset task ID, a training run log, an ablation table row, or a public
source. Synthesis-memo disagreements cite specific paper sections.
Public-artifact quality Datasheet completeness, model card quality if applicable, blog post
intellectual depth, reproducibility (a stranger can hit your headline
number in under an hour), license correctness, attribution cleanliness.


Build the bench. Train the model. Ship it.