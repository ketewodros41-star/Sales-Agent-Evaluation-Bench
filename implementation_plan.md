# Final Submission — Implementation Plan
**Author:** Kidus Gashaw | **Date:** 2026-04-30 | **Deadline:** Saturday 21:00 UTC

---

## Gap Analysis: What Is Done vs. What Is Still Needed

### Already Complete ✅
- `audit_memo.md` — 5 gaps, 11 probe IDs, 6 trace IDs
- `schema.json` — full schema + 3 example tasks
- `scoring_evaluator.py` — 5-dimension deterministic scorer
- `tenacious_bench_v0.1/` — 200 tasks across 3 partitions
- `contamination_check.json` — n-gram PASS
- `inter_rater_agreement.md` — κ ≥ 0.80 all dimensions
- `datasheet.md` — all 7 Gebru sections + Pushkarna layers
- `methodology.md` — Path B rationale, split protocol, contamination results
- `synthesis_memos/` — 2 of 7 required memos (Liu et al. + Gebru/Pushkarna)
- `report_draft.md` — cross-tab composition, IRA, 3 scored examples, plan
- `README.md`, `cost_log.md`, `score_two.md`, `progress_update.md`

### Still Required for Final Submission ❌

| Artifact | Status | Blocker |
|---|---|---|
| 2 remaining common memos (Chen et al., Gu et al.) | ❌ | Writing only |
| 3 path-specific memos (SimPO, Prometheus 2, Preference Leakage) | ❌ | Writing only |
| `training_data/preference_pairs.jsonl` | ❌ | Requires API calls + your baseline agent |
| `methodology_rationale.md` | ❌ | Writing only |
| `training/` — run script, hyperparameters, loss logs | ❌ | Requires Colab T4 run |
| `ablations/` — ablation_results.json, held_out_traces.jsonl, stats | ❌ | Requires training to finish first |
| `evidence_graph.json` | ❌ | Writing only (built after ablations) |
| `memo.pdf` (2-page executive memo) | ❌ | Writing only (built after ablations) |
| HuggingFace dataset published | ❌ | Requires your HF account |
| Blog post published | ❌ | Requires writing + your account |
| Community engagement (GitHub issue or equivalent) | ❌ | Requires your account |
| Demo video (max 6 minutes) | ❌ | Requires screen recording |

---

## What I Can Build For You vs. What Only You Can Do

### I can build (no human action needed):
- All 5 remaining synthesis memos
- `methodology_rationale.md`
- Training run script (`training/train.py`) with correct Unsloth/SimPO hyperparameters
- Preference pair construction script (`training_data/build_pairs.py`)
- Ablation analysis script (`ablations/run_ablation.py`)
- Statistical significance test script (paired bootstrap)
- `evidence_graph.json` template (you fill in the numbers after training)
- `memo.pdf` content as markdown (you fill in the Delta A/B numbers after training)
- Blog post draft (you fill in results and publish)

### Only you can do (requires your accounts, compute, or physical presence):
1. **Run the baseline agent on 100 train tasks** — needs your Week 10 agent and API key
2. **Run the SimPO training on Colab T4** — needs GPU compute and your Google account
3. **Run the held-out ablation** — needs the trained adapter from step 2
4. **Publish the dataset to HuggingFace** — needs your HF write token
5. **Publish the LoRA adapter** — Path B: not required (spec says model card only for Path A/C) — but publishing the judge is optional and impressive
6. **Write and publish the blog post** — needs your HF community / Substack account
7. **Post the community engagement** — needs your GitHub account (τ²-Bench issue or equivalent)
8. **Record the 6-minute demo video** — screen recording, no login required

---

## Day-by-Day Implementation Plan

### Day 4 (Today) — Training Data + Memos
*I will build the scripts. You run them.*

**Step 1 — Build preference pairs (you run this)**
```bash
# Generate baseline outputs from your Week 10 agent on all 100 train tasks
# Then run:
python training_data/build_pairs.py \
  --tasks tenacious_bench_v0.1/train/tasks.jsonl \
  --baseline-outputs training_data/baseline_outputs.jsonl \
  --output training_data/preference_pairs.jsonl \
  --delta 0.20 \
  --seed 42
```
Target: ≥ 80 valid pairs. If fewer, lower `--delta` to `0.15`.

**Step 2 — Write synthesis memos (I will write these)**
- `synthesis_memos/contamination_survey.md` — Chen et al. EMNLP 2025
- `synthesis_memos/llm_as_judge_survey.md` — Gu et al. 2024–2025
- `synthesis_memos/simpo_memo.md` — Meng et al. NeurIPS 2024
- `synthesis_memos/prometheus2_memo.md` — Kim et al. 2024
- `synthesis_memos/preference_leakage_memo.md` — Li et al. 2025

**Step 3 — Write `methodology_rationale.md` (I will write this)**

---

### Day 5 Morning — Training Run (YOU must do this on Colab T4)

**Step 1 — Open the Unsloth Colab notebook**
Go to: https://colab.research.google.com/github/unslothai/unsloth/blob/main/README.md
Select T4 runtime (Runtime → Change runtime type → T4 GPU)

**Step 2 — Run the training script I will prepare**
```bash
python training/train_simpo.py \
  --pairs training_data/preference_pairs.jsonl \
  --model unsloth/Qwen2.5-1.5B-Instruct \
  --output training/qwen_simpo_judge \
  --epochs 3 \
  --beta 2.0 \
  --gamma 0.5 \
  --lr 5e-5 \
  --seed 42
```
Wall time target: ≤ 30 minutes. Watch the loss curve.

**Kill criterion — check at 10 minutes:**
- Loss not below 0.65 → training data quality problem, not a compute problem. Stop and diagnose.
- Loss decreasing slowly → reduce `--lr` to `2e-5`, allow 45 minutes
- Loss oscillating → increase `--grad-accum` to 16

**Step 3 — Push the adapter to HuggingFace (YOU must do this)**
```bash
huggingface-cli login   # paste your write token
python training/push_adapter.py --adapter training/qwen_simpo_judge
```

---

### Day 5 Afternoon — Ablations (YOU run the scripts I prepare)

```bash
# Delta A: trained judge vs Week 10 baseline on held-out
python ablations/run_ablation.py \
  --held-out tenacious_bench_v0.1/held_out/tasks.jsonl \
  --adapter training/qwen_simpo_judge \
  --baseline week10 \
  --output ablations/ablation_results.json \
  --seed 42

# Delta B: trained judge vs prompt-only same backbone
python ablations/run_ablation.py \
  --held-out tenacious_bench_v0.1/held_out/tasks.jsonl \
  --adapter training/qwen_simpo_judge \
  --baseline prompt-only \
  --output ablations/ablation_results.json \
  --append
```

Each pass costs ~$1.20 (eval-tier, 40 tasks). Run Delta A first. Run Delta B only after Delta A confirms lift — if Delta A is flat, diagnose training before spending on Delta B.

---

### Day 6 — Analysis + Memo + Evidence Graph (I will build the templates; you fill in numbers)

**Step 1 — Statistical significance (script I will provide)**
```bash
python ablations/bootstrap_test.py \
  --results ablations/ablation_results.json \
  --n-bootstrap 1000 \
  --output ablations/significance.json
```

**Step 2 — Fill in `evidence_graph.json`**
Every number in your memo must trace back to a task ID, a training log line, or an ablation table row. I will build the skeleton; you add the actual numbers from your training run.

**Step 3 — Fill in `memo.pdf` (I will write the template)**
Page 1 needs: Delta A lift + 95% CI, Delta B result, cost per task (with/without trained judge), deployment recommendation.
Page 2 needs: 4 failure modes v0.1 doesn't capture, one honest unresolved training failure, the kill-switch condition.

---

### Day 7 — Publish + Engage + Video (All YOU)

**HuggingFace dataset (30 minutes)**
```bash
python publishing/push_dataset.py \
  --dataset tenacious_bench_v0.1 \
  --repo your-hf-handle/tenacious-bench-v0.1 \
  --license cc-by-4.0
```
Keep held-out sealed. Release only train + dev. Held-out releases 2026-05-05.

**Blog post (1–2 hours)**
I will write a full draft. You edit, add your real numbers, and publish on HuggingFace Community, Substack, or your personal site.
Structure: the gap → the audit method → the dataset → the training experiment → the honest result → what's next.
Target: 1,200–2,000 words.

**Community engagement (20 minutes)**
Simplest route: open a GitHub issue on the τ²-Bench repo.
Title: "Tenacious-Bench v0.1: a B2B sales evaluation gap τ²-Bench retail cannot grade"
Body: 3–4 sentences on the gap, link to your HuggingFace dataset.
This is a permanent public artifact with your name on it. Keep it factual and specific.

**Demo video (45–60 minutes)**
Six minutes, no login required. Record these five segments:
1. HuggingFace dataset page — show the datasheet tab and partition files (1 min)
2. Clone the repo, run `scoring_evaluator.py` on one task, show the per-dimension output (2 min)
3. Open `ablations/ablation_results.json`, point to the Delta A row, trace the number to a task ID in `held_out_traces.jsonl` (1 min)
4. Show the blog post in browser (30 sec)
5. Show the community engagement artifact — the GitHub issue URL (30 sec)

Use Loom (free, no login to view) or OBS. Keep it under 6 minutes — the spec is strict on this.

---

## Full Deliverables Checklist for Saturday Submission

### GitHub Repo
- [ ] `training_data/preference_pairs.jsonl` — ≥ 80 valid pairs
- [ ] `training_data/baseline_outputs.jsonl` — baseline agent outputs on train partition
- [ ] `methodology_rationale.md` — path B rationale, ≥3 trace IDs, ≥2 papers
- [ ] `training/train_simpo.py` — training script with pinned hyperparameters
- [ ] `training/run_log.json` — loss curve, wall time, final loss
- [ ] `ablations/ablation_results.json` — Delta A and Delta B
- [ ] `ablations/held_out_traces.jsonl` — raw scoring traces
- [ ] `ablations/significance.json` — paired bootstrap results
- [ ] `evidence_graph.json` — every claim traced to a source
- [ ] `synthesis_memos/` — all 7 memos (4 common + 3 path-specific)
- [ ] `memo.pdf` (or `memo.md` converted) — exactly 2 pages
- [ ] `README.md` updated with HuggingFace URLs and final status

### Public Artifacts
- [ ] HuggingFace dataset URL (share in submission)
- [ ] Blog post URL (share in submission)
- [ ] Community engagement URL — GitHub issue or equivalent
- [ ] Demo video URL (Loom or YouTube, no login required)

### Not Required for Path B
- ~~HuggingFace model URL~~ (only required for Path A and C)
- ~~model_card.md~~ (only required for Path A and C)

---

## Honest Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Preference pair gap too small for SimPO to converge | Medium | High | Lower δ threshold to 0.15 on signal_grounding tasks; document exception |
| Colab T4 session times out mid-training | Low | Medium | Save checkpoint every 100 steps; resume from checkpoint |
| Delta A is flat or negative | Low-Medium | Medium | Still a publishable finding — report it honestly in memo and blog; Delta B becomes the primary finding |
| Budget overrun on ablations | Low | Low | Three ablation passes maximum; no exploratory held-out calls |
| Demo video exceeds 6 minutes | Low | Low | Script the 5 segments before recording; practice once |

---

## What I Will Build Next (in order)

1. `synthesis_memos/` — all 5 remaining memos
2. `methodology_rationale.md`
3. `training/train_simpo.py` — full Unsloth/SimPO training script ready to run on Colab T4
4. `training_data/build_pairs.py` — preference pair construction script
5. `ablations/run_ablation.py` and `ablations/bootstrap_test.py`
6. `memo.md` template (you fill numbers after Day 5–6)
7. Blog post draft
8. `evidence_graph.json` skeleton

Tell me which to start with and I'll build it now.
