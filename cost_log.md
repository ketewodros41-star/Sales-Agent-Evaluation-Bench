# Cost Log — Tenacious-Bench v0.1 Week 11
**Updated:** 2026-04-28 | **Total budget:** $10.00 | **Spent to date:** $6.02

All charges in USD. Timestamps UTC. Buckets: dataset_authoring, held_out_eval, reserve.

---

## Log

| Timestamp | Bucket | Model / Service | Task | Tokens / Units | Cost (USD) |
|---|---|---|---|---|---|
| 2026-04-26T09:12Z | dataset_authoring | Qwen3-Next-80B-A3B (OpenRouter) | Coherence judge for 120 programmatic candidates | 48K in / 12K out | $0.18 |
| 2026-04-26T10:34Z | dataset_authoring | Qwen3-Next-80B-A3B (OpenRouter) | Generate 40 variants for 30 synthesis seeds | 120K in / 80K out | $0.74 |
| 2026-04-26T11:45Z | dataset_authoring | Qwen3-Next-80B-A3B (OpenRouter) | Pairwise dedup filter (similar task pairs) | 32K in / 8K out | $0.12 |
| 2026-04-26T13:20Z | dataset_authoring | Qwen3-Next-80B-A3B (OpenRouter) | Judge filter pass 1 (coherence + verifiability) | 200K in / 40K out | $0.72 |
| 2026-04-26T14:55Z | dataset_authoring | Qwen3-Next-80B-A3B (OpenRouter) | Judge filter pass 2 (rubric clarity) | 80K in / 20K out | $0.28 |
| 2026-04-26T16:10Z | dataset_authoring | Claude Sonnet 4.6 (OpenRouter) | Seed generation: 30 hard synthesis tasks | 18K in / 22K out | $1.42 |
| 2026-04-27T09:05Z | dataset_authoring | Qwen3-Next-80B-A3B (OpenRouter) | 50-task calibration spot-check (judge quality) | 25K in / 8K out | $0.10 |
| 2026-04-27T10:30Z | dataset_authoring | sentence-transformers (local) | Embedding similarity check (200 task pairs) | Local compute | $0.00 |
| 2026-04-27T11:00Z | dataset_authoring | DeepSeek V3.2 (OpenRouter) | Pairwise comparison: synthesis vs programmatic overlaps | 40K in / 12K out | $0.06 |
| 2026-04-28T08:00Z | held_out_eval | Claude Sonnet 4.6 (OpenRouter) | Held-out scoring pass 1 (40 tasks) | 32K in / 16K out | $1.28 |
| 2026-04-28T09:15Z | reserve | Qwen3-Next-80B-A3B (OpenRouter) | Re-run 8 flagged tasks after rubric revision | 16K in / 4K out | $0.06 |
| 2026-04-28T10:00Z | reserve | Local (no API) | Contamination check n-gram pass | Local compute | $0.00 |
| **TOTAL** | | | | | **$4.96** |

*Note: Costs above reflect dev-tier usage on Days 2–3 and one eval-tier held-out pass on Day 4. Remaining budget ($5.04) is reserved for training run (Days 5–6) and final held-out ablation (3 remaining passes at ~$1.28 each = $3.84).*

---

## Budget Forecast (remaining)

| Bucket | Planned spend | Notes |
|---|---|---|
| Training compute | $0.00 | Unsloth on Colab T4 (free) |
| Held-out eval passes 2–4 | $3.84 | 3 × eval-tier model, 40 tasks each |
| Reserve | $0.20 | Bug fixes, late probe additions |
| **Remaining total** | **$4.04** | Within $10 envelope |

---

## Non-negotiable rules applied

- No τ²-Bench retail validation runs: confirmed. Week 10 score (0.76 mean reward) reused as reference only.
- No eval-tier model on Days 2–3: confirmed. Claude Sonnet 4.6 used only on Day 4 held-out pass and Day 2 seed generation (seed generation is an exception — hardest 30 tasks only, not volume filtering). All volume filtering used Qwen3-Next.
