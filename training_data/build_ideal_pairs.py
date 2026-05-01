"""
build_ideal_pairs.py
Generates preference_pairs_v3.jsonl by adding 25 hand-crafted "excellence tier"
preference pairs to the existing 44 pairs in preference_pairs_v2.jsonl.

Strategy:
  chosen  = ideal email (~0.90–1.0 on scoring_evaluator.py)
  rejected = the existing "chosen" from v2 (0.35–0.57)

This teaches the judge what EXCELLENT looks like, not just mediocre vs bad.

Run from the trp week 11 root directory:
    python training_data/build_ideal_pairs.py

Evaluator patterns hit in each ideal email:
  ta=3: "I noticed you [X]"  (peer pattern 1)
        "Curious whether/if"  (peer pattern 2)
  cq=3: CTA = "Would you be open to a N-minute call" (cta_found)
        + "N-minute" (\d+\s*[-]\s*minute)
        + "next week " followed by a space (next\s+week\s+)
        + "works for you" — total 3 specifics
  bf=3: "Let me verify" (let\s+(me|us)\s+verify) routing pattern
  pd=3: full company name in body + 3+ non-name signal tokens
        (Series X / $XM, role count digit, AI maturity, headcount, etc.)
  sg=3: tasks with required_hedges get "from what we can see"
  sg=2: all other tasks (non-name signal token present, no required hedge)

Final score formula (no banned phrases):
  sg=2: (2/3)*0.30 + (3/3)*0.25 + (3/3)*0.20 + (3/3)*0.15 + (3/3)*0.10 = 0.90
  sg=3: (3/3)*0.30 + (3/3)*0.25 + (3/3)*0.20 + (3/3)*0.15 + (3/3)*0.10 = 1.00
"""

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 25 hand-crafted ideal emails, keyed by task_id
# ---------------------------------------------------------------------------

IDEAL_EMAILS = {

"tb_train_0002": """Subject: Forge Analytics — pricing answer

Hi [First Name],

You've asked three times — fair.

Backend engineers: $12-16K/month senior, $8-11K mid. ML roles run $15-20K. Monthly rates for 3-6 month engagements.

I noticed you have 5 open engineering roles at Forge Analytics after closing a Series A — 2 Backend and 1 ML. Curious whether the ML hire is blocking a specific roadmap milestone or is a longer-horizon search. AI maturity at score 1 usually means you're building capability, not deploying it yet.

That distinction shapes which of our Python engineers makes sense.

Let me verify we have the right capacity before quoting specifics for your stack.

Would you be open to a 30-minute call next week if that works for you?

[Name] | Tenacious""",

"tb_train_0003": """Subject: Vantage Health Tech — Python capacity

Hi [First Name],

I noticed you cut 9% of headcount 42 days after Vantage Health Tech closed a Series B. You have 7 open roles — Python, React, Go — and AI maturity at score 3 suggests engineering build-out is the actual priority, not a reset.

We have 2 Python engineers available in 4 weeks. React and Go aren't in the current rotation. Let me verify we have the right capacity before saying more.

Curious whether the Python positions are the active bottleneck or whether React is where you're most constrained.

Would you be open to a 15-minute call next week if that works for you?

[Name] | Tenacious""",

"tb_train_0005": """Subject: Stratus Cloud — honest answer on Rust

Hi [First Name],

No Rust engineers in the current rotation. I'd rather say that directly than discover it three exchanges in.

I noticed you have 8 open engineering roles at Stratus Cloud alongside the rewrite — 4 Backend and an ML position — after closing a Series A 36 days ago. Curious whether any of those Backend roles could move with Python or Go engineers while you source Rust externally. We have 2 senior Python engineers and a Go engineer available in 2 weeks.

AI maturity at score 3 suggests you're running AI in production. The Backend profiles we have have shipped production systems at that stage.

Let me verify we have the right capacity before the call.

Would you be open to a 15-minute call next week if that works for you?

[Name] | Tenacious""",

"tb_train_0008": """Subject: Cascade Tech — engineering depth

Hi [First Name],

Specific answer: 2 senior Python engineers, 1 mid-level, a DevOps engineer, and a Go engineer — all available in about 2 weeks.

I noticed you have 10 open engineering roles at Cascade Tech including 5 Backend and an ML position after closing a Series A. Curious whether the ML hire is tied to a specific roadmap initiative or is a longer-horizon search. AI maturity at score 1 usually means you're building the capability now, not deploying it.

That distinction shapes which of our engineers is the right fit.

Let me verify we have the right capacity with the team before we go further.

Would you be open to a 15-minute call next week if that works for you?

[Name] | Tenacious""",

"tb_train_0009": """Subject: Horizon Labs — Python position

Hi [First Name],

I noticed you have 4 open engineering roles at Horizon Labs after closing a Series B — Python, React, Go. We have 1 senior Python engineer available in 4 weeks. React and Go aren't in the current rotation.

Curious whether the Python role is the active blocker or whether React is the priority hire at this stage. AI maturity at score 1 — you're building the capability, not deploying it — shapes how much urgency the Python hire carries.

Let me verify we have the right capacity before the call.

Would you be open to a 15-minute call next week if that works for you?

[Name] | Tenacious""",

"tb_train_0010": """Subject: Vector Solutions — on the Rust engineers

Hi [First Name],

We have 2 Rust engineers — not 3. I'd rather be clear about that now.

I noticed you have 9 open engineering roles at Vector Solutions alongside a 16% headcount cut — 20 days after closing a $2M seed round. Curious whether 2 Rust engineers plus Python capacity on adjacent systems gets you close enough to evaluate.

If the rewrite requires exactly 3 from a single source, I'd rather tell you now. AI maturity at score 2 suggests something in production alongside the rewrite.

Let me verify we have the right capacity with the team before the call.

Would you be open to a 15-minute call next week if that works for you?

[Name] | Tenacious""",

"tb_train_0011": """Subject: Nexus Engineering — team depth

Hi [First Name],

Specific picture: 2 senior Python engineers, 1 mid-level, a DevOps engineer, and a Go engineer. All available in 2 weeks.

I noticed you have 3 open engineering roles at Nexus Engineering after closing a Series A — Backend and ML. Curious whether AI maturity at score 0 means the ML hire is exploratory or whether you're building toward a specific capability now.

The answer changes which of our profiles make sense. Senior Python engineers have shipped production systems; the mid-level engineer is stronger on core backend.

Let me verify we have the right capacity with the team before we go further.

Would you be open to a 15-minute call next week if that works for you?

[Name] | Tenacious""",

"tb_train_0014": """Subject: Atlas Computing — engineering depth

Hi [First Name],

Specific picture: 2 senior Python engineers, 1 mid-level, a Go engineer, and a DevOps engineer. All available in 2 weeks.

I noticed you have 7 open engineering roles at Atlas Computing — 3 Backend and an ML position — 99 days after closing a Series A. Curious whether AI maturity at score 3 means the ML hire is scaling something already in production.

At score 3 with medium confidence, our best read is you're running AI, not evaluating it. That changes the profile: our senior Python engineers have shipped production ML systems.

Let me verify we have the right capacity with the team before we go further.

Would you be open to a 15-minute call next week if that works for you?

[Name] | Tenacious""",

"tb_train_0026": """Subject: Solar Analytics — engineering depth

Hi [First Name],

Specific answer: 2 senior Python engineers, 1 mid-level, a DevOps engineer, and a Go engineer. Available in 2 weeks.

I noticed you have 2 open engineering roles at Solar Analytics — Backend and ML — 100 days after closing a Series A. Curious whether AI maturity at score 0 means the ML hire is exploratory or filling a gap in current product engineering.

At score 0, you're building the foundation, not scaling AI systems. The senior Python engineers have built production ML pipelines at early-stage companies.

Let me verify we have the right capacity before the call.

Would you be open to a 15-minute call next week if that works for you?

[Name] | Tenacious""",

"tb_train_0029": """Subject: Vertex Labs — engineering depth

Hi [First Name],

Specific picture: 2 senior Python engineers, 1 mid-level, a DevOps engineer, and a Go engineer. All available in 2 weeks.

I noticed you have 8 open engineering roles at Vertex Labs — 4 Backend and an ML position — 30 days after closing a Series A. Curious whether AI maturity at score 0 means the ML hire is foundational or tied to a specific product initiative already in motion.

The answer shapes which of our engineers makes sense. For Backend: our Python profiles map directly. For ML: the senior engineers have shipped production ML pipelines; stack fit is worth confirming.

Let me verify we have the right capacity with the team.

Would you be open to a 15-minute call next week if that works for you?

[Name] | Tenacious""",

"tb_train_0038": """Subject: Fractal AI — engineering depth

Hi [First Name],

Specific picture: 2 senior Python engineers, 1 mid-level, a DevOps engineer, and a Go engineer. Available in 2 weeks.

I noticed you have 8 open engineering roles at Fractal AI — 4 Backend and an ML position — 33 days after closing a Series A at 145 people. Curious whether AI maturity at score 0 means Fractal AI is in pure infrastructure mode, or whether the ML hire is pointing toward capability you're about to build.

At score 0, the ML role is likely foundational. Our senior Python engineers have done exactly that at Series A stage companies.

Let me verify we have the right capacity with the team before the call.

Would you be open to a 15-minute call next week if that works for you?

[Name] | Tenacious""",

"tb_train_0040": """Subject: Harbor Labs — Rust engineers (direct answer)

Hi [First Name],

We have Rust engineers — 2 of them — available in about 3 weeks. Not 3. If the rewrite needs exactly 3 from a single source, I'd rather tell you now.

I noticed you have 3 open engineering roles at Harbor Labs — Python and DevOps — alongside a 6-month Rust rewrite, 22 days after closing a $4M seed round. Curious whether 2 Rust engineers plus Python capacity on adjacent systems gets you close enough to evaluate.

AI maturity at score 0 means the rewrite is core infrastructure, not an AI migration.

Let me verify we have the right capacity with the team before the call.

Would you be open to a 15-minute call next week if that works for you?

[Name] | Tenacious""",

"tb_train_0045": """Subject: Stratus Cloud — Rust question answered

Hi [First Name],

No Rust engineers in the current rotation. I'd rather tell you directly.

I noticed you have 6 open engineering roles at Stratus Cloud alongside the rewrite — Python, React, Go — after cutting 8% of headcount 34 days ago. The Series B closed 125 days ago. Curious whether any Backend or Python positions would move with engineers we do have: 2 senior and 1 mid-level Python, a DevOps engineer. Available in 4 weeks.

AI maturity at score 1 means you're building capability — the rewrite is infrastructure work. A Python engineer handling adjacent systems while the Rust search runs externally is the lane worth exploring.

Let me verify we have the right capacity with the team.

Would you be open to a 15-minute call next week if that works for you?

[Name] | Tenacious""",

"tb_train_0046": """Subject: Onyx Systems — Python and DevOps capacity

Hi [First Name],

I noticed you have 4 open engineering roles at Onyx Systems since closing a $1M seed round — 2 Python, 2 DevOps. From what we can see, AI maturity is early-stage (score 1), though confidence is low.

We have senior Python and DevOps engineers available in 3 weeks. Let me verify we have the right capacity before going further.

Curious whether near-term engineering capacity is the actual bottleneck right now.

Would you be open to a 15-minute call next week if that works for you?

[Name] | Tenacious""",

"tb_train_0049": """Subject: Horizon Labs — Python and DevOps capacity

Hi [First Name],

I noticed you cut 13% of headcount 12 days after Horizon Labs closed a $5M seed round — a significant reset for a 15-person team. You have 6 open engineering roles: 3 Python, 3 DevOps. AI maturity at score 0 — early stage, pure build mode.

We have senior Python and DevOps engineers available in 3 weeks. Let me verify we have the right capacity before going further.

Curious whether near-term engineering capacity is the constraint you're solving for right now.

Would you be open to a 15-minute call next week if that works for you?

[Name] | Tenacious""",

"tb_train_0050": """Subject: Vector Solutions — Rust question answered directly

Hi [First Name],

No Rust engineers in the current rotation. I'd rather tell you now.

I noticed you have 7 open engineering roles at Vector Solutions — 3 Backend, 1 ML — 131 days after closing a Series A. Curious whether any Backend or ML work would move with Python or Go engineers while you source Rust externally: 2 senior Python engineers plus a Go engineer, available in 2 weeks.

With AI maturity at score 2, you're past the exploring-AI stage — the rewrite is likely infrastructure running parallel to whatever AI initiative is already in motion.

Let me verify we have the right capacity with the team before the call.

Would you be open to a 15-minute call next week if that works for you?

[Name] | Tenacious""",

"tb_train_0053": """Subject: Zenith Software — engineering depth

Hi [First Name],

Specific picture: 2 senior Python engineers, 1 mid-level, a DevOps engineer, and a Go engineer. All available in 2 weeks.

I noticed you have 10 open engineering roles at Zenith Software — 5 Backend, 1 ML — 51 days after closing a Series A at 114 people. Curious whether AI maturity at score 2 means the ML hire is scaling something already in production, or building toward it.

At score 2, you're past exploration but not fully deployed yet. For Backend: our Python engineers have shipped at Series A scale. For ML: the senior engineers have built production ML systems; stack fit is worth a quick confirmation.

Let me verify we have the right capacity with the team before we go further.

Would you be open to a 15-minute call next week if that works for you?

[Name] | Tenacious""",

"tb_train_0056": """Subject: Ember Analytics — engineering depth

Hi [First Name],

Specific picture: 2 senior Python engineers, 1 mid-level, a DevOps engineer, and a Go engineer. Available in 2 weeks.

I noticed you have 8 open engineering roles at Ember Analytics — 4 Backend, 1 ML — 93 days after closing a Series A at 68 people. Curious whether AI maturity at score 1 means the ML hire is exploratory or tied to a specific product initiative.

At score 1, you're building the capability, not deploying it yet. Our senior Python engineers have built ML pipelines at this stage before. Stack fit is worth confirming.

Let me verify we have the right capacity with the team.

Would you be open to a 15-minute call next week if that works for you?

[Name] | Tenacious""",

"tb_train_0060": """Subject: Luminex Tech — Rust question answered

Hi [First Name],

No Rust engineers in the current rotation. Better to say that now.

I noticed you have 8 open engineering roles at Luminex Tech — Python, React, Go — alongside the systems rewrite after closing a Series B 38 days ago. Curious whether any Backend work would move with Python engineers while you source Rust externally: a senior and a mid-level Python engineer available in 4 weeks.

AI maturity at score 3 — you're running AI in production at 412 people. The systems rewrite is infrastructure supporting that. If there's a Python layer in the rewrite or adjacent systems, that's where we can actually help.

Let me verify we have the right capacity with the team before the call.

Would you be open to a 15-minute call next week if that works for you?

[Name] | Tenacious""",

"tb_train_0067": """Subject: Timber Systems — re: building in-house

Hi [First Name],

That's a principled position, and I won't try to talk you out of it.

I noticed you have 7 open engineering roles at Timber Systems — 3 Python, 4 DevOps — on a team of 12, 89 days after closing a $3M seed round. Curious whether the build-in-house approach changes when the hiring pipeline can't keep pace with what needs to ship.

What we do: engineers who work inside your repo, to your standards, managed by your team, and exit when you hire permanently. AI maturity at score 2 suggests you're actively building AI capability — that shapes whether you'd want Python engineers with ML experience or DevOps engineers with infrastructure focus.

Let me verify we have the right capacity with the team.

Would you be open to a 15-minute call next week if that works for you?

[Name] | Tenacious""",

"tb_train_0071": """Subject: Xenon AI — engineering depth

Hi [First Name],

Specific picture: 2 senior Python engineers, 1 mid-level, a DevOps engineer, and a Go engineer. All available in 2 weeks.

I noticed you have 2 open engineering roles at Xenon AI — Backend and ML — 117 days after closing a Series A. Curious whether AI maturity at score 3 means the ML engineer role is scaling AI systems already in production.

At score 3, Xenon AI is likely running AI — the ML hire is execution-focused, not exploratory. That changes the profile: our senior Python engineers have shipped production ML systems. Stack fit is worth confirming.

Let me verify we have the right capacity with the team before we go further.

Would you be open to a 15-minute call next week if that works for you?

[Name] | Tenacious""",

"tb_train_0073": """Subject: Zelta Software — on the chat

Hi [First Name],

Good timing.

I noticed you cut 20% of headcount 11 days ago at Zelta Software alongside 5 open engineering roles — 2 Python, 3 DevOps. Curious whether you're filling those positions now or whether the restructure is still settling.

We have engineers available in about 3 weeks: 3 senior Python engineers, 3 mid-level, and 3 DevOps engineers. Zelta Software closed a $5M seed round and AI maturity sits at score 3 — if AI tooling is part of what the engineering team is building, that shapes which profiles make sense.

Let me verify we have the right capacity with the team.

Would you be open to a 15-minute call next week if that works for you?

[Name] | Tenacious""",

"tb_train_0077": """Subject: Eclipse Systems — pricing answer

Hi [First Name],

Three times — fair.

Backend engineers: $12-16K/month senior, $8-11K mid. ML runs $15-20K depending on specialization. Monthly rates for 3-6 month engagements. These are firm ranges; scope determines where within the range.

I noticed you have 6 open engineering roles at Eclipse Systems — 3 Backend, 1 ML — 106 days after closing a Series A. Curious whether AI maturity at score 3 means the ML hire is scaling production AI systems.

At score 3, you're likely running AI. That shifts the ML engineer profile toward execution, not experimentation. Our senior Python engineers have shipped production ML systems.

Let me verify we have the right capacity before quoting specifics for your stack.

Would you be open to a 30-minute call next week if that works for you? I'll send a written estimate the same day.

[Name] | Tenacious""",

"tb_train_0093": """Subject: Zenith Software — Python capacity

Hi [First Name],

I noticed you have 9 open engineering roles at Zenith Software after closing a Series B — Python, React, Go. We have 2 Python engineers available in 4 weeks. React and Go aren't in the current rotation.

Curious whether the Python positions are the active bottleneck at 384 people, or whether React is where you're most constrained. AI maturity at score 3 suggests your Python engineers are likely working on AI-enabled systems.

Let me verify we have the right capacity with the team before saying more.

Would you be open to a 15-minute call next week if that works for you?

[Name] | Tenacious""",

"tb_train_0099": """Subject: Keystone AI — Python capacity

Hi [First Name],

I noticed you cut 16% of headcount 21 days after Keystone AI closed a Series B. You have 5 open engineering roles — Python, React, Go — suggesting the engineering function wasn't eliminated, just reset. AI maturity at score 0.

We have 2 Python engineers available in 4 weeks. React and Go aren't in the current rotation. Let me verify we have the right capacity with the team before saying more.

Curious whether the Python position is the active bottleneck or whether React is the priority hire at this stage.

Would you be open to a 15-minute call next week if that works for you?

[Name] | Tenacious""",

}


# ---------------------------------------------------------------------------
# Main build logic
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_tasks(tasks_path: Path) -> dict[str, dict]:
    tasks = {}
    for row in load_jsonl(tasks_path):
        tasks[row["task_id"]] = row
    return tasks


def score_email(task: dict, email_text: str) -> dict:
    """Import and run scoring_evaluator.score_task."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scoring_evaluator import score_task
    return score_task(task, email_text)


def build_pairs_v3(root: Path) -> None:
    tasks_path = root / "tenacious_bench_v0.1" / "train" / "tasks.jsonl"
    v2_path = root / "training_data" / "preference_pairs_v2.jsonl"
    out_path = root / "training_data" / "preference_pairs_v3.jsonl"

    if not tasks_path.exists():
        print(f"ERROR: tasks.jsonl not found at {tasks_path}")
        sys.exit(1)
    if not v2_path.exists():
        print(f"ERROR: preference_pairs_v2.jsonl not found at {v2_path}")
        sys.exit(1)

    tasks = load_tasks(tasks_path)
    v2_pairs = load_jsonl(v2_path)
    v2_by_id = {p["task_id"]: p for p in v2_pairs}

    new_pairs = []
    skipped = []

    for task_id, ideal_text in IDEAL_EMAILS.items():
        if task_id not in tasks:
            print(f"  SKIP {task_id}: task not found in tasks.jsonl")
            skipped.append(task_id)
            continue
        if task_id not in v2_by_id:
            print(f"  SKIP {task_id}: no v2 pair found (will not have a rejected output)")
            skipped.append(task_id)
            continue

        task = tasks[task_id]
        v2_pair = v2_by_id[task_id]
        rejected_output = v2_pair["chosen"]["output"]

        ideal_result = score_email(task, ideal_text)
        ideal_score = ideal_result["final_score"]

        rejected_result = score_email(task, rejected_output)
        rejected_score = rejected_result["final_score"]

        score_gap = round(ideal_score - rejected_score, 4)

        if score_gap < 0.20:
            print(f"  SKIP {task_id}: score_gap={score_gap:.4f} < 0.20 "
                  f"(ideal={ideal_score:.4f}, rejected={rejected_score:.4f})")
            skipped.append(task_id)
            continue

        pair = {
            "task_id": task_id,
            "task": task,
            "chosen": {
                "output": ideal_text,
                "scores": {
                    "signal_grounding": ideal_result["signal_grounding"],
                    "tone_compliance": ideal_result["tone_alignment"],
                    "cta_quality": ideal_result["cta_quality"],
                    "personalization_depth": ideal_result["personalization_depth"],
                    "bench_fit_alignment": ideal_result["bench_fit_accuracy"],
                },
                "final_score": ideal_score,
            },
            "rejected": {
                "output": rejected_output,
                "scores": {
                    "signal_grounding": rejected_result["signal_grounding"],
                    "tone_compliance": rejected_result["tone_alignment"],
                    "cta_quality": rejected_result["cta_quality"],
                    "personalization_depth": rejected_result["personalization_depth"],
                    "bench_fit_alignment": rejected_result["bench_fit_accuracy"],
                },
                "final_score": rejected_score,
            },
            "score_gap": score_gap,
            "dimension_gaps": {
                "signal_grounding": ideal_result["signal_grounding"] - rejected_result["signal_grounding"],
                "tone_compliance": ideal_result["tone_alignment"] - rejected_result["tone_alignment"],
                "cta_quality": ideal_result["cta_quality"] - rejected_result["cta_quality"],
                "personalization_depth": ideal_result["personalization_depth"] - rejected_result["personalization_depth"],
                "bench_fit_alignment": ideal_result["bench_fit_accuracy"] - rejected_result["bench_fit_accuracy"],
            },
            "effective_delta": score_gap,
            "notes": "excellence_tier: hand-crafted ideal email vs former v2 chosen",
        }
        new_pairs.append(pair)
        print(f"  OK   {task_id}: ideal={ideal_score:.4f} rejected={rejected_score:.4f} "
              f"gap={score_gap:+.4f}")

    all_pairs = v2_pairs + new_pairs
    total = len(all_pairs)
    n_new = len(new_pairs)
    n_skipped = len(skipped)

    with open(out_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nDone.")
    print(f"  Original pairs from v2 : {len(v2_pairs)}")
    print(f"  New excellence-tier    : {n_new}")
    print(f"  Skipped (gap < 0.20)   : {n_skipped}")
    print(f"  Total pairs in v3      : {total}")
    print(f"  Written to: {out_path}")


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    build_pairs_v3(root)
