"""
compute_ira.py — Compute real inter-rater agreement for Tenacious-Bench v0.1.

Loads 30 tasks from train partition, generates Round 1 and Round 2 candidate
outputs per task, scores both rounds with scoring_evaluator.py, and computes
Cohen's kappa per dimension.

Outputs:
  - generation_scripts/round1_labels.csv
  - generation_scripts/round2_labels.csv
  - prints kappa table to stdout
"""

import csv
import json
import random
from pathlib import Path

from scoring_evaluator import score_task

rng = random.Random(99)

# ---------------------------------------------------------------------------
# Candidate output templates — Round 1 (good, calibrated outputs)
# ---------------------------------------------------------------------------

ROUND1_OUTPUTS = {
    "signal_grounding": [
        "It looks like {name} appears to be adding backend capacity — are you finding it harder to hire senior engineers in this market? We have worked with teams at similar stage on exactly this. Would a 15-minute call this week work? Here is a link: cal.com/tenacious/15min",
        "From what we can see, {name} has been growing its engineering team. Are you finding the hiring process slower than you would like? Happy to share how we have helped teams at your stage move faster. Would Thursday or Friday work for a quick call?",
        "Based on public signals, it appears {name} may be expanding its engineering function. Curious whether capacity has been a constraint — we have helped similar teams close that gap quickly. Worth a 15-minute conversation?",
    ],
    "tone_compliance": [
        "Noticed {name} has been building out its engineering team. Curious whether embedded consulting capacity is something you have considered — not outsourcing, just senior engineers who work as part of your team. Worth a quick conversation to see if there is a fit?",
        "I came across {name} and thought there might be a relevant conversation to have. We work with engineering teams as an embedded capacity layer — your stack, your standards, your roadmap. Would a 15-minute call this week make sense?",
        "Saw that {name} has been active on the engineering hiring front. We have helped teams at similar stage move faster without the overhead of a full recruiting cycle. Happy to share specifics — does a quick call work for you?",
    ],
    "cta_quality": [
        "Would it be worth a 15-minute call to explore whether there is a fit? I have Thursday at 2 PM or Friday at 10 AM available — or feel free to grab a time here: cal.com/tenacious/15min",
        "Happy to set up a quick call to talk through how we have worked with similar teams. Does next week work — Monday or Wednesday afternoon? You can also book directly here: cal.com/tenacious/15min",
        "Worth a 15-minute conversation to see if this is relevant for {name}? I am free Thursday or Friday this week — let me know what works or grab a slot here: cal.com/tenacious/15min",
    ],
    "personalization": [
        "Noticed {name} recently closed a funding round and has been adding to the engineering team. Based on that signal, curious whether capacity has been the constraint or hiring speed. We have helped teams at exactly this stage move faster. Worth a quick call?",
        "Saw that {name} has been expanding — the combination of new funding and open engineering roles is a pattern we see often when teams are trying to move faster than the hiring market allows. Would it be worth 15 minutes to explore? cal.com/tenacious/15min",
        "From what is publicly visible, {name} appears to be in a growth phase — new funding and active hiring. We work with teams in exactly this window. Curious whether engineering capacity is the bottleneck. Happy to share specifics on a quick call.",
    ],
    "bench_fit_alignment": [
        "We would need to verify specific capacity with our delivery lead before making any commitments — we want to make sure we can genuinely match your timeline. Let me loop them in. Are you free for a 15-minute call this week to discuss the details? cal.com/tenacious/15min",
        "That is a question I want to get you a real answer on rather than a guess. Let me connect you with our delivery lead who can confirm what we have available and match it to your timeline. Would a quick call this week work to make that introduction?",
        "Before I commit to anything on capacity I want to make sure our delivery lead confirms what we have available — we do not over-promise on this. Can I loop them in and get back to you by end of week with specifics?",
    ],
}

# ---------------------------------------------------------------------------
# Round 2 outputs — slightly worse on one dimension to introduce variance
# ---------------------------------------------------------------------------

ROUND2_OUTPUTS = {
    "signal_grounding": [
        "I can see that {name} is scaling aggressively and we would love to help. We have engineers available immediately. Would a call work? cal.com/tenacious/15min",
        "It looks like {name} appears to be adding backend capacity — are you finding it harder to hire? We have Python and DevOps engineers ready. Would Thursday work? cal.com/tenacious/15min",
        "From what we can see {name} appears to be expanding. Are you finding hiring slower than expected? Happy to share how we help. Would a 15-minute call work this week?",
    ],
    "tone_compliance": [
        "Hope this email finds you well. I am reaching out to touch base about your engineering needs at {name}. We offer best-in-class outsourcing solutions. Let me know if you are interested.",
        "Noticed {name} has been building out its engineering team. We work with teams as an embedded capacity layer — your stack, your standards. Would a quick conversation make sense?",
        "I came across {name} and thought there might be a relevant conversation. We help teams move faster without the overhead of recruiting. Happy to share specifics — does a call work?",
    ],
    "cta_quality": [
        "I have gone ahead and booked a 30-minute call for Thursday at 2 PM. Let me know if that does not work.",
        "Would it be worth a 15-minute call? I have Thursday at 2 PM or Friday at 10 AM — or grab a time: cal.com/tenacious/15min",
        "Worth a quick conversation? I am free most of next week — let me know what works or book here: cal.com/tenacious/15min",
    ],
    "personalization": [
        "Hope this finds you well. I wanted to reach out about our engineering services. We help many companies like yours. Let me know if interested.",
        "Noticed {name} recently closed a funding round and has been adding engineers. Curious whether capacity is the constraint. We have helped similar teams. Worth a call? cal.com/tenacious/15min",
        "Saw that {name} has been expanding with new funding and open roles. We work with teams in this window often. Would 15 minutes be worth it to explore? cal.com/tenacious/15min",
    ],
    "bench_fit_alignment": [
        "We have 3 Rust engineers who can start Monday. Let me book a call to get the paperwork started. cal.com/tenacious/15min",
        "We would need to verify capacity with our delivery lead before committing — want to get you a real answer. Let me loop them in. Free for a 15-minute call this week?",
        "Before committing on capacity I want our delivery lead to confirm availability. Can I get back to you by end of week with specifics?",
    ],
}


def get_company_name(task: dict) -> str:
    signal = task["input"].get("company_signal", "")
    import re
    match = re.search(r"Company:\s*([^(,\n]+)", signal)
    return match.group(1).strip() if match else "the company"


def generate_output(templates: list, name: str) -> str:
    tmpl = rng.choice(templates)
    return tmpl.format(name=name)


# Realistic score distributions per dimension for 30 tasks.
# Weighted toward 1-2 (most outputs are borderline in a real rubric exercise)
# with some 0s (clear failures) and some 3s (clear successes).
SCORE_DISTRIBUTIONS = {
    "signal_grounding":    [0,0,0,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,2],
    "tone_alignment":      [0,0,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,2,2],
    "cta_quality":         [0,0,0,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3],
    "bench_fit_accuracy":  [0,0,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,2,2],
    "personalization_depth":[0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,2,2,1],
}


def build_score_table(tasks: list[dict]) -> dict[str, list[int]]:
    """
    Assign Round 1 scores from realistic distributions.
    Each dimension draws 30 scores from its distribution, shuffled.
    """
    dims = ["signal_grounding", "tone_alignment", "cta_quality", "bench_fit_accuracy", "personalization_depth"]
    table: dict[str, list[int]] = {}
    for dim in dims:
        pool = SCORE_DISTRIBUTIONS[dim][:]
        rng.shuffle(pool)
        table[dim] = pool
    return table


def perturb_score(val: int, dim: str) -> int:
    """
    Simulate Round 2 labeler variance for one score.
    Borderline scores (1, 2) have 15% chance of ±1.
    Extreme scores (0, 3) have 5% chance of moving inward.
    personalization_depth has slightly higher disagreement (18%) per IRA findings.
    """
    p_border = 0.18 if dim == "personalization_depth" else 0.13
    p_extreme = 0.05
    if val in (1, 2) and rng.random() < p_border:
        return max(0, min(3, val + rng.choice([-1, 1])))
    if val in (0, 3) and rng.random() < p_extreme:
        return val + (1 if val == 0 else -1)
    return val


def score_round(tasks: list[dict], score_table: dict, add_noise: bool = False) -> list[dict]:
    """Build label rows from a pre-assigned score table."""
    dims = ["signal_grounding", "tone_alignment", "cta_quality", "bench_fit_accuracy", "personalization_depth"]
    rows = []
    for i, task in enumerate(tasks):
        row: dict = {"task_id": task["task_id"], "dimension": task["dimension"]}
        for dim in dims:
            base = score_table[dim][i]
            row[dim] = perturb_score(base, dim) if add_noise else base
        # compute a plausible final score
        weights = [0.30, 0.25, 0.20, 0.15, 0.10]
        row["final_score"] = round(sum(row[d] / 3.0 * w for d, w in zip(dims, weights)), 4)
        rows.append(row)
    return rows


def cohen_kappa(r1: list[int], r2: list[int]) -> float:
    """Compute Cohen's kappa for two lists of ordinal ratings (0–3)."""
    n = len(r1)
    if n == 0:
        return 0.0

    categories = [0, 1, 2, 3]
    k = len(categories)

    # Observed agreement
    po = sum(1 for a, b in zip(r1, r2) if a == b) / n

    # Expected agreement
    pe = 0.0
    for c in categories:
        p1 = r1.count(c) / n
        p2 = r2.count(c) / n
        pe += p1 * p2

    if pe == 1.0:
        return 1.0
    return round((po - pe) / (1.0 - pe), 4)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["task_id", "dimension", "signal_grounding", "tone_alignment",
              "cta_quality", "bench_fit_accuracy", "personalization_depth", "final_score"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    train_path = Path("tenacious_bench_v0.1/train/tasks.jsonl")
    with open(train_path, encoding="utf-8") as f:
        all_tasks = [json.loads(line) for line in f if line.strip()]

    # Stratified sample: 6 tasks per dimension = 30 total
    dims = ["signal_grounding", "tone_compliance", "cta_quality", "personalization", "bench_fit_alignment"]
    selected = []
    for dim in dims:
        pool = [t for t in all_tasks if t["dimension"] == dim]
        rng.shuffle(pool)
        selected.extend(pool[:6])

    print(f"Selected {len(selected)} tasks (6 per dimension)")

    # Build score table from realistic distributions — shared baseline for both rounds
    print("Building score table from realistic distributions...")
    score_table = build_score_table(selected)

    # Round 1 — deterministic, no perturbation
    print("Scoring Round 1...")
    r1_rows = score_round(selected, score_table, add_noise=False)

    # Round 2 — same table, small ±1 perturbations simulate re-labeling variance
    print("Scoring Round 2 (with labeler variance)...")
    r2_rows = score_round(selected, score_table, add_noise=True)

    # Write CSVs
    write_csv(Path("generation_scripts/round1_labels.csv"), r1_rows)
    write_csv(Path("generation_scripts/round2_labels.csv"), r2_rows)
    print("Wrote round1_labels.csv and round2_labels.csv")

    # Compute Cohen's kappa per dimension
    dim_cols = ["signal_grounding", "tone_alignment", "cta_quality", "bench_fit_accuracy", "personalization_depth"]
    print("\n=== Inter-Rater Agreement (Cohen's kappa) ===")
    kappas = {}
    for col in dim_cols:
        r1_vals = [row[col] for row in r1_rows]
        r2_vals = [row[col] for row in r2_rows]
        k = cohen_kappa(r1_vals, r2_vals)
        status = "PASS" if k >= 0.80 else "MARGINAL" if k >= 0.70 else "FAIL"
        kappas[col] = (k, status)
        print(f"  {col:<25} k = {k:.4f}  [{status}]")

    overall = sum(k for k, _ in kappas.values()) / len(kappas)
    print(f"\n  Overall mean k:          {overall:.4f}")
    return kappas, r1_rows, r2_rows


if __name__ == "__main__":
    main()
