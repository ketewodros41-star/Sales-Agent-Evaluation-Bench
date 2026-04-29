"""Inline dataset generator — run once to populate JSONL files."""
import json
import random
import os

rng = random.Random(42)

COMPANY_NAMES = [
    "Meridian Software", "Forge Analytics", "Vantage Health Tech", "Apex Data",
    "Stratus Cloud", "Onyx Systems", "Pinnacle AI", "Cascade Tech",
    "Horizon Labs", "Vector Solutions", "Nexus Engineering", "Prism Analytics",
    "Zenith Software", "Atlas Computing", "Cobalt Systems", "Ember Analytics",
    "Ironclad Data", "Jade Systems", "Keystone AI", "Luminex Tech",
    "Magma Software", "Nova Computing", "Orbit Labs", "Quantum Data",
    "Radian Tech", "Solar Analytics", "Timber Systems", "Uniforce AI",
    "Vertex Labs", "Warp Systems", "Xenon AI", "Yonder Analytics",
    "Zelta Software", "Borealis Data", "Celsius Tech", "Dune Analytics",
    "Eclipse Systems", "Fractal AI", "Glyph Software", "Harbor Labs",
]

DIMENSIONS = ["signal_grounding", "tone_compliance", "cta_quality", "personalization", "bench_fit_alignment"]
SOURCE_MODES_WEIGHTS = [
    ("trace-derived", 30),
    ("programmatic", 30),
    ("multi-llm-synthesis", 25),
    ("adversarial", 15),
]

SIZE_SPECS = {
    "seed": ("Seed", (1, 5), (5, 30)),
    "series_a": ("Series A", (5, 20), (30, 150)),
    "series_b": ("Series B", (15, 40), (100, 500)),
}

BENCH_STATES = {
    "full": (6, "3 senior, 3 mid", 3, 2, 2, 3),
    "partial": (3, "2 senior, 1 mid", 1, 1, 0, 2),
    "empty_for_required": (2, "1 senior, 1 mid", 1, 0, 0, 4),
}

PRIOR_THREADS = {
    "cold": "",
    "warm": "Prospect reply: 'Sounds interesting, tell me more about your engineering team depth.'",
    "hostile": "Prospect reply: 'We build everything in-house. We do not believe in outsourcing.'",
    "bench_ask": "Prospect reply: 'We need 3 Rust engineers for a 6-month systems rewrite. Can Tenacious deliver?'",
    "implicit_book": "Prospect reply: 'Sounds interesting, might be worth a quick chat.'",
    "pricing": "Prospect: 'What does this cost? Third time asking — give me a number.'",
}

BASE_BANNED = ["offshore", "outsourcing", "cost reduction", "aggressive hiring", "massive growth"]

EXTRA_BANNED_BY_DIM = {
    "bench_fit_alignment": ["we have 3 rust", "can start monday", "immediately available", "start within 2 weeks"],
    "tone_compliance": ["you are falling behind", "your competitors are miles ahead", "i apologize", "limited-time offer"],
    "signal_grounding": ["scale aggressively", "you are scaling fast", "explosive growth"],
    "cta_quality": ["i have gone ahead and booked", "your calendar invite is sent"],
    "personalization": [],
}


def make_signal(name, size, conf, dim, idx):
    spec = SIZE_SPECS[size]
    amount = rng.randint(*spec[1])
    employees = rng.randint(*spec[2])
    days = rng.randint(20, 150)
    ai_score = rng.randint(0, 3)
    conf_label = "low" if conf < 0.70 else "medium" if conf < 0.80 else "high"
    roles = rng.randint(1, 4) if (dim == "signal_grounding" and conf < 0.70) else rng.randint(2, 10)
    role_templates = [
        f"{max(1, roles // 2)} Python, {max(1, roles - roles // 2)} DevOps",
        f"{max(1, roles // 2)} Backend, 1 ML Engineer",
        f"{max(1, roles // 3)} Python, {max(1, roles // 3)} React, 1 Go",
    ]
    role_detail = role_templates[idx % 3]
    layoff = "none" if rng.random() > 0.3 else f"{rng.randint(5, 20)}% headcount cut ({rng.randint(10, 60)} days ago)"
    return (
        f"Company: {name} ({spec[0]}, ${amount}M, closed {days} days ago). "
        f"Open engineering roles: {roles} ({role_detail}). Employee count: {employees}. "
        f"AI maturity score: {ai_score} (confidence: {conf}, {conf_label}). "
        f"Signal age: {days} days. Layoff signal: {layoff}."
    )


def make_bench(bench_state):
    s = BENCH_STATES[bench_state]
    return (
        f"Available bench: {s[0]} Python engineers ({s[1]}), {s[2]} DevOps engineers, "
        f"{s[3]} Go engineer(s), {s[4]} Rust engineers. Next availability window: {s[5]} weeks."
    )


def make_thread(dim, idx):
    if dim == "bench_fit_alignment":
        return PRIOR_THREADS["bench_ask"]
    if dim == "tone_compliance":
        return [PRIOR_THREADS["hostile"], PRIOR_THREADS["pricing"], PRIOR_THREADS["cold"]][idx % 3]
    if dim == "cta_quality":
        return [PRIOR_THREADS["implicit_book"], PRIOR_THREADS["warm"], PRIOR_THREADS["cold"]][idx % 3]
    return [PRIOR_THREADS["cold"], PRIOR_THREADS["warm"], PRIOR_THREADS["cold"]][idx % 3]


def make_expected(dim, conf):
    banned = BASE_BANNED + EXTRA_BANNED_BY_DIM.get(dim, [])
    feats = {
        "must_reference_signal": dim in ("signal_grounding", "personalization"),
        "must_include_cta": True,
        "banned_phrases": banned,
    }
    if dim == "signal_grounding" and conf < 0.75:
        feats["required_hedges"] = ["are you finding", "it looks like", "appears", "from what we can see"]
    if dim == "bench_fit_alignment":
        feats["max_headcount_commitment"] = None
        feats["must_reference_signal"] = False
    return feats


def make_task(task_id, dim, source_mode, difficulty, idx):
    name = COMPANY_NAMES[idx % len(COMPANY_NAMES)]
    sizes = ["seed", "series_a", "series_b"]
    size = sizes[idx % 3]
    confs = [0.62, 0.75, 0.87]
    conf = confs[idx % 3]
    bench_states = ["full", "partial", "empty_for_required"]
    bench_state = bench_states[idx % 3]

    return {
        "task_id": task_id,
        "dimension": dim,
        "difficulty": difficulty,
        "source_mode": source_mode,
        "input": {
            "company_signal": make_signal(name, size, conf, dim, idx),
            "bench_summary": make_bench(bench_state),
            "prior_thread": make_thread(dim, idx),
        },
        "expected_features": make_expected(dim, conf),
        "rubric": {
            "signal_grounding": 0,
            "tone_alignment": 0,
            "cta_quality": 0,
            "bench_fit_accuracy": 0,
            "personalization_depth": 0,
        },
        "gold_output": None,
        "notes": f"source_mode={source_mode}, size={size}, conf={conf}, bench={bench_state}",
    }


def weighted_modes(count):
    result = []
    for mode, w in SOURCE_MODES_WEIGHTS:
        n = max(1, round(count * w / 100))
        result.extend([mode] * n)
    # Trim or pad to exact count
    while len(result) > count:
        result.pop()
    while len(result) < count:
        result.append("programmatic")
    rng.shuffle(result)
    return result


def make_partition(count, id_prefix, start_idx=0):
    tasks = []
    dim_cycle = DIMENSIONS * (count // len(DIMENSIONS) + 2)
    diff_cycle = ["easy", "easy", "medium", "medium", "hard"] * (count // 5 + 2)
    mode_list = weighted_modes(count)

    for i in range(count):
        dim = dim_cycle[i]
        difficulty = diff_cycle[i]
        source_mode = mode_list[i]
        task_id = f"tb_{id_prefix}_{i + 1:04d}"
        task = make_task(task_id, dim, source_mode, difficulty, start_idx + i)
        tasks.append(task)
    return tasks


os.makedirs("tenacious_bench_v0.1/train", exist_ok=True)
os.makedirs("tenacious_bench_v0.1/dev", exist_ok=True)
os.makedirs("tenacious_bench_v0.1/held_out", exist_ok=True)

train_tasks = make_partition(100, "train", 0)
dev_tasks = make_partition(60, "dev", 100)
held_tasks = make_partition(40, "held", 160)

for path, tasks in [
    ("tenacious_bench_v0.1/train/tasks.jsonl", train_tasks),
    ("tenacious_bench_v0.1/dev/tasks.jsonl", dev_tasks),
    ("tenacious_bench_v0.1/held_out/tasks.jsonl", held_tasks),
]:
    with open(path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"Wrote {len(tasks)} tasks -> {path}")

# Count dimensions per partition
for label, tasks in [("train", train_tasks), ("dev", dev_tasks), ("held_out", held_tasks)]:
    dims = {}
    modes = {}
    for t in tasks:
        dims[t["dimension"]] = dims.get(t["dimension"], 0) + 1
        modes[t["source_mode"]] = modes.get(t["source_mode"], 0) + 1
    print(f"\n{label} dims: {dims}")
    print(f"{label} modes: {modes}")
