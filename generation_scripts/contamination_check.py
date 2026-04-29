"""
contamination_check.py — Tenacious-Bench v0.1 contamination verification.

Runs three checks before any task enters the held-out partition:
1. N-gram overlap (8-gram) between held-out and train inputs
2. Embedding similarity (cosine) between held-out and train inputs
3. Duplicate input detection across all partitions

Outputs a contamination report to tenacious_bench_v0.1/contamination_check.json

Usage:
    python contamination_check.py \
        --train ../tenacious_bench_v0.1/train/tasks.jsonl \
        --dev ../tenacious_bench_v0.1/dev/tasks.jsonl \
        --held-out ../tenacious_bench_v0.1/held_out/tasks.jsonl \
        --output ../tenacious_bench_v0.1/contamination_check.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Embedding similarity check requires sentence-transformers (optional dependency)
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NGRAM_SIZE = 8
EMBEDDING_SIMILARITY_THRESHOLD = 0.85
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_tasks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"Warning: {path} not found. Returning empty list.", file=sys.stderr)
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_input_text(task: dict[str, Any]) -> str:
    """Concatenate all input fields for contamination checking."""
    inp = task.get("input", {})
    return " ".join([
        inp.get("company_signal", ""),
        inp.get("bench_summary", ""),
        inp.get("prior_thread", ""),
    ]).strip()


# ---------------------------------------------------------------------------
# Check 1: Exact duplicates
# ---------------------------------------------------------------------------

def check_duplicates(
    train: list[dict], dev: list[dict], held_out: list[dict]
) -> dict[str, Any]:
    all_tasks = [("train", t) for t in train] + [("dev", t) for t in dev] + [("held_out", t) for t in held_out]
    seen: dict[str, list[str]] = defaultdict(list)

    for partition, task in all_tasks:
        text = get_input_text(task)
        key = text.strip().lower()
        seen[key].append(f"{partition}/{task.get('task_id', '?')}")

    duplicates = {k: v for k, v in seen.items() if len(v) > 1}

    cross_partition_dupes = {}
    for key, ids in duplicates.items():
        partitions = {i.split("/")[0] for i in ids}
        if len(partitions) > 1:
            cross_partition_dupes[key[:80] + "..."] = ids

    return {
        "total_duplicates": len(duplicates),
        "cross_partition_duplicates": len(cross_partition_dupes),
        "cross_partition_detail": cross_partition_dupes,
        "status": "fail" if cross_partition_dupes else "pass",
    }


# ---------------------------------------------------------------------------
# Check 2: N-gram overlap
# ---------------------------------------------------------------------------

def get_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    tokens = text.lower().split()
    return set(tuple(tokens[i: i + n]) for i in range(max(0, len(tokens) - n + 1)))


def ngram_overlap(text_a: str, text_b: str, n: int = NGRAM_SIZE) -> float:
    ng_a = get_ngrams(text_a, n)
    ng_b = get_ngrams(text_b, n)
    if not ng_a or not ng_b:
        return 0.0
    intersection = ng_a & ng_b
    return len(intersection) / min(len(ng_a), len(ng_b))


def check_ngram_overlap(
    train: list[dict], held_out: list[dict], n: int = NGRAM_SIZE
) -> dict[str, Any]:
    train_texts = [get_input_text(t) for t in train]
    held_texts = [get_input_text(t) for t in held_out]

    flagged_pairs = []
    max_overlap = 0.0

    for i, h_text in enumerate(held_texts):
        for j, t_text in enumerate(train_texts):
            overlap = ngram_overlap(h_text, t_text, n)
            if overlap > max_overlap:
                max_overlap = overlap
            if overlap > 0.0:
                flagged_pairs.append({
                    "held_out_id": held_out[i].get("task_id", f"held_{i}"),
                    "train_id": train[j].get("task_id", f"train_{j}"),
                    "overlap": round(overlap, 4),
                })

    # Sort by overlap descending
    flagged_pairs.sort(key=lambda x: x["overlap"], reverse=True)
    violations = [p for p in flagged_pairs if p["overlap"] >= 1.0]

    return {
        "ngram_size": n,
        "max_overlap": round(max_overlap, 4),
        "pairs_with_any_overlap": len(flagged_pairs),
        "full_ngram_violations": len(violations),
        "violation_detail": violations[:10],
        "status": "fail" if violations else "pass",
    }


# ---------------------------------------------------------------------------
# Check 3: Embedding similarity
# ---------------------------------------------------------------------------

def check_embedding_similarity(
    train: list[dict], held_out: list[dict], threshold: float = EMBEDDING_SIMILARITY_THRESHOLD
) -> dict[str, Any]:
    if not EMBEDDINGS_AVAILABLE or not NUMPY_AVAILABLE:
        return {
            "status": "skipped",
            "reason": "sentence-transformers or numpy not installed. Run: pip install sentence-transformers numpy",
            "max_similarity": None,
        }

    model = SentenceTransformer(EMBEDDING_MODEL)
    train_texts = [get_input_text(t) for t in train]
    held_texts = [get_input_text(t) for t in held_out]

    if not train_texts or not held_texts:
        return {"status": "skipped", "reason": "empty partition", "max_similarity": None}

    print("Computing embeddings for train partition...", file=sys.stderr)
    train_emb = model.encode(train_texts, normalize_embeddings=True, show_progress_bar=False)
    print("Computing embeddings for held_out partition...", file=sys.stderr)
    held_emb = model.encode(held_texts, normalize_embeddings=True, show_progress_bar=False)

    # Cosine similarity matrix (train_emb already normalized, so dot product = cosine)
    sim_matrix = np.dot(held_emb, train_emb.T)

    max_sim = float(sim_matrix.max())
    violations = []
    for i in range(len(held_out)):
        for j in range(len(train)):
            if sim_matrix[i, j] >= threshold:
                violations.append({
                    "held_out_id": held_out[i].get("task_id", f"held_{i}"),
                    "train_id": train[j].get("task_id", f"train_{j}"),
                    "cosine_similarity": round(float(sim_matrix[i, j]), 4),
                })

    violations.sort(key=lambda x: x["cosine_similarity"], reverse=True)

    return {
        "model": EMBEDDING_MODEL,
        "threshold": threshold,
        "max_similarity": round(max_sim, 4),
        "violations_above_threshold": len(violations),
        "violation_detail": violations[:10],
        "status": "fail" if violations else "pass",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Tenacious-Bench v0.1 contamination checker")
    parser.add_argument("--train", default="../tenacious_bench_v0.1/train/tasks.jsonl")
    parser.add_argument("--dev", default="../tenacious_bench_v0.1/dev/tasks.jsonl")
    parser.add_argument("--held-out", default="../tenacious_bench_v0.1/held_out/tasks.jsonl")
    parser.add_argument("--output", default="../tenacious_bench_v0.1/contamination_check.json")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding similarity check")
    args = parser.parse_args()

    print("Loading partitions...", file=sys.stderr)
    train = load_tasks(Path(args.train))
    dev = load_tasks(Path(args.dev))
    held_out = load_tasks(Path(args.held_out))
    print(f"  train: {len(train)} tasks, dev: {len(dev)} tasks, held_out: {len(held_out)} tasks", file=sys.stderr)

    print("Check 1: Duplicate detection...", file=sys.stderr)
    dup_result = check_duplicates(train, dev, held_out)

    print("Check 2: N-gram overlap (held_out vs train)...", file=sys.stderr)
    ngram_result = check_ngram_overlap(train, held_out)

    if args.skip_embeddings:
        emb_result = {"status": "skipped", "reason": "--skip-embeddings flag set", "max_similarity": None}
    else:
        print("Check 3: Embedding similarity (held_out vs train)...", file=sys.stderr)
        emb_result = check_embedding_similarity(train, held_out)

    overall_status = "pass"
    for result in [dup_result, ngram_result, emb_result]:
        if result.get("status") == "fail":
            overall_status = "fail"
            break

    report = {
        "tenacious_bench_version": "0.1",
        "partition_sizes": {"train": len(train), "dev": len(dev), "held_out": len(held_out)},
        "overall_status": overall_status,
        "check_1_duplicates": dup_result,
        "check_2_ngram_overlap": ngram_result,
        "check_3_embedding_similarity": emb_result,
        "summary": {
            "duplicates": dup_result.get("cross_partition_duplicates", 0),
            "max_ngram_overlap": ngram_result.get("max_overlap", 0.0),
            "max_embedding_similarity": emb_result.get("max_similarity"),
            "status": overall_status,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nContamination report written to {output_path}")
    print(f"Overall status: {overall_status.upper()}")
    print(json.dumps(report["summary"], indent=2))

    sys.exit(0 if overall_status == "pass" else 1)


if __name__ == "__main__":
    main()
