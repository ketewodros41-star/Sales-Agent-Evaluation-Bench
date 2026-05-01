"""
scoring_evaluator.py — Tenacious-Bench v0.1 deterministic scorer.
Loads a benchmark task JSON, accepts a candidate output string,
and returns a numerical score dict with no human in the loop.

Usage:
    python scoring_evaluator.py --task path/to/task.json --output "candidate text"
    python scoring_evaluator.py --task path/to/task.json --output-file path/to/output.txt
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

# Regex patterns for CTA detection
CTA_PATTERNS = [
    r"calendly\.com",
    r"cal\.com",
    r"book\s+a\s+call",
    r"schedule\s+(a\s+)?(quick\s+)?(call|chat|meeting|conversation)",
    r"would\s+(you\s+)?be\s+open\s+to",
    r"happy\s+to\s+find\s+a\s+time",
    r"grab\s+(a\s+)?(quick\s+)?slot",
    r"\d+\s*[-–]\s*minute\s+call",
    r"pick\s+a\s+time",
    r"let\s+me\s+know\s+if\s+you.{0,20}open",
]

# Banned phrase patterns — exact list from Tenacious Style Guide v2 "Banned Phrases" section
DEFAULT_BANNED_PHRASES = [
    "world-class",
    "top talent",
    "a-players",
    "rockstar",
    "ninja",
    "wizard",
    "skyrocket",
    "supercharge",
    "10x",
    "i hope this email finds you well",
    "just following up",
    "circling back",
    "quick question",
    "quick chat",
    "synergize",
    "synergy",
    "leverage",
    "ecosystem",
    "game-changer",
    "game changer",
    "disruptor",
    "paradigm shift",
    "you'll regret missing this",
    "don't miss out",
    "per my last email",
    "you're falling behind",
    "behind the curve",
]

# Hedging patterns required when signal confidence is low
HEDGE_PATTERNS = [
    r"are\s+you\s+finding",
    r"it\s+(looks|appears)\s+like",
    r"from\s+what\s+we\s+(can\s+)?see",
    r"it\s+seems\s+(like\s+)?you",
    r"based\s+on\s+public\s+(signals|data|information)",
    r"publicly\s+available\s+(data|signals)",
    r"appears\s+to\s+be",
    r"we\s+noticed",
]

# Signals that indicate bench over-commitment
HEADCOUNT_COMMITMENT_PATTERNS = [
    r"\b\d+\s+(engineers?|developers?|specialists?)\s+(can|could|will|would)\s+start",
    r"we\s+have\s+\d+\s+(engineers?|developers?)\s+available\s+immediately",
    r"can\s+start\s+(this\s+)?(monday|today|tomorrow|immediately|right\s+away)",
    r"\d+\s+(engineers?|developers?)\s+ready\s+to\s+go",
]

# Delivery-lead routing patterns (correct bench-fit behaviour)
ROUTING_PATTERNS = [
    r"delivery\s+lead",
    r"loop\s+in",
    r"connect\s+you\s+with",
    r"let\s+(me|us)\s+verify",
    r"confirm\s+(capacity|availability)",
    r"check\s+with\s+(our\s+)?team",
]


# ---------------------------------------------------------------------------
# Core scoring functions
# ---------------------------------------------------------------------------

def score_signal_grounding(output: str, task: dict[str, Any]) -> int:
    """
    Score 0–3:
    0 = no signal reference or wrong signal
    1 = generic reference (company name only, no signal detail)
    2 = specific accurate signal reference
    3 = specific signal reference with appropriate confidence hedging
    """
    company_signal = task["input"].get("company_signal", "")
    output_lower = output.lower()

    # Extract key signal tokens from company_signal
    signal_tokens = _extract_signal_tokens(company_signal)

    # Check if any signal token appears in output
    matches = [tok for tok in signal_tokens if tok.lower() in output_lower]
    if not matches:
        return 0

    # If only company name matches (too generic), score 1
    company_name = _extract_company_name(company_signal)
    non_name_matches = [m for m in matches if m.lower() != company_name.lower()]
    if not non_name_matches:
        return 1

    # Check if hedging is required and present
    required_hedges = task.get("expected_features", {}).get("required_hedges", [])
    if required_hedges:
        hedge_present = _check_hedge_present(output_lower)
        return 3 if hedge_present else 2

    return 2


def score_tone_alignment(output: str, task: dict[str, Any]) -> int:
    """
    Score 0–3:
    0 = banned phrase detected
    1 = no banned phrases but generic/template tone, OR word count exceeds limit
    2 = Tenacious-voice compliant (direct, no filler, within word count)
    3 = peer-to-peer expert register
    """
    output_lower = output.lower()

    # Collect all banned phrases (task-specific + defaults)
    banned = task.get("expected_features", {}).get("banned_phrases", []) + DEFAULT_BANNED_PHRASES
    for phrase in banned:
        if phrase.lower() in output_lower:
            return 0

    # Word count check: cold ≤ 120, warm reply ≤ 200 (prior_thread non-empty = warm)
    prior_thread = task.get("input", {}).get("prior_thread", "")
    word_limit = 200 if prior_thread else 120
    word_count = len(output.split())
    if word_count > word_limit:
        return 1

    # Check for template / filler language
    filler_patterns = [
        r"i\s+am\s+reaching\s+out\s+to",
        r"touch\s+base",
        r"i\s+wanted\s+to\s+check\s+in",
        r"best\s+in\s+class\b",
        r"end[- ]to[- ]end\s+solution",
        r"value[- ]add",
        r"add\s+value\s+to",
    ]
    for pat in filler_patterns:
        if re.search(pat, output_lower):
            return 1

    # Check for peer-to-peer indicators
    peer_patterns = [
        r"\b(we|our\s+team)\s+(worked|built|shipped|deployed)\b",
        r"\b(I\s+)?(noticed|saw|found)\s+(that\s+)?you\b",
        r"curious\s+(if|whether|how)",
        r"worth\s+a\s+(quick\s+)?conversation",
        r"thought\s+it\s+might\s+be\s+relevant",
    ]
    peer_hits = sum(1 for p in peer_patterns if re.search(p, output_lower))
    return 3 if peer_hits >= 2 else 2


def score_cta_quality(output: str, task: dict[str, Any]) -> int:
    """
    Score 0–3:
    0 = forced booking (books without consent) or no CTA when required
    1 = generic CTA ("let me know if interested")
    2 = stage-appropriate CTA (slot offer, not booking)
    3 = stage-appropriate CTA with friction-reducing specifics (duration, topic, link)
    """
    must_include_cta = task.get("expected_features", {}).get("must_include_cta", True)
    output_lower = output.lower()

    # Check for forced booking (worst case)
    forced_booking_patterns = [
        r"i.ve\s+(gone\s+ahead\s+and\s+)?booked\s+(a|you|us)",
        r"i\s+took\s+the\s+liberty\s+of\s+booking",
        r"your\s+calendar\s+invite\s+is\s+(on\s+its\s+way|sent)",
    ]
    for pat in forced_booking_patterns:
        if re.search(pat, output_lower):
            return 0

    # Check CTA presence
    cta_found = any(re.search(pat, output_lower) for pat in CTA_PATTERNS)
    if not cta_found:
        return 0 if must_include_cta else 2

    # Check for friction-reducing specifics
    specifics_patterns = [
        r"\d+\s*[-–]\s*minute",
        r"cal\.com",
        r"calendly\.com",
        r"(monday|tuesday|wednesday|thursday|friday)\s+or\s+(monday|tuesday|wednesday|thursday|friday)",
        r"next\s+week\s+(or\s+the\s+week\s+after)?",
        r"works\s+for\s+you",
    ]
    specifics_hits = sum(1 for p in specifics_patterns if re.search(p, output_lower))
    return 3 if specifics_hits >= 2 else 2


def score_bench_fit(output: str, task: dict[str, Any]) -> int:
    """
    Score 0–3:
    0 = "bench" used in prospect-facing output (Professional violation), or impossible headcount commitment
    1 = bench not referenced when required
    2 = capacity referenced without over-commitment
    3 = bench-gated routing with delivery-lead handoff
    """
    output_lower = output.lower()

    # "bench" in prospect-facing output is a Professional-marker violation (Style Guide v2)
    if re.search(r"\bbench\b", output_lower):
        return 0

    # Check for headcount over-commitment
    for pat in HEADCOUNT_COMMITMENT_PATTERNS:
        if re.search(pat, output_lower):
            return 0

    bench_summary = task["input"].get("bench_summary", "")
    requires_bench = "0" in bench_summary or "Rust" in bench_summary

    # Check for routing behaviour (positive signal: routes to human/delivery lead)
    routing_present = any(re.search(p, output_lower) for p in ROUTING_PATTERNS)
    if routing_present:
        return 3

    # Capacity/availability mentioned without commitment
    capacity_mention = re.search(r"\b(capacity|availability|engineer|team)\b", output_lower)
    if capacity_mention:
        return 2

    return 1 if requires_bench else 2


def score_personalization(output: str, task: dict[str, Any]) -> int:
    """
    Score 0–3:
    0 = purely generic (no company name, no signal)
    1 = company name only
    2 = one signal detail referenced
    3 = multi-signal synthesis with inference shown
    """
    output_lower = output.lower()
    company_signal = task["input"].get("company_signal", "")

    company_name = _extract_company_name(company_signal)
    signal_tokens = _extract_signal_tokens(company_signal)

    # No company name → purely generic
    if company_name and company_name.lower() not in output_lower:
        return 0

    # Count distinct signal references
    signal_hits = [tok for tok in signal_tokens if tok.lower() in output_lower and tok.lower() != company_name.lower()]

    if not signal_hits:
        return 1
    if len(signal_hits) >= 3:
        return 3
    return 2


# ---------------------------------------------------------------------------
# Penalty functions
# ---------------------------------------------------------------------------

def banned_phrase_penalty(output: str, task: dict[str, Any]) -> int:
    """Returns count of banned phrases found in output."""
    output_lower = output.lower()
    banned = task.get("expected_features", {}).get("banned_phrases", []) + DEFAULT_BANNED_PHRASES
    return sum(1 for phrase in banned if phrase.lower() in output_lower)


# ---------------------------------------------------------------------------
# Final scoring
# ---------------------------------------------------------------------------

def compute_final_score(scores: dict[str, int], penalty: int) -> float:
    """
    Weighted average of dimension scores (each 0–3, normalized to 0–1)
    minus penalty deduction.

    Weights: signal_grounding 0.30, tone_alignment 0.25, cta_quality 0.20,
             bench_fit_accuracy 0.15, personalization_depth 0.10
    Penalty: -0.15 per banned phrase found (floor 0.0)
    """
    weights = {
        "signal_grounding": 0.30,
        "tone_alignment": 0.25,
        "cta_quality": 0.20,
        "bench_fit_accuracy": 0.15,
        "personalization_depth": 0.10,
    }
    weighted_sum = sum(
        (scores.get(dim, 0) / 3.0) * w
        for dim, w in weights.items()
    )
    deduction = penalty * 0.15
    return max(0.0, round(weighted_sum - deduction, 4))


def score_task(task: dict[str, Any], candidate_output: str) -> dict[str, Any]:
    """
    Main entry point. Returns structured scoring result.
    Gracefully handles missing task fields and empty/malformed candidate output.
    """
    if not isinstance(candidate_output, str):
        candidate_output = ""
    output = candidate_output.strip()
    task = task if isinstance(task, dict) else {}
    # Ensure the nested "input" key always exists so dimension functions can safely call .get()
    if "input" not in task:
        task = dict(task, input={})

    sg = score_signal_grounding(output, task)
    ta = score_tone_alignment(output, task)
    cq = score_cta_quality(output, task)
    bf = score_bench_fit(output, task)
    pd_ = score_personalization(output, task)
    bp = banned_phrase_penalty(output, task)

    scores = {
        "signal_grounding": sg,
        "tone_alignment": ta,
        "cta_quality": cq,
        "bench_fit_accuracy": bf,
        "personalization_depth": pd_,
    }

    result = {
        "task_id": task.get("task_id", "unknown"),
        "dimension": task.get("dimension", "unknown"),
        "signal_grounding": sg,
        "tone_alignment": ta,
        "cta_quality": cq,
        "bench_fit_accuracy": bf,
        "personalization_depth": pd_,
        "cta_present": 1 if cq > 0 else 0,
        "banned_phrase_penalty": bp,
        "final_score": compute_final_score(scores, bp),
    }
    return result


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _extract_company_name(company_signal: str) -> str:
    match = re.search(r"Company:\s*([^(,\n]+)", company_signal)
    if match:
        return match.group(1).strip()
    return ""


def _extract_signal_tokens(company_signal: str) -> list[str]:
    """Extract key searchable tokens from a company_signal string."""
    tokens = []

    company_name = _extract_company_name(company_signal)
    if company_name:
        tokens.append(company_name)

    # Funding amount / round
    funding_match = re.search(r"(Series [A-C]|\$[\d.]+M)", company_signal)
    if funding_match:
        tokens.append(funding_match.group(1))

    # Role count
    role_match = re.search(r"(\d+)\s+open\s+engineering\s+roles?", company_signal)
    if role_match:
        tokens.append(role_match.group(1))

    # AI maturity score
    ai_match = re.search(r"AI\s+maturity\s+score:\s*(\d+)", company_signal)
    if ai_match:
        tokens.append(f"AI maturity")
        tokens.append(ai_match.group(1))

    # Layoff signal
    if "layoff" in company_signal.lower():
        tokens.append("layoff")
        tokens.append("headcount")

    return tokens


def _check_hedge_present(output_lower: str) -> bool:
    return any(re.search(pat, output_lower) for pat in HEDGE_PATTERNS)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Tenacious-Bench v0.1 scoring evaluator")
    parser.add_argument("--task", required=True, help="Path to task JSON file")
    parser.add_argument("--output", default=None, help="Candidate output string")
    parser.add_argument("--output-file", default=None, help="Path to file containing candidate output")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON result")
    args = parser.parse_args()

    task_path = Path(args.task)
    if not task_path.exists():
        print(f"Error: task file not found: {task_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(task_path, "r", encoding="utf-8") as f:
            task = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"Error: task file is not valid JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.output_file:
        with open(args.output_file, "r", encoding="utf-8") as f:
            candidate = f.read()
    elif args.output:
        candidate = args.output
    else:
        print("Error: provide --output or --output-file", file=sys.stderr)
        sys.exit(1)

    result = score_task(task, candidate)

    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent))


if __name__ == "__main__":
    main()
