"""
Run the Week 10 baseline agent on all 100 training tasks.
Uses the same model (openai/gpt-4o-mini) and same system prompt as outreach_generator.py.

Output: training_data/baseline_outputs.jsonl
  One JSON per line: {"task_id": "...", "output": "<email text>", "model": "...", "cost_usd": 0.0}
"""

import json
import os
import sys
import time
from pathlib import Path

# Load env from Week 10
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / "trp week 10" / ".env")

from openai import OpenAI

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("ERROR: OPENROUTER_API_KEY not found")
    sys.exit(1)

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

# Same model as Week 10 agent — this IS the baseline
MODEL = "openai/gpt-4o-mini"

SYSTEM_PROMPT = """You are writing outbound email for Tenacious Consulting and Outsourcing.

Tenacious tone markers:
1. DIRECT: Short sentences. No fluff. Get to the point in the first line.
2. GROUNDED: Every claim references a specific, verifiable signal. No vague assertions.
3. RESPECTFUL: Never condescending. Never assume the prospect doesn't know their market.
4. HONEST UNDER UNCERTAINTY: Low-confidence signals are phrased as questions, not assertions.
5. HUMAN: Reads like a thoughtful person wrote it, not a template.
6. NEVER: 'offshore' or 'outsourcing' in subject lines. Never 'scale faster than your competitors'.
7. ACV REFERENCE: Don't quote specific pricing. Route deeper pricing to human.
8. CHANNEL: Email is for cold outreach. SMS only for scheduling from warm leads who replied.

Rules:
- Every factual claim must be grounded in the provided company_signal
- If a signal has confidence < 0.75, phrase as a question, not an assertion
- Never claim 'aggressive hiring' if fewer than 5 open roles
- Never commit to bench capacity — say 'we have engineers available' not specific counts
- Never mention competitors by name in outreach emails
- Keep emails under 180 words
- Do not fabricate additional case studies beyond those in seed materials"""


def generate_baseline_email(task: dict) -> tuple[str, float]:
    """Generate email using Week 10 agent prompt. Returns (email_text, cost_usd)."""
    inp = task.get("input", {})
    company_signal = inp.get("company_signal", "")
    bench_summary = inp.get("bench_summary", "")
    prior_thread = inp.get("prior_thread", "")

    user_content = f"""Write a signal-grounded outbound email for this prospect.

COMPANY SIGNAL:
{company_signal}

BENCH AVAILABLE:
{bench_summary}
"""
    if prior_thread:
        user_content += f"\nPRIOR THREAD (reply to this context):\n{prior_thread}\n"

    user_content += "\nWrite the email body only (no subject line, no JSON wrapper). Under 200 words."

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=350,
            temperature=0.7,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        email_text = response.choices[0].message.content.strip()

        # Estimate cost: gpt-4o-mini is $0.15/1M input, $0.60/1M output
        usage = response.usage
        cost = (usage.prompt_tokens * 0.15 + usage.completion_tokens * 0.60) / 1_000_000

        return email_text, cost

    except Exception as e:
        print(f"  ERROR generating for {task['task_id']}: {e}")
        return "", 0.0


def main():
    tasks_path = Path(__file__).parent.parent / "tenacious_bench_v0.1" / "train" / "tasks.jsonl"
    output_path = Path(__file__).parent / "baseline_outputs.jsonl"

    tasks = []
    with open(tasks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    print(f"Loaded {len(tasks)} training tasks")
    print(f"Model: {MODEL}")
    print(f"Output: {output_path}")
    print("-" * 50)

    total_cost = 0.0
    success = 0
    failed = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for i, task in enumerate(tasks, 1):
            task_id = task["task_id"]
            print(f"[{i:3d}/100] {task_id} (dim={task.get('dimension','?')}, diff={task.get('difficulty','?')}) ... ", end="", flush=True)

            email, cost = generate_baseline_email(task)

            if email:
                record = {
                    "task_id": task_id,
                    "output": email,
                    "model": MODEL,
                    "cost_usd": round(cost, 6),
                    "dimension": task.get("dimension"),
                    "difficulty": task.get("difficulty"),
                    "source_mode": task.get("source_mode"),
                }
                out.write(json.dumps(record) + "\n")
                out.flush()
                total_cost += cost
                success += 1
                print(f"OK ({len(email.split())} words, ${cost:.5f})")
            else:
                failed += 1
                print("FAILED")

            # Throttle slightly to avoid rate limits
            if i % 10 == 0:
                time.sleep(1)

    print("-" * 50)
    print(f"Done: {success} succeeded, {failed} failed")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
