"""
Push the trained SimPO LoRA adapter to HuggingFace Hub.

Run:
    huggingface-cli login   # paste your write token
    python training/push_adapter.py --adapter training/qwen_simpo_judge --repo your-hf-handle/tenacious-judge-v0.1
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Local adapter path")
    parser.add_argument(
        "--repo",
        required=True,
        help="HuggingFace repo ID, e.g. your-handle/tenacious-judge-v0.1",
    )
    args = parser.parse_args()

    adapter_path = Path(args.adapter)
    run_log_path = adapter_path / "run_log.json"

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Install transformers and peft: pip install transformers peft")
        raise

    print(f"Loading adapter from {adapter_path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    model = AutoModelForCausalLM.from_pretrained(str(adapter_path))

    print(f"Pushing to {args.repo}...")
    model.push_to_hub(args.repo, private=False)
    tokenizer.push_to_hub(args.repo, private=False)

    if run_log_path.exists():
        with open(run_log_path) as f:
            run_log = json.load(f)
        print(
            f"Training summary: final_loss={run_log.get('final_loss')}, "
            f"elapsed={run_log.get('total_elapsed_s')}s, "
            f"pairs={run_log.get('n_pairs')}"
        )

    print(f"Adapter published at: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
