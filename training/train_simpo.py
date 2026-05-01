"""
SimPO preference training script for Tenacious-Bench judge.
Uses TRL's DPOTrainer with loss_type='simpo' — handles gradient flow correctly.
Target: Colab T4 (15GB VRAM), Qwen2.5-1.5B-Instruct backbone, ≤30 min wall time.

Run:
    python train_simpo.py \
        --pairs preference_pairs.jsonl \
        --model unsloth/Qwen2.5-1.5B-Instruct \
        --output qwen_simpo_judge \
        --epochs 3 \
        --beta 2.0 \
        --gamma 0.5 \
        --lr 5e-5 \
        --seed 42

Kill criterion: if training loss has not dropped below 0.65 within 10 minutes, halt.
"""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a scoring judge evaluating B2B sales outreach emails for Tenacious, "
    "an AI talent platform. Given a task and a candidate email, score it on five "
    "dimensions: signal_grounding, tone_compliance, cta_quality, "
    "personalization_depth, and bench_fit_alignment."
)


def load_pairs(path: str) -> list[dict]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    log.info("Loaded %d preference pairs from %s", len(pairs), path)
    return pairs


def format_prompt(task: dict) -> str:
    """Build the judge prompt for a task."""
    inp = task.get("input", {})
    return (
        f"Task context:\n"
        f"Company signal: {inp.get('company_signal', '')}\n"
        f"Bench: {inp.get('bench_summary', '')}\n"
        f"Prior thread: {inp.get('prior_thread', '') or 'None'}\n\n"
        f"Score the following email on signal_grounding, tone_compliance, "
        f"cta_quality, personalization_depth, and bench_fit_alignment (0-3 each).\n\n"
        f"Email:\n"
    )


def pairs_to_hf_dataset(pairs: list[dict], tokenizer):
    """Convert preference pairs to HuggingFace dataset format for TRL DPOTrainer."""
    from datasets import Dataset

    records = []
    for p in pairs:
        prompt = format_prompt(p["task"])
        chosen_text = p["chosen"]["output"]
        rejected_text = p["rejected"]["output"]

        # Apply chat template to get formatted strings
        chosen_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt + chosen_text},
        ]
        rejected_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt + rejected_text},
        ]

        records.append({
            "prompt": prompt,
            "chosen": chosen_text,
            "rejected": rejected_text,
            "chosen_score": p["chosen"]["final_score"],
            "rejected_score": p["rejected"]["final_score"],
            "score_gap": p["score_gap"],
        })

    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser(description="Train SimPO preference judge via TRL")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--model", default="unsloth/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", default="qwen_simpo_judge")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    # Load model via Unsloth
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=1024,
            dtype=None,  # auto: float16 on T4, bfloat16 on Ampere+
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )
        log.info("Loaded model via Unsloth (4-bit + LoRA)")
    except Exception as e:
        log.error("Unsloth load failed: %s", e)
        raise

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and format preference pairs
    pairs = load_pairs(args.pairs)
    dataset = pairs_to_hf_dataset(pairs, tokenizer)
    log.info("Dataset: %d examples", len(dataset))

    # TRL DPOTrainer with SimPO loss
    try:
        from trl import DPOTrainer, DPOConfig

        training_args = DPOConfig(
            output_dir=args.output,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            beta=args.beta,
            loss_type="simpo",
            simpo_gamma=args.gamma,
            max_length=1024,
            max_prompt_length=512,
            logging_steps=1,
            save_strategy="no",
            seed=args.seed,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            report_to="none",
            remove_unused_columns=False,
        )

        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )

        log.info(
            "Starting SimPO training via TRL DPOTrainer: "
            "β=%.1f γ=%.2f lr=%.0e epochs=%d pairs=%d",
            args.beta, args.gamma, args.lr, args.epochs, len(pairs)
        )
        start = time.time()
        train_result = trainer.train()
        elapsed = time.time() - start

        final_loss = train_result.training_loss
        log.info("Training complete. Final loss: %.4f | Time: %.0fs", final_loss, elapsed)

        if final_loss > 0.65:
            log.warning(
                "KILL CRITERION: final loss %.4f > 0.65. "
                "Training did not converge. Check preference pair quality.",
                final_loss
            )

    except (ImportError, TypeError) as e:
        # Fallback: try CPOTrainer with simpo loss_type
        log.warning("DPOConfig simpo not available (%s), trying CPOTrainer...", e)
        from trl import CPOTrainer, CPOConfig

        training_args = CPOConfig(
            output_dir=args.output,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            beta=args.beta,
            loss_type="simpo",
            simpo_gamma=args.gamma,
            max_length=1024,
            max_prompt_length=512,
            logging_steps=1,
            save_strategy="no",
            seed=args.seed,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            report_to="none",
        )

        trainer = CPOTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )

        log.info("Starting SimPO training via TRL CPOTrainer")
        start = time.time()
        train_result = trainer.train()
        elapsed = time.time() - start
        final_loss = train_result.training_loss
        log.info("Training complete. Final loss: %.4f | Time: %.0fs", final_loss, elapsed)

    # Save adapter and run log
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    run_log = {
        "model": args.model,
        "epochs": args.epochs,
        "beta": args.beta,
        "gamma": args.gamma,
        "lr": args.lr,
        "seed": args.seed,
        "n_pairs": len(pairs),
        "final_loss": round(final_loss, 4),
        "total_elapsed_s": round(elapsed, 1),
        "converged": final_loss <= 0.65,
    }
    with open(output_path / "run_log.json", "w") as f:
        json.dump(run_log, f, indent=2)

    log.info("Adapter saved to %s", output_path)
    log.info("Final loss: %.4f | Converged: %s", final_loss, run_log["converged"])


if __name__ == "__main__":
    main()
