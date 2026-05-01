"""
SimPO preference training — self-contained, no TRL dependency.
Uses transformers + PEFT + BitsAndBytes directly with a manual SimPO loss loop.

Run:
    python train_simpo.py \
        --pairs preference_pairs_v3.jsonl \
        --output qwen_simpo_judge_v3 \
        --epochs 3
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a scoring judge evaluating B2B sales outreach emails for Tenacious, "
    "an AI talent platform. Given a task and a candidate email, score it on five "
    "dimensions: signal_grounding, tone_compliance, cta_quality, "
    "personalization_depth, and bench_fit_alignment."
)


def load_pairs(path: str) -> list:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    log.info("Loaded %d preference pairs from %s", len(pairs), path)
    return pairs


def format_prompt(task: dict) -> str:
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


def length_normalized_logprob(model, inputs: dict, device: str) -> torch.Tensor:
    """Compute length-normalized sum of log-probabilities for a sequence."""
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]          # [1, L-1, V]
    labels = input_ids[:, 1:]                    # [1, L-1]

    log_probs = F.log_softmax(logits, dim=-1)
    token_lp = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [1, L-1]
    return token_lp.sum() / labels.shape[1]      # scalar


def simpo_loss(chosen_lp: torch.Tensor, rejected_lp: torch.Tensor,
               beta: float, gamma: float) -> torch.Tensor:
    """SimPO: -log σ(β * (r_chosen - r_rejected - γ))"""
    return -F.logsigmoid(beta * (chosen_lp - rejected_lp - gamma))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--model", default="unsloth/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--model-revision", default=None,
                        help="HuggingFace model revision/commit hash for reproducibility")
    parser.add_argument("--output", default="qwen_simpo_judge")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Effective batch size per step (default 1 — single pair per update, "
                             "memory-constrained by Colab T4 with 4-bit quantization)")
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=10,
                        help="Linear warmup steps before reaching peak lr (default 10)")
    parser.add_argument("--scheduler", default="cosine",
                        choices=["cosine", "linear", "none"],
                        help="LR scheduler after warmup (default: cosine)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    # ── Load model (transformers + PEFT, no unsloth / no TRL) ──────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    log.info("Loading tokenizer from %s (revision=%s)", args.model, args.model_revision or "latest")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Loading model (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        quantization_config=bnb_config,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    log.info("Model ready — transformers+PEFT (4-bit + LoRA r=16)")

    # ── Data ───────────────────────────────────────────────────────────────
    pairs = load_pairs(args.pairs)

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    total_steps = len(pairs) * args.epochs
    log.info("Hyperparameters: lr=%s batch_size=%d warmup=%d scheduler=%s epochs=%d "
             "beta=%.1f gamma=%.2f lora_r=16 lora_alpha=16 total_steps=%d",
             args.lr, args.batch_size, args.warmup_steps, args.scheduler,
             args.epochs, args.beta, args.gamma, total_steps)

    # LR scheduler: linear warmup then cosine/linear decay
    def get_lr_scale(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / max(args.warmup_steps, 1)
        if args.scheduler == "none":
            return 1.0
        progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        if args.scheduler == "cosine":
            import math
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        # linear
        return max(0.0, 1.0 - progress)

    # ── Training loop ──────────────────────────────────────────────────────
    model.train()
    start = time.time()
    step = 0
    running_loss = 0.0
    margin_history = []

    for epoch in range(args.epochs):
        random.shuffle(pairs)
        epoch_loss = 0.0

        for pair in pairs:
            prompt = format_prompt(pair["task"])
            chosen_text   = prompt + pair["chosen"]["output"]
            rejected_text = prompt + pair["rejected"]["output"]

            chosen_inputs   = tokenizer(chosen_text,   return_tensors="pt",
                                        truncation=True, max_length=1024)
            rejected_inputs = tokenizer(rejected_text, return_tensors="pt",
                                        truncation=True, max_length=1024)

            chosen_lp   = length_normalized_logprob(model, chosen_inputs,   device)
            rejected_lp = length_normalized_logprob(model, rejected_inputs, device)

            loss = simpo_loss(chosen_lp, rejected_lp, args.beta, args.gamma)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update LR according to schedule
            lr_scale = get_lr_scale(step)
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * lr_scale

            step += 1
            epoch_loss  += loss.item()
            running_loss += loss.item()
            margin_history.append((chosen_lp.item() - rejected_lp.item()))

            if step % 10 == 0:
                avg_margin = float(np.mean(margin_history[-10:]))
                log.info("step=%d loss=%.4f rewards/margin=%.4f",
                         step, loss.item(), avg_margin)

        log.info("Epoch %d/%d  avg_loss=%.4f",
                 epoch + 1, args.epochs, epoch_loss / len(pairs))

    elapsed      = time.time() - start
    final_loss   = running_loss / step if step > 0 else 0.0
    final_margin = float(np.mean(margin_history[-10:])) if margin_history else 0.0

    log.info("Training complete | steps=%d loss=%.4f margin=%.4f time=%.0fs",
             step, final_loss, final_margin, elapsed)

    # ── Save adapter ───────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    run_log = {
        "model":           args.model,
        "model_revision":  args.model_revision,
        "epochs":          args.epochs,
        "batch_size":      args.batch_size,
        "beta":            args.beta,
        "gamma":           args.gamma,
        "lr":              args.lr,
        "warmup_steps":    args.warmup_steps,
        "scheduler":       args.scheduler,
        "lora_r":          16,
        "lora_alpha":      16,
        "lora_dropout":    0.0,
        "seed":            args.seed,
        "n_pairs":         len(pairs),
        "total_steps":     step,
        "final_loss":      round(final_loss, 4),
        "final_margin":    round(final_margin, 4),
        "total_elapsed_s": round(elapsed, 1),
        "converged":       final_margin > 0,
    }
    with open(output_path / "run_log.json", "w") as f:
        json.dump(run_log, f, indent=2)

    log.info("Adapter saved → %s", output_path)
    log.info("Converged (margin > 0): %s", run_log["converged"])


if __name__ == "__main__":
    main()
