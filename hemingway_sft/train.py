"""LoRA supervised fine-tuning of Gemma 4 E4B on the generated pairs.

Style is a per-token property of the training completions, so plain
supervised fine-tuning is the right tool and one stage is enough. The
dataset is in conversational prompt-completion form, which makes TRL mask
the instruction and score only the passage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

DEFAULT_MODEL = "google/gemma-4-E4B-it"
SMOKE_EXAMPLES = 20

# The Per-Layer Embedding tables are lookup tables rather than transforms, so
# naming the projections keeps rank where it can change the model's behavior.
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def lora_config(rank: int) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(LORA_TARGET_MODULES),
    )


def training_config(output_dir: Path, epochs: int, learning_rate: float) -> SFTConfig:
    return SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        max_length=1024,
        packing=False,
        completion_only_loss=True,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        report_to=[],
    )


def build_trainer(
    model: str,
    data_dir: Path,
    output_dir: Path,
    epochs: int,
    learning_rate: float,
    rank: int,
    smoke: bool = False,
) -> SFTTrainer:
    train_dataset = load_dataset("json", data_files=str(data_dir / "train.jsonl"), split="train")
    eval_dataset = load_dataset("json", data_files=str(data_dir / "valid.jsonl"), split="train")
    if smoke:
        train_dataset = train_dataset.select(range(min(SMOKE_EXAMPLES, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(SMOKE_EXAMPLES, len(eval_dataset))))
    return SFTTrainer(
        model=model,
        args=training_config(output_dir, epochs, learning_rate),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config(rank),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/hemingway-e4b"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Train on a handful of examples to prove the model loads and steps.",
    )
    args = parser.parse_args()

    trainer = build_trainer(
        args.model,
        args.data_dir,
        args.output_dir,
        args.epochs,
        args.learning_rate,
        args.lora_rank,
        args.smoke,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    print(f"adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
