"""Fold the adapter into the base weights for local inference.

Training leaves a LoRA adapter, which only loads with the base model beside
it and the peft library installed. Merging writes one ordinary model
directory instead, which a laptop runtime can quantize and load on its own.

Merging runs on the CPU because it moves weights rather than computing over
them, so it needs host memory and no GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "google/gemma-4-E4B-it"


def merge(base: str, adapter: Path, output: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer is None:
        raise ValueError(f"no tokenizer found for {base}")
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
    merged = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    merged.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=Path("runs/hemingway-e4b"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/hemingway-e4b-merged"))
    args = parser.parse_args()

    merge(args.model, args.adapter, args.output_dir)
    print(f"merged model written to {args.output_dir}")


if __name__ == "__main__":
    main()
