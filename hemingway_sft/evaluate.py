"""Compare the tuned adapter against the base model on unseen prompts.

Two checks run here. A style profile measures the generated prose against
the held-out book, and a side-by-side generation lets a reader judge the
voice directly. Reading the samples catches failures the numbers miss, so
every sample prints in full.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from hemingway_sft.corpus import read_jsonl
from hemingway_sft.style import format_profile, profile

DEFAULT_MODEL = "google/gemma-4-12B-it"
MAX_NEW_TOKENS = 320
TEMPERATURE = 0.9
TOP_P = 0.9

EVAL_PROMPTS: tuple[str, ...] = (
    "Write a scene where two men wait at a railway station in the rain.",
    "Describe a man cleaning a fish beside a river in the early morning.",
    "Write a conversation between a soldier and a nurse about whether the war will end.",
    "Describe walking into a cafe in a foreign city where nobody knows you.",
    "Write a scene where a woman tells a man she is leaving, in a hotel room.",
    "Describe the last hour of a long drive through dry country.",
)


def load_model(base: str, adapter: Path | None) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model, tokenizer


def generate(model: Any, tokenizer: Any, prompt: str) -> str:
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id,
        )
    completion = output[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(completion, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=Path("runs/hemingway-12b"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    passages = read_jsonl(args.data_dir / "passages.jsonl")
    reference = "\n\n".join(p.text for p in passages if p.split == "heldout")
    print(format_profile("hemingway", profile(reference)))

    for label, adapter in (("base", None), ("tuned", args.adapter)):
        model, tokenizer = load_model(args.model, adapter)
        samples = [generate(model, tokenizer, prompt) for prompt in EVAL_PROMPTS]
        print(format_profile(label, profile("\n\n".join(samples))))
        print()
        for prompt, sample in zip(EVAL_PROMPTS, samples, strict=True):
            print(f"--- {label}: {prompt}")
            print(sample)
            print()


if __name__ == "__main__":
    main()
