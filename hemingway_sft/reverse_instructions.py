"""Generate an instruction for every training passage, then emit SFT pairs.

Fine-tuning on raw novel text produces a model that continues Hemingway
rather than one that answers a request in his voice. Reverse instructions
avoid that. An existing model reads each human-written passage and writes
the instruction that would have produced it, and the passage becomes the
target completion.

The technique is from Köksal et al. 2023, "LongForm: Effective Instruction
Tuning with Reverse Instructions". The README separates it from the two
neighbouring techniques it gets mistaken for.
"""

from __future__ import annotations

import argparse
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel, ValidationError

from hemingway_sft.corpus import Passage, read_jsonl

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_CONCURRENCY = 8
MAX_TOKENS = 2048
MAX_RETRIES = 5
VALIDATION_STRIDE = 20

SYSTEM_PROMPT = """You read a passage from a novel and recover the instruction a \
person would have typed to get it.

Write the instruction so that:

- It names the situation, the people, and the setting concretely, in one or two \
imperative sentences.
- It reads as an ordinary request for prose, with no mention of the author, the \
book, or any quality of the writing style. An instruction that asks for terse \
sentences or names Hemingway defeats the purpose, because the model must learn \
the voice as its default rather than as a style it switches on when asked.
"""


class ReverseInstruction(BaseModel):
    instruction: str


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class TrainingExample(BaseModel):
    prompt: list[ChatMessage]
    completion: list[ChatMessage]


class CachedInstruction(BaseModel):
    index: int
    instruction: str


def instruction_for(client: anthropic.Anthropic, model: str, passage: Passage) -> str:
    response = client.messages.parse(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        output_config={"effort": "low"},
        output_format=ReverseInstruction,
        messages=[{"role": "user", "content": f"<passage>\n{passage.text}\n</passage>"}],
    )
    parsed = response.parsed_output
    if parsed is None:
        raise ValueError(
            f"model returned no parsed instruction, stop_reason={response.stop_reason}"
        )
    instruction = parsed.instruction.strip()
    if not instruction:
        raise ValueError("model returned an empty instruction")
    return instruction


def load_cache(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        rows = [CachedInstruction.model_validate_json(line) for line in handle if line.strip()]
    return {row.index: row.instruction for row in rows}


def collect(
    passages: list[Passage],
    cache_path: Path,
    model: str,
    concurrency: int,
) -> dict[int, str]:
    """Fill the cache with one instruction per passage and return every result.

    Each result is appended as soon as it arrives, so an interrupted run
    resumes instead of paying for the same passages twice.
    """
    cached = load_cache(cache_path)
    pending = [i for i in range(len(passages)) if i not in cached]
    if not pending:
        return cached

    client = anthropic.Anthropic(max_retries=MAX_RETRIES)
    write_lock = threading.Lock()
    failures = 0

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as handle:

        def worker(index: int) -> None:
            nonlocal failures
            try:
                instruction = instruction_for(client, model, passages[index])
            except (anthropic.APIError, ValidationError, ValueError) as err:
                with write_lock:
                    failures += 1
                    print(f"passage {index} failed: {err}")
                return
            row = CachedInstruction(index=index, instruction=instruction)
            with write_lock:
                cached[index] = instruction
                handle.write(row.model_dump_json() + "\n")
                handle.flush()
                done = len(cached)
                if done % 25 == 0:
                    print(f"{done}/{len(passages)} instructions")

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(worker, pending))

    if failures:
        print(f"{failures} passages failed and were skipped")
    return cached


def to_examples(passages: list[Passage], instructions: dict[int, str]) -> list[TrainingExample]:
    return [
        TrainingExample(
            prompt=[ChatMessage(role="user", content=instructions[index])],
            completion=[ChatMessage(role="assistant", content=passage.text)],
        )
        for index, passage in enumerate(passages)
        if index in instructions
    ]


def write_jsonl(examples: Iterable[TrainingExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(example.model_dump_json() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    passages = [p for p in read_jsonl(args.data_dir / "passages.jsonl") if p.split == "train"]
    if args.limit is not None:
        passages = passages[: args.limit]

    instructions = collect(
        passages,
        args.data_dir / "instructions.jsonl",
        args.model,
        args.concurrency,
    )
    examples = to_examples(passages, instructions)

    validation = examples[::VALIDATION_STRIDE]
    training = [e for i, e in enumerate(examples) if i % VALIDATION_STRIDE != 0]
    write_jsonl(training, args.data_dir / "train.jsonl")
    write_jsonl(validation, args.data_dir / "valid.jsonl")

    print(f"train: {len(training)} examples")
    print(f"valid: {len(validation)} examples")


if __name__ == "__main__":
    main()
