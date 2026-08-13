"""The before-and-after measurement for a training run.

Post-training is data, an algorithm, an objective, and a way to tell whether
the objective moved. This module is the fourth. Two numbers carry most of
what separates Hemingway from an assistant voice.

A reward function would score the same two numbers, which is why they sit
apart from the script that loads a model.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from hemingway_sft.corpus import read_jsonl

# Two fixed-width branches, because a sentence can end on the terminal mark or
# on a closing quote after it, and Python allows no variable-width lookbehind.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=[.!?][\"'’”])\s+")
_ADVERB = re.compile(r"\b\w+ly\b", re.I)
_WORD = re.compile(r"[A-Za-z’']+")


@dataclass(frozen=True)
class StyleProfile:
    words: int
    mean_sentence_words: float
    adverb_rate: float


def profile(text: str) -> StyleProfile:
    sentences = [s for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    words = _WORD.findall(text)
    if not sentences or not words:
        raise ValueError("cannot profile text with no sentences or no words")

    lengths = [len(_WORD.findall(s)) for s in sentences]
    return StyleProfile(
        words=len(words),
        mean_sentence_words=statistics.mean(lengths),
        adverb_rate=len(_ADVERB.findall(text)) / len(words),
    )


def format_profile(name: str, style: StyleProfile) -> str:
    # Sentence length swings with dialogue density, so the sample size prints
    # beside it and a small sample reads as the estimate it is.
    return (
        f"{name:<10} "
        f"words {style.words:6d}  "
        f"sentence words {style.mean_sentence_words:5.1f}  "
        f"adverbs {style.adverb_rate:.4f}"
    )


def main() -> None:
    """Profile text from a file or standard input, beside the held-out reference.

    Reading standard input lets a local generation be piped straight in, so the
    same numbers are available on a laptop as on the training host.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=None)
    parser.add_argument("--name", default="samples")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    text = args.samples.read_text(encoding="utf-8") if args.samples else sys.stdin.read()
    if not text.strip():
        raise ValueError("no text given")

    passages_path = args.data_dir / "passages.jsonl"
    if passages_path.exists():
        held = [p.text for p in read_jsonl(passages_path) if p.split == "heldout"]
        if held:
            print(format_profile("hemingway", profile("\n\n".join(held))))
    print(format_profile(args.name, profile(text)))
