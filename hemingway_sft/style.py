"""The before-and-after measurement for a training run.

Post-training is data, an algorithm, an objective, and a way to tell whether
the objective moved. This module is the fourth. Two numbers carry most of
what separates Hemingway from an assistant voice.

A reward function would score the same two numbers, which is why they sit
apart from the script that loads a model.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

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
