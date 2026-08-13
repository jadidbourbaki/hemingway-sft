"""Measure whether generated prose quotes the training novels verbatim.

A style model and a memorising model both produce Hemingway-shaped sentences,
so the two are told apart by the length of the longest word span a sample
shares with the corpus it trained on.

The threshold comes from Hemingway himself. *In Our Time* never trained the
model, and its longest shared span with the three training novels is seven
words. It shares no eight word span at all. Eight words is therefore where a
match stops being ordinary English or an author's own habit.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from hemingway_sft.corpus import read_jsonl

MEMORISED_SPAN_WORDS = 8

# A shorter seed finds every candidate but costs more extension work, and a
# longer one would miss a match that starts below the threshold.
SEED_WORDS = 6

_WORD = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class Span:
    words: int
    text: str

    @property
    def memorised(self) -> bool:
        return self.words >= MEMORISED_SPAN_WORDS


@dataclass(frozen=True)
class CorpusIndex:
    tokens: tuple[str, ...]
    seeds: dict[tuple[str, ...], tuple[int, ...]]


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def build_index(text: str) -> CorpusIndex:
    tokens = tuple(tokenize(text))
    seeds: dict[tuple[str, ...], list[int]] = {}
    for i in range(len(tokens) - SEED_WORDS + 1):
        seeds.setdefault(tokens[i : i + SEED_WORDS], []).append(i)
    return CorpusIndex(tokens=tokens, seeds={k: tuple(v) for k, v in seeds.items()})


def longest_shared_span(sample: str, index: CorpusIndex) -> Span:
    """Find the longest run of words the sample and the corpus have in common.

    Each seed hit is extended word by word, so a span longer than the seed is
    measured at its true length rather than rounded to the seed.
    """
    sample_tokens = tokenize(sample)
    corpus = index.tokens
    best = Span(words=0, text="")

    for i in range(len(sample_tokens) - SEED_WORDS + 1):
        seed = tuple(sample_tokens[i : i + SEED_WORDS])
        for j in index.seeds.get(seed, ()):
            n = SEED_WORDS
            while (
                i + n < len(sample_tokens)
                and j + n < len(corpus)
                and sample_tokens[i + n] == corpus[j + n]
            ):
                n += 1
            if n > best.words:
                best = Span(words=n, text=" ".join(sample_tokens[i : i + n]))
    return best


def report(samples: list[str], index: CorpusIndex) -> list[Span]:
    spans = [longest_shared_span(s, index) for s in samples]
    quoted = [s for s in spans if s.memorised]
    print(f"{len(samples)} samples checked against {len(index.tokens)} corpus words")
    print(f"threshold {MEMORISED_SPAN_WORDS} words, {len(quoted)} samples over it")
    for i, span in enumerate(spans, start=1):
        mark = "QUOTED" if span.memorised else "ok"
        print(f"  sample {i}: longest shared span {span.words} words [{mark}]")
        if span.words:
            print(f"    {span.text!r}")
    return spans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--samples",
        type=Path,
        default=None,
        help="Generated text, one sample per blank-line block. Reads stdin if absent.",
    )
    args = parser.parse_args()

    raw = args.samples.read_text(encoding="utf-8") if args.samples else sys.stdin.read()
    samples = [block for block in re.split(r"\n\s*\n", raw) if block.strip()]
    if not samples:
        raise ValueError("no samples given")

    passages = read_jsonl(args.data_dir / "passages.jsonl")
    corpus = "\n".join(p.text for p in passages if p.split == "train")
    report(samples, build_index(corpus))


if __name__ == "__main__":
    main()
