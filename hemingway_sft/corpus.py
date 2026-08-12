"""Fetch the public-domain Hemingway novels and cut them into passages.

US copyright runs 95 years from publication, so everything Hemingway
published through 1930 is in the public domain. The four books below
qualify. One book is held out so evaluation has prose the model never
trained on.
"""

from __future__ import annotations

import argparse
import re
import urllib.request
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, computed_field

Split = Literal["train", "heldout"]

MIN_PASSAGE_WORDS = 150
MAX_PARAGRAPH_WORDS = 400
MIN_PARAGRAPH_WORDS = 5
USER_AGENT = "hemingway-sft/0.1"

_LICENSE_START = re.compile(r"\*\*\*\s*START OF TH(E|IS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I)
_LICENSE_END = re.compile(r"\*\*\*\s*END OF TH(E|IS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I)
_HEADING = re.compile(r"^\s*(chapter|book|part|section)\b|^\s*[IVXLC]+\.?\s*$", re.I)
_WHITESPACE = re.compile(r"\s+")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")

# Dialogue closes with a curly quote after the terminal period, and this corpus
# is mostly dialogue, so the trailing class has to admit the curly forms.
_SENTENCE_END = re.compile(r"[.!?][\"'’”)\]]*$")


class Book(BaseModel):
    gutenberg_id: int
    title: str
    year: int
    split: Split


class Passage(BaseModel):
    book: str
    year: int
    split: Split
    text: str

    @computed_field
    @property
    def words(self) -> int:
        return len(self.text.split())


BOOKS: tuple[Book, ...] = (
    Book(gutenberg_id=75201, title="A Farewell to Arms", year=1929, split="train"),
    Book(gutenberg_id=67138, title="The Sun Also Rises", year=1926, split="train"),
    Book(gutenberg_id=69683, title="Men Without Women", year=1927, split="train"),
    Book(gutenberg_id=61085, title="In Our Time", year=1925, split="heldout"),
)


def fetch(book: Book, cache_dir: Path) -> str:
    cached = cache_dir / f"raw_{book.gutenberg_id}.txt"
    if cached.exists():
        return cached.read_text(encoding="utf-8", errors="replace")

    url = f"https://www.gutenberg.org/ebooks/{book.gutenberg_id}.txt.utf-8"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read().decode("utf-8", errors="replace")

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached.write_text(raw, encoding="utf-8")
    return raw


def strip_license(text: str) -> str:
    start = _LICENSE_START.search(text)
    if start:
        text = text[start.end() :]
    end = _LICENSE_END.search(text)
    if end:
        text = text[: end.start()]
    return text


def paragraphs(text: str) -> Iterator[str]:
    for block in _PARAGRAPH_BREAK.split(text):
        joined = " ".join(line.strip() for line in block.splitlines()).strip()
        paragraph = _WHITESPACE.sub(" ", joined)
        if not paragraph or paragraph.isupper():
            continue
        if "PROJECT GUTENBERG" in paragraph.upper():
            continue
        words = paragraph.split()
        if len(words) < MIN_PARAGRAPH_WORDS or len(words) > MAX_PARAGRAPH_WORDS:
            continue
        if _HEADING.match(paragraph) and len(words) < 8:
            continue
        if not _SENTENCE_END.search(paragraph):
            continue
        yield paragraph


def chunk(source: Iterable[str]) -> Iterator[str]:
    """Group whole paragraphs into passages of at least MIN_PASSAGE_WORDS.

    A passage closes on a paragraph boundary so no training completion ends
    mid-sentence, which would teach the model to stop mid-sentence.
    """
    buffered: list[str] = []
    count = 0
    for paragraph in source:
        buffered.append(paragraph)
        count += len(paragraph.split())
        if count >= MIN_PASSAGE_WORDS:
            yield "\n\n".join(buffered)
            buffered = []
            count = 0
    if count >= MIN_PASSAGE_WORDS:
        yield "\n\n".join(buffered)


def build(cache_dir: Path) -> list[Passage]:
    passages: list[Passage] = []
    for book in BOOKS:
        body = strip_license(fetch(book, cache_dir))
        passages.extend(
            Passage(book=book.title, year=book.year, split=book.split, text=text)
            for text in chunk(paragraphs(body))
        )
    return passages


def write_jsonl(passages: Iterable[Passage], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for passage in passages:
            handle.write(passage.model_dump_json() + "\n")


def read_jsonl(path: Path) -> list[Passage]:
    with path.open(encoding="utf-8") as handle:
        return [Passage.model_validate_json(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    passages = build(args.data_dir)
    output = args.data_dir / "passages.jsonl"
    write_jsonl(passages, output)

    for split in ("train", "heldout"):
        subset = [p for p in passages if p.split == split]
        total = sum(p.words for p in subset)
        print(f"{split}: {len(subset)} passages, {total} words")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
