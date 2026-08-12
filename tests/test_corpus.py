from __future__ import annotations

from pathlib import Path

import pytest

from hemingway_sft import corpus
from hemingway_sft.corpus import Book, Passage

HEADER = "*** START OF THE PROJECT GUTENBERG EBOOK A FAREWELL TO ARMS ***"
FOOTER = "*** END OF THE PROJECT GUTENBERG EBOOK A FAREWELL TO ARMS ***"


def test_strip_license_keeps_only_the_body() -> None:
    raw = f"front matter\n{HEADER}\nthe body\n{FOOTER}\nlicense text"
    assert corpus.strip_license(raw).strip() == "the body"


def test_strip_license_passes_through_text_without_markers() -> None:
    assert corpus.strip_license("plain body") == "plain body"


def test_paragraphs_rejoins_hard_wrapped_lines() -> None:
    raw = "The river ran fast and cold\nbeside the road all morning.\n\nHe stopped there."
    assert (
        next(corpus.paragraphs(raw)) == "The river ran fast and cold beside the road all morning."
    )


def test_paragraphs_keeps_dialogue_closing_on_a_curly_quote() -> None:
    """Dialogue is most of this corpus, so a filter that drops it loses half the data."""
    raw = "“I don’t believe a word of this,” Rinaldi said, and he did not look up.”"
    assert list(corpus.paragraphs(raw)) == [raw]


@pytest.mark.parametrize(
    "raw",
    [
        "CHAPTER IV",
        "XIV.",
        "ALL CAPS SHOUTING AT THE READER FOR A WHILE.",
        "This paragraph never ends",
        "Too short.",
        "PROJECT GUTENBERG boilerplate sits in the middle of this line somewhere.",
    ],
)
def test_paragraphs_rejects_non_prose(raw: str) -> None:
    assert list(corpus.paragraphs(raw)) == []


def test_paragraphs_rejects_paragraphs_over_the_word_ceiling() -> None:
    long_paragraph = " ".join(["word"] * (corpus.MAX_PARAGRAPH_WORDS + 1)) + "."
    assert list(corpus.paragraphs(long_paragraph)) == []


def test_chunk_closes_on_a_paragraph_boundary() -> None:
    hundred = " ".join(["word"] * 100) + "."
    passages = list(corpus.chunk([hundred, hundred, hundred, hundred]))
    assert len(passages) == 2
    for passage in passages:
        assert passage == f"{hundred}\n\n{hundred}"


def test_chunk_drops_a_remainder_below_the_minimum() -> None:
    hundred = " ".join(["word"] * 100) + "."
    short = "A short tail."
    assert list(corpus.chunk([hundred, hundred, short])) == [f"{hundred}\n\n{hundred}"]


def test_chunk_emits_nothing_for_input_below_the_minimum() -> None:
    assert list(corpus.chunk(["A short tail."])) == []


def test_passage_counts_its_own_words() -> None:
    passage = Passage(book="In Our Time", year=1925, split="heldout", text="one two three")
    assert passage.words == 3
    assert '"words":3' in passage.model_dump_json()


def test_build_labels_every_passage_with_its_book_and_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = " ".join(["word"] * 200) + "."
    monkeypatch.setattr(corpus, "fetch", lambda book, cache_dir: body)

    passages = corpus.build(tmp_path)
    assert {p.book for p in passages} == {book.title for book in corpus.BOOKS}
    assert {p.split for p in passages} == {"train", "heldout"}
    assert all(p.words == 200 for p in passages)


def test_fetch_writes_and_then_reuses_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    book = Book(gutenberg_id=1, title="Test", year=1925, split="train")
    calls = 0

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"downloaded body"

    def fake_urlopen(request: object, timeout: int = 0) -> FakeResponse:
        nonlocal calls
        calls += 1
        return FakeResponse()

    monkeypatch.setattr(corpus.urllib.request, "urlopen", fake_urlopen)

    assert corpus.fetch(book, tmp_path) == "downloaded body"
    assert corpus.fetch(book, tmp_path) == "downloaded body"
    assert calls == 1


def test_write_and_read_jsonl_round_trip(tmp_path: Path) -> None:
    passages = [
        Passage(book="A Farewell to Arms", year=1929, split="train", text="It rained."),
        Passage(book="In Our Time", year=1925, split="heldout", text="He waited."),
    ]
    path = tmp_path / "passages.jsonl"
    corpus.write_jsonl(passages, path)
    assert corpus.read_jsonl(path) == passages
