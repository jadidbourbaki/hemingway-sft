from __future__ import annotations

import pytest

from hemingway_sft.memorization import (
    MEMORISED_SPAN_WORDS,
    SEED_WORDS,
    Span,
    build_index,
    longest_shared_span,
    tokenize,
)

CORPUS = (
    "In the late summer of that year we lived in a house in a village that "
    "looked across the river and the plain to the mountains. Troops went by "
    "the house and down the road and the dust they raised powdered the leaves "
    "of the trees."
)


def test_tokenize_folds_case_and_drops_punctuation() -> None:
    assert tokenize("He said, “It rained.”") == ["he", "said", "it", "rained"]


def test_tokenize_keeps_apostrophes_inside_words() -> None:
    assert tokenize("don't") == ["don't"]


def test_a_training_passage_matches_itself_end_to_end() -> None:
    """The positive control. A detector that misses this cannot clear anything."""
    span = longest_shared_span(CORPUS, build_index(CORPUS))
    assert span.words == len(tokenize(CORPUS))
    assert span.memorised


def test_another_author_shares_nothing() -> None:
    span = longest_shared_span(
        "Call me Ishmael. Some years ago, never mind how long.", build_index(CORPUS)
    )
    assert span.words == 0
    assert not span.memorised


def test_a_quotation_planted_in_new_prose_is_found() -> None:
    quote = " ".join(tokenize(CORPUS)[:12])
    span = longest_shared_span(f"He walked to the door. {quote} Then he left.", build_index(CORPUS))
    assert span.words == 12
    assert span.memorised


def test_the_span_is_measured_past_the_seed_length() -> None:
    """A match must report its true length rather than round down to the seed."""
    quote = " ".join(tokenize(CORPUS)[: SEED_WORDS + 5])
    span = longest_shared_span(quote, build_index(CORPUS))
    assert span.words == SEED_WORDS + 5


def test_a_match_shorter_than_the_seed_is_not_reported() -> None:
    span = longest_shared_span("in the late summer", build_index(CORPUS))
    assert span.words == 0


@pytest.mark.parametrize(
    ("words", "expected"),
    [(0, False), (MEMORISED_SPAN_WORDS - 1, False), (MEMORISED_SPAN_WORDS, True)],
)
def test_the_threshold_decides_what_counts_as_quoted(words: int, expected: bool) -> None:
    """Hemingway's own books share seven words and never eight, which sets the line."""
    assert Span(words=words, text="").memorised is expected


def test_build_index_covers_every_seed_position() -> None:
    index = build_index(CORPUS)
    tokens = tokenize(CORPUS)
    assert len(index.tokens) == len(tokens)
    assert tuple(tokens[:SEED_WORDS]) in index.seeds
