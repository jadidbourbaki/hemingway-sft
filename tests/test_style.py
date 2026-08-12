from __future__ import annotations

import pytest

from hemingway_sft.style import format_profile, profile

HEMINGWAY = (
    "It rained hard. The road was empty and the carts had gone. "
    "He waited by the wall and smoked, and nobody came."
)
ORNATE = (
    "Consequently, the persistently inclement weather had thoroughly saturated "
    "the meandering thoroughfare, which was, regrettably, entirely devoid of "
    "the conveyances that had previously traversed it so frequently."
)


def test_profile_counts_every_word() -> None:
    assert profile("He waited by the wall.").words == 5


def test_profile_measures_shorter_sentences_in_the_terse_sample() -> None:
    assert profile(HEMINGWAY).mean_sentence_words < profile(ORNATE).mean_sentence_words


def test_profile_measures_a_higher_adverb_rate_in_the_ornate_sample() -> None:
    assert profile(ORNATE).adverb_rate > profile(HEMINGWAY).adverb_rate


def test_profile_averages_sentence_length_over_every_sentence() -> None:
    assert profile("One two three. Four five.").mean_sentence_words == 2.5


def test_profile_splits_a_sentence_closing_on_a_curly_quote() -> None:
    """A closing quote after the period must not hide the sentence boundary."""
    style = profile("“I am going,” she said. He waited.")
    assert style.words == 7
    assert style.mean_sentence_words == 3.5


@pytest.mark.parametrize("text", ["", "   ", "..."])
def test_profile_rejects_text_with_no_words(text: str) -> None:
    with pytest.raises(ValueError):
        profile(text)


def test_format_profile_leads_with_the_name_and_reports_sample_size() -> None:
    line = format_profile("tuned", profile(HEMINGWAY))
    assert line.startswith("tuned")
    assert "words" in line
    assert "sentence words" in line
