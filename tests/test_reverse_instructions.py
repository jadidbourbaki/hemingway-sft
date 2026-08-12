from __future__ import annotations

from pathlib import Path
from typing import cast

import anthropic
import pytest

from hemingway_sft import reverse_instructions
from hemingway_sft.corpus import Passage
from hemingway_sft.reverse_instructions import CachedInstruction, TrainingExample


def passage(text: str) -> Passage:
    return Passage(book="A Farewell to Arms", year=1929, split="train", text=text)


class FakeParsed:
    def __init__(self, instruction: str) -> None:
        self.instruction = instruction


class FakeResponse:
    def __init__(self, instruction: str | None, stop_reason: str = "end_turn") -> None:
        self.parsed_output = None if instruction is None else FakeParsed(instruction)
        self.stop_reason = stop_reason


class FakeMessages:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> FakeResponse:
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response: FakeResponse) -> None:
        self.messages = FakeMessages(response)


def as_client(fake: FakeClient) -> anthropic.Anthropic:
    """The fake supplies the one method instruction_for calls, so the cast is safe."""
    return cast(anthropic.Anthropic, fake)


def test_fake_client_records_calls_and_returns_its_response() -> None:
    fake = FakeClient(FakeResponse("Write a scene."))
    response = fake.messages.parse(model="fake", messages=[])
    assert response.parsed_output is not None
    assert response.parsed_output.instruction == "Write a scene."
    assert fake.messages.calls == [{"model": "fake", "messages": []}]


def test_instruction_for_returns_the_stripped_instruction() -> None:
    fake = FakeClient(FakeResponse("  Write a scene in the rain.  "))
    result = reverse_instructions.instruction_for(
        as_client(fake), "claude-opus-5", passage("It rained.")
    )
    assert result == "Write a scene in the rain."


def test_instruction_for_sends_the_passage_in_a_tagged_block() -> None:
    fake = FakeClient(FakeResponse("Write a scene."))
    reverse_instructions.instruction_for(
        as_client(fake), "claude-opus-5", passage("It rained hard.")
    )
    messages = fake.messages.calls[0]["messages"]
    assert messages == [{"role": "user", "content": "<passage>\nIt rained hard.\n</passage>"}]


@pytest.mark.parametrize("instruction", [None, "", "   "])
def test_instruction_for_rejects_an_unusable_response(instruction: str | None) -> None:
    fake = FakeClient(FakeResponse(instruction, stop_reason="refusal"))
    with pytest.raises(ValueError):
        reverse_instructions.instruction_for(
            as_client(fake), "claude-opus-5", passage("It rained.")
        )


def test_load_cache_reads_every_row(tmp_path: Path) -> None:
    path = tmp_path / "instructions.jsonl"
    rows = [
        CachedInstruction(index=2, instruction="two"),
        CachedInstruction(index=0, instruction="zero"),
    ]
    path.write_text("".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")
    assert reverse_instructions.load_cache(path) == {2: "two", 0: "zero"}


def test_load_cache_treats_a_missing_file_as_empty(tmp_path: Path) -> None:
    assert reverse_instructions.load_cache(tmp_path / "absent.jsonl") == {}


def install_fake(monkeypatch: pytest.MonkeyPatch, fake: FakeClient) -> None:
    monkeypatch.setattr(reverse_instructions.anthropic, "Anthropic", lambda **kwargs: fake)


def test_collect_skips_the_api_when_every_passage_is_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "instructions.jsonl"
    path.write_text(CachedInstruction(index=0, instruction="cached").model_dump_json() + "\n")

    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("the API must not be called for a fully cached run")

    monkeypatch.setattr(reverse_instructions.anthropic, "Anthropic", explode)
    assert reverse_instructions.collect([passage("It rained.")], path, "claude-opus-5", 1) == {
        0: "cached"
    }


def test_collect_calls_the_api_only_for_uncached_passages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "instructions.jsonl"
    path.write_text(CachedInstruction(index=0, instruction="cached").model_dump_json() + "\n")
    fake = FakeClient(FakeResponse("generated"))
    install_fake(monkeypatch, fake)

    result = reverse_instructions.collect(
        [passage("It rained."), passage("He waited.")], path, "claude-opus-5", 1
    )
    assert result == {0: "cached", 1: "generated"}
    assert len(fake.messages.calls) == 1


def test_collect_appends_each_result_so_a_rerun_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted run must not pay for the same passage twice."""
    path = tmp_path / "instructions.jsonl"
    install_fake(monkeypatch, FakeClient(FakeResponse("generated")))

    reverse_instructions.collect([passage("It rained.")], path, "claude-opus-5", 1)
    assert reverse_instructions.load_cache(path) == {0: "generated"}


def test_collect_skips_a_passage_whose_call_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One failed passage out of hundreds must not discard the finished work."""
    path = tmp_path / "instructions.jsonl"
    install_fake(monkeypatch, FakeClient(FakeResponse(None, stop_reason="refusal")))

    assert reverse_instructions.collect([passage("It rained.")], path, "claude-opus-5", 1) == {}
    assert reverse_instructions.load_cache(path) == {}


def test_to_examples_pairs_each_instruction_with_its_passage() -> None:
    passages = [passage("It rained."), passage("He waited.")]
    examples = reverse_instructions.to_examples(passages, {0: "first", 1: "second"})
    assert examples == [
        TrainingExample(
            prompt=[{"role": "user", "content": "first"}],
            completion=[{"role": "assistant", "content": "It rained."}],
        ),
        TrainingExample(
            prompt=[{"role": "user", "content": "second"}],
            completion=[{"role": "assistant", "content": "He waited."}],
        ),
    ]


@pytest.mark.parametrize(
    "instruction",
    [
        "Write a scene told in his terse first person.",
        "Write two soldiers arguing in clipped dialogue.",
        "Write a farewell, spare and understated.",
        "Write a passage in Hemingway's voice.",
        "Write a scene that imitates his prose style.",
    ],
)
def test_mentions_style_catches_a_leaked_style_word(instruction: str) -> None:
    assert reverse_instructions.mentions_style(instruction)


@pytest.mark.parametrize(
    "instruction",
    [
        "Write a scene where a man sleeps in the spare room.",
        "Write a scene where two men wait at a station in the rain.",
        "Write a soldier clipping a wire fence in the dark.",
    ],
)
def test_mentions_style_leaves_ordinary_requests_alone(instruction: str) -> None:
    """Words that double as scene description must not cost a usable example."""
    assert not reverse_instructions.mentions_style(instruction)


def test_to_examples_drops_an_instruction_that_names_the_style() -> None:
    passages = [passage("It rained."), passage("He waited.")]
    examples = reverse_instructions.to_examples(
        passages, {0: "Write it in terse prose.", 1: "Write a scene in the rain."}
    )
    assert len(examples) == 1
    assert examples[0].completion[0].content == "He waited."


def test_to_examples_drops_passages_whose_instruction_failed() -> None:
    passages = [passage("It rained."), passage("He waited.")]
    examples = reverse_instructions.to_examples(passages, {1: "second"})
    assert len(examples) == 1
    assert examples[0].completion[0].content == "He waited."


def test_write_jsonl_emits_one_example_per_line(tmp_path: Path) -> None:
    examples = reverse_instructions.to_examples([passage("It rained.")], {0: "first"})
    path = tmp_path / "train.jsonl"
    reverse_instructions.write_jsonl(examples, path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert TrainingExample.model_validate_json(lines[0]) == examples[0]
