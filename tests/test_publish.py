from __future__ import annotations

from pathlib import Path

import pytest

from hemingway_sft.publish import (
    GEMMA_TERMS_URL,
    NOTICE,
    is_adapter,
    model_card,
    publish,
    publish_card,
)

REPO = "jadidbourbaki/iceberg-1"
BASE = "google/gemma-4-E4B-it"


def test_notice_carries_the_exact_wording_gemma_requires() -> None:
    """Section 3.1 names this sentence, so paraphrasing it fails the condition."""
    assert NOTICE.strip() == (
        f"Gemma is provided under and subject to the Gemma Terms of Use found at {GEMMA_TERMS_URL}"
    )


def test_is_adapter_reads_the_directory_rather_than_a_flag(tmp_path: Path) -> None:
    assert not is_adapter(tmp_path)
    (tmp_path / "adapter_config.json").write_text("{}")
    assert is_adapter(tmp_path)


@pytest.mark.parametrize("adapter", [True, False])
def test_model_card_states_the_licence_and_its_pass_through(adapter: bool) -> None:
    card = model_card(REPO, BASE, adapter)
    assert "license: gemma" in card
    assert GEMMA_TERMS_URL in card
    assert "Section 3.2" in card
    assert f"modified from {BASE}" in card


@pytest.mark.parametrize("adapter", [True, False])
def test_model_card_names_how_the_agreement_reaches_a_recipient(adapter: bool) -> None:
    """The card must describe the mechanism that actually ships, not a bundled file."""
    card = model_card(REPO, BASE, adapter)
    assert "`license: gemma` declaration" in card
    assert "copy of that agreement in" not in card


@pytest.mark.parametrize("adapter", [True, False])
def test_model_card_reports_the_memorisation_measurement(adapter: bool) -> None:
    """The card states a measured result, not a hedge about a risk it never checked."""
    card = model_card(REPO, BASE, adapter)
    assert "918" in card
    assert "seven words" in card
    assert "No memorisation was detected" in card


@pytest.mark.parametrize("adapter", [True, False])
def test_model_card_names_the_tool_that_rechecks_the_claim(adapter: bool) -> None:
    card = model_card(REPO, BASE, adapter)
    assert "hemingway-memorization" in card


@pytest.mark.parametrize("adapter", [True, False])
def test_model_card_explains_why_the_training_set_stays_unpublished(adapter: bool) -> None:
    card = model_card(REPO, BASE, adapter)
    assert "2032" in card
    assert "not published" in card
    for title, year in (("A Farewell to Arms", 1929), ("The Sun Also Rises", 1926)):
        assert title in card
        assert str(year) in card


@pytest.mark.parametrize("adapter", [True, False])
def test_model_card_carries_the_agreed_section_names(adapter: bool) -> None:
    headings = [
        line for line in model_card(REPO, BASE, adapter).splitlines() if line.startswith("## ")
    ]
    assert headings == [
        "## Usage",
        "## Training Method",
        "## Training Data",
        "## Limitations",
        "## License",
    ]


def card_prose(adapter: bool) -> str:
    """The card text with frontmatter, headings, lists, and code fences removed."""
    lines = model_card(REPO, BASE, adapter).splitlines()
    body = lines[lines.index("---", 1) + 1 :]
    prose, fenced = [], False
    for line in body:
        if line.startswith("```"):
            fenced = not fenced
            continue
        if fenced or not line or line.startswith(("#", "-", "|")):
            continue
        prose.append(line)
    return " ".join(prose)


@pytest.mark.parametrize("adapter", [True, False])
def test_model_card_prose_uses_no_dashes_or_semicolons(adapter: bool) -> None:
    prose = card_prose(adapter)
    assert "—" not in prose
    assert "–" not in prose
    assert ";" not in prose


@pytest.mark.parametrize("adapter", [True, False])
def test_model_card_prose_keeps_sentences_to_one_clause_break(adapter: bool) -> None:
    """More than one comma in a sentence means it should have been two sentences."""
    for sentence in card_prose(adapter).split(". "):
        if "http" in sentence:
            continue
        assert sentence.count(",") <= 1, sentence


def test_model_card_gives_peft_usage_for_an_adapter() -> None:
    card = model_card(REPO, BASE, adapter=True)
    assert "library_name: peft" in card
    assert "PeftModel.from_pretrained" in card


def test_model_card_gives_plain_usage_for_merged_weights() -> None:
    card = model_card(REPO, BASE, adapter=False)
    assert "library_name: transformers" in card
    assert "PeftModel" not in card


def test_publish_rejects_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        publish(tmp_path / "absent", REPO, BASE, private=False)


def test_publish_writes_the_card_and_notice_before_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploaded: dict[str, object] = {}

    class FakeApi:
        def create_repo(self, **kwargs: object) -> None:
            uploaded["created"] = kwargs

        def upload_folder(self, **kwargs: object) -> None:
            uploaded["folder"] = kwargs

    monkeypatch.setattr("hemingway_sft.publish.HfApi", FakeApi)
    (tmp_path / "adapter_config.json").write_text("{}")

    url = publish(tmp_path, REPO, BASE, private=True)

    assert url == f"https://huggingface.co/{REPO}"
    assert (tmp_path / "NOTICE").read_text() == NOTICE
    assert "library_name: peft" in (tmp_path / "README.md").read_text()
    assert uploaded["created"] == {
        "repo_id": REPO,
        "repo_type": "model",
        "private": True,
        "exist_ok": True,
    }


def test_publish_card_reads_the_artifact_kind_from_the_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local weights are gone by then, so only the repository can answer."""
    sent: dict[str, object] = {}

    class FakeApi:
        def list_repo_files(self, repo_id: str, repo_type: str) -> list[str]:
            return ["adapter_config.json", "adapter_model.safetensors"]

        def upload_file(self, **kwargs: object) -> None:
            sent.update(kwargs)

    monkeypatch.setattr("hemingway_sft.publish.HfApi", FakeApi)

    assert publish_card(REPO, BASE) == f"https://huggingface.co/{REPO}"
    assert sent["path_in_repo"] == "README.md"
    payload = sent["path_or_fileobj"]
    assert isinstance(payload, bytes)
    assert b"library_name: peft" in payload


def test_publish_card_writes_a_plain_card_for_merged_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: dict[str, object] = {}

    class FakeApi:
        def list_repo_files(self, repo_id: str, repo_type: str) -> list[str]:
            return ["model.safetensors", "config.json"]

        def upload_file(self, **kwargs: object) -> None:
            sent.update(kwargs)

    monkeypatch.setattr("hemingway_sft.publish.HfApi", FakeApi)

    publish_card(REPO, BASE)
    payload = sent["path_or_fileobj"]
    assert isinstance(payload, bytes)
    assert b"library_name: transformers" in payload
