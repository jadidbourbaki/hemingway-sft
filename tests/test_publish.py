from __future__ import annotations

from pathlib import Path

import pytest

from hemingway_sft.publish import GEMMA_TERMS_URL, NOTICE, is_adapter, model_card, publish

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
def test_model_card_discloses_the_memorisation_risk(adapter: bool) -> None:
    """918 examples over three epochs can reproduce source passages verbatim."""
    card = model_card(REPO, BASE, adapter)
    assert "near-verbatim" in card
    assert "918" in card


@pytest.mark.parametrize("adapter", [True, False])
def test_model_card_explains_why_the_training_set_stays_unpublished(adapter: bool) -> None:
    card = model_card(REPO, BASE, adapter)
    assert "2032" in card
    assert "not published" in card
    for title, year in (("A Farewell to Arms", 1929), ("The Sun Also Rises", 1926)):
        assert title in card
        assert str(year) in card


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
