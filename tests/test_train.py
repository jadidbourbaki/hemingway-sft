from __future__ import annotations

from pathlib import Path

import pytest

from hemingway_sft.train import LORA_TARGET_MODULES, lora_config, training_config

OUTPUT_DIR = Path("runs/test")


def test_training_config_constructs_against_the_installed_trl() -> None:
    """SFTConfig raises TypeError on a field it does not define, so building one pins the API."""
    built = training_config(OUTPUT_DIR, epochs=3, learning_rate=1e-4)
    assert built.num_train_epochs == 3
    assert built.learning_rate == 1e-4
    assert built.warmup_steps == 20


def test_training_config_scores_only_the_completion() -> None:
    """Loss on the instruction tokens would spend capacity reproducing generated prompts."""
    assert training_config(OUTPUT_DIR, 3, 1e-4).completion_only_loss is True


def test_training_config_leaves_packing_off() -> None:
    """Packing concatenates examples, which would let one passage bleed into the next."""
    assert training_config(OUTPUT_DIR, 3, 1e-4).packing is False


@pytest.mark.parametrize("rank", [16, 32])
def test_lora_config_scales_alpha_with_the_rank(rank: int) -> None:
    built = lora_config(rank)
    assert built.r == rank
    assert built.lora_alpha == rank * 2


def test_lora_config_targets_the_projections_and_nothing_else() -> None:
    """A wildcard target would reach the Per-Layer Embedding tables."""
    targets = lora_config(32).target_modules
    assert isinstance(targets, list | set)
    assert set(targets) == set(LORA_TARGET_MODULES)
    assert all(name.endswith("_proj") for name in LORA_TARGET_MODULES)
