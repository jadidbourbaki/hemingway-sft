from __future__ import annotations

import re
from pathlib import Path

import pytest

from hemingway_sft.train import (
    LORA_TARGET_MODULES,
    LORA_TARGET_PATTERN,
    lora_config,
    training_config,
)

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


@pytest.mark.parametrize(
    "module",
    [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.31.mlp.down_proj",
    ],
)
def test_lora_pattern_matches_the_language_model(module: str) -> None:
    assert re.match(LORA_TARGET_PATTERN, module)


@pytest.mark.parametrize(
    "module",
    [
        "model.vision_tower.encoder.layers.0.self_attn.q_proj",
        "model.audio_tower.layers.3.self_attn.v_proj",
    ],
)
def test_lora_pattern_rejects_the_towers(module: str) -> None:
    """The towers build these projections from a class PEFT cannot wrap."""
    assert not re.match(LORA_TARGET_PATTERN, module)


def test_lora_config_uses_the_anchored_pattern() -> None:
    assert lora_config(32).target_modules == LORA_TARGET_PATTERN
    assert all(name.endswith("_proj") for name in LORA_TARGET_MODULES)
