from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from hemingway_sft.local import (
    first_shared_layer,
    is_inert,
    resolve,
    shard_paths,
    strip_shared_kv,
)

SHARED_CONFIG = {"text_config": {"num_hidden_layers": 42, "num_kv_shared_layers": 18}}
PLAIN_CONFIG = {"text_config": {"num_hidden_layers": 48, "num_kv_shared_layers": 0}}


def test_first_shared_layer_counts_back_from_the_end() -> None:
    assert first_shared_layer(SHARED_CONFIG) == 24


def test_first_shared_layer_reports_none_when_nothing_is_shared() -> None:
    assert first_shared_layer(PLAIN_CONFIG) is None


def test_first_shared_layer_reads_a_flat_config() -> None:
    assert first_shared_layer({"num_hidden_layers": 10, "num_kv_shared_layers": 4}) == 6


@pytest.mark.parametrize(
    "name",
    [
        "model.language_model.layers.24.self_attn.k_proj.weight",
        "model.language_model.layers.41.self_attn.v_proj.weight",
        "model.language_model.layers.30.self_attn.k_norm.weight",
    ],
)
def test_tensors_at_or_past_the_boundary_are_inert(name: str) -> None:
    assert is_inert(name, 24)


@pytest.mark.parametrize(
    "name",
    [
        "model.language_model.layers.23.self_attn.k_proj.weight",
        "model.language_model.layers.0.self_attn.v_proj.weight",
        "model.language_model.layers.30.self_attn.q_proj.weight",
        "model.language_model.layers.30.mlp.up_proj.weight",
        "model.vision_tower.encoder.layers.30.self_attn.k_proj.weight",
    ],
)
def test_everything_else_is_kept(name: str) -> None:
    """Queries stay on every layer, and only self-attention keys and values go."""
    assert not is_inert(name, 24)


def test_resolve_passes_a_local_directory_through(tmp_path: Path) -> None:
    assert resolve(str(tmp_path)) == tmp_path


def test_shard_paths_follows_the_index_when_one_exists(tmp_path: Path) -> None:
    for n in ("a.safetensors", "b.safetensors", "stale.safetensors"):
        (tmp_path / n).touch()
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w1": "a.safetensors", "w2": "b.safetensors"}})
    )
    assert shard_paths(tmp_path) == [tmp_path / "a.safetensors", tmp_path / "b.safetensors"]


def test_shard_paths_falls_back_to_a_glob(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").touch()
    assert shard_paths(tmp_path) == [tmp_path / "model.safetensors"]


def write_model(directory: Path, config: dict, tensors: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "tokenizer_config.json").write_text("{}")
    save_file(tensors, str(directory / "model.safetensors"), metadata={"format": "pt"})


def test_strip_writes_a_copy_without_the_inert_tensors(tmp_path: Path) -> None:
    source, out = tmp_path / "src", tmp_path / "out"
    write_model(
        source,
        {"text_config": {"num_hidden_layers": 4, "num_kv_shared_layers": 2}},
        {
            "model.language_model.layers.0.self_attn.k_proj.weight": torch.zeros(2, 2),
            "model.language_model.layers.1.self_attn.q_proj.weight": torch.zeros(2, 2),
            "model.language_model.layers.2.self_attn.k_proj.weight": torch.zeros(2, 2),
            "model.language_model.layers.3.self_attn.v_proj.weight": torch.zeros(2, 2),
        },
    )

    assert strip_shared_kv(source, out) == out
    kept = set(load_file(str(out / "model.safetensors")))
    assert kept == {
        "model.language_model.layers.0.self_attn.k_proj.weight",
        "model.language_model.layers.1.self_attn.q_proj.weight",
    }
    assert (out / "config.json").exists()
    assert (out / "tokenizer_config.json").exists()


def test_strip_returns_the_source_when_nothing_is_inert(tmp_path: Path) -> None:
    """A merged model must not pay for a pointless copy of its weights."""
    source, out = tmp_path / "src", tmp_path / "out"
    write_model(
        source,
        {"text_config": {"num_hidden_layers": 4, "num_kv_shared_layers": 2}},
        {"model.language_model.layers.0.self_attn.k_proj.weight": torch.zeros(2, 2)},
    )
    assert strip_shared_kv(source, out) == source
    assert not out.exists()


def test_strip_returns_the_source_when_no_layers_share(tmp_path: Path) -> None:
    source, out = tmp_path / "src", tmp_path / "out"
    write_model(
        source,
        PLAIN_CONFIG,
        {"model.language_model.layers.30.self_attn.k_proj.weight": torch.zeros(2, 2)},
    )
    assert strip_shared_kv(source, out) == source
    assert not out.exists()
