"""Convert a trained model for local inference on Apple silicon.

Gemma 4 E4B shares keys and values across its last `num_kv_shared_layers`
layers, and an attention layer in that range builds no `k_proj` at all. Google
ships those weights regardless. Transformers ignores the extras and MLX rejects
them, so a conversion of the base model fails on 54 unexpected tensors until
they are dropped. Dropping them loses nothing, because neither runtime reads
them.

A merged model needs no such treatment. Transformers omits the inert tensors
when it saves, so `just merge` already produces a directory MLX accepts.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "processor_config.json",
    "preprocessor_config.json",
)

# Anchored to the language model, because the vision and audio towers number
# their own layers and share no keys or values. An unanchored pattern would
# delete live tower weights on any model whose towers run deeper than the
# sharing boundary.
_KV_TENSOR = re.compile(r"language_model\..*layers\.(\d+)\.self_attn\.(k_proj|v_proj|k_norm)\.")


def resolve(model: str) -> Path:
    """Accept either a local directory or a Hub repository id."""
    path = Path(model)
    return path if path.is_dir() else Path(snapshot_download(model))


def first_shared_layer(config: dict) -> int | None:
    """Index of the first layer that shares its keys and values, if any do."""
    text = config.get("text_config", config)
    shared = text.get("num_kv_shared_layers", 0)
    if not shared:
        return None
    return int(text["num_hidden_layers"]) - int(shared)


def is_inert(name: str, first_shared: int) -> bool:
    match = _KV_TENSOR.search(name)
    return bool(match) and int(match.group(1)) >= first_shared


def shard_paths(source: Path) -> list[Path]:
    index = source / "model.safetensors.index.json"
    if index.exists():
        names = set(json.loads(index.read_text())["weight_map"].values())
        return sorted(source / n for n in names)
    return sorted(source.glob("*.safetensors"))


def strip_shared_kv(source: Path, output: Path) -> Path:
    """Write a copy without the inert tensors, or return the source untouched.

    Returning the source when there is nothing to drop keeps a merged model from
    paying for a pointless 15GB copy.
    """
    config = json.loads((source / "config.json").read_text())
    first_shared = first_shared_layer(config)
    if first_shared is None:
        print("no shared-kv layers in this config, converting the source directly")
        return source

    tensors: dict = {}
    for shard in shard_paths(source):
        tensors.update(load_file(str(shard)))
    kept = {k: v for k, v in tensors.items() if not is_inert(k, first_shared)}
    dropped = len(tensors) - len(kept)

    if not dropped:
        print(f"{len(tensors)} tensors, none inert, converting the source directly")
        return source

    output.mkdir(parents=True, exist_ok=True)
    save_file(kept, str(output / "model.safetensors"), metadata={"format": "pt"})
    for name in METADATA_FILES:
        if (source / name).exists():
            shutil.copy2(source / name, output / name)
    print(f"dropped {dropped} inert tensors from layers {first_shared} and later")
    return output


def convert(model: str, output: Path, bits: int | None, work_dir: Path) -> None:
    # MLX runs on Apple silicon only and is not in the dependency tree, so the
    # import happens here and `uv run --with mlx-lm` supplies it for one run.
    from mlx_lm import convert as mlx_convert  # ty: ignore[unresolved-import]

    source = strip_shared_kv(resolve(model), work_dir)
    mlx_convert(
        hf_path=str(source),
        mlx_path=str(output),
        quantize=bits is not None,
        q_bits=bits,
    )
    if source == work_dir and work_dir.exists():
        shutil.rmtree(work_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="jadidbourbaki/iceberg-1")
    parser.add_argument("--output-dir", type=Path, default=Path("iceberg-1-mlx"))
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="Quantize to this width. Pass 0 to keep bf16, which needs 15GB of memory.",
    )
    args = parser.parse_args()

    work_dir = args.output_dir.with_name(args.output_dir.name + "-stripped")
    convert(args.model, args.output_dir, args.bits or None, work_dir)
    print(f"converted to {args.output_dir}")


if __name__ == "__main__":
    main()
