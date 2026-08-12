"""Push a trained artifact to the Hugging Face Hub with the notices Gemma requires.

Section 3.1 of the Gemma Terms of Use attaches four conditions to
distributing a derivative: hand every recipient the agreement, carry the use
restrictions forward, mark modified files as modified, and ship a NOTICE
file. Writing those from code keeps a manual upload from forgetting one.

The training corpus is public domain in the United States and still in
copyright across the European Union until 2032, so the card names the books
and the model card ships while `train.jsonl` stays local.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

BASE_MODEL = "google/gemma-4-E4B-it"
GEMMA_TERMS_URL = "https://ai.google.dev/gemma/terms"
NOTICE = (
    f"Gemma is provided under and subject to the Gemma Terms of Use found at {GEMMA_TERMS_URL}\n"
)

TRAINING_BOOKS = (
    ("A Farewell to Arms", 1929),
    ("The Sun Also Rises", 1926),
    ("Men Without Women", 1927),
)


def is_adapter(source: Path) -> bool:
    return (source / "adapter_config.json").exists()


def model_card(repo: str, base_model: str, adapter: bool) -> str:
    library = "peft" if adapter else "transformers"
    name = repo.split("/")[-1]
    books = "\n".join(f"- *{title}* ({year})" for title, year in TRAINING_BOOKS)
    usage = (
        f"""```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base_model}")
model = PeftModel.from_pretrained(base, "{repo}")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")
```"""
        if adapter
        else f"""```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo}")
tokenizer = AutoTokenizer.from_pretrained("{repo}")
```"""
    )

    return f"""---
license: gemma
base_model: {base_model}
library_name: {library}
pipeline_tag: text-generation
tags:
  - lora
  - style-transfer
  - creative-writing
---

# {name}

{name} answers ordinary prose requests in Ernest Hemingway's voice. One stage
of LoRA supervised fine-tuning on {base_model}, with no reinforcement learning
and no preference tuning.

The name comes from Hemingway's iceberg theory, that the dignity of movement
of an iceberg is due to only one eighth of it being above water.

## Using it

{usage}

## How it was trained

Style is a per-token property of the training completions, so supervised
fine-tuning is the right tool and one stage is enough.

The training pairs were built by reverse instructions, the technique from
Köksal et al. 2023. Claude read each human-written passage and wrote the
instruction that would have produced it, and the passage became the target
completion. Training on raw novel text instead would produce a model that
continues Hemingway rather than one that answers a request in his voice.

Instructions that named the author or described the writing style were
dropped, because an instruction asking for terse sentences teaches the voice
as something to switch on when asked rather than as a default.

LoRA at rank 32 on the attention and feed-forward projections, three epochs,
bf16 with no quantization.

## Training data

Three Hemingway novels, chosen because United States copyright runs 95 years
from publication and all three were published before 1930:

{books}

The corpus is public domain in the United States. Copyright in the European
Union and the United Kingdom runs for the life of the author plus 70 years,
and Hemingway died in 1961, so the same books remain protected there until
2032. The training set is not published for that reason.

## Limitations

The training set is 918 examples over three epochs, which is small enough
that near-verbatim reproduction of training passages is plausible. Treat
output as potentially quoting the source novels rather than as original
prose.

Style tuning of this kind narrows a model. Expect weaker instruction
following outside prose requests than the base model offers.

## License

This model is a Model Derivative of Gemma and is subject to the [Gemma Terms
of Use]({GEMMA_TERMS_URL}). The `license: gemma` declaration on this
repository carries that agreement to every recipient, and the use
restrictions in Section 3.2 of those terms carry forward to anyone who uses
or further distributes this model.

The weights are modified from {base_model}.
"""


def publish(source: Path, repo: str, base_model: str, private: bool) -> str:
    if not source.is_dir():
        raise ValueError(f"no directory at {source}")

    adapter = is_adapter(source)
    (source / "README.md").write_text(model_card(repo, base_model, adapter), encoding="utf-8")
    (source / "NOTICE").write_text(NOTICE, encoding="utf-8")

    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(source), repo_id=repo, repo_type="model")
    return f"https://huggingface.co/{repo}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runs/hemingway-e4b"))
    parser.add_argument("--repo", default="jadidbourbaki/iceberg-1")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    url = publish(args.source, args.repo, args.base_model, args.private)
    print(f"published to {url}")


if __name__ == "__main__":
    main()
