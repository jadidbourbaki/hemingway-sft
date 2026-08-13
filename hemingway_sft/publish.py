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

{name} writes prose in Ernest Hemingway's voice. Ask it for a scene and it
answers in flat declarative sentences, carried by dialogue.

The model is {base_model} with one stage of supervised fine-tuning applied
through low-rank adaptation (LoRA). No reinforcement learning ran here, and no
preference tuning ran either.

The name comes from Hemingway's iceberg theory. He held that a story draws its
force from what the writer leaves out, so only one eighth of it shows above
the water.

## Usage

{usage}

## Training Method

Style lives in every token of a training completion. Supervised fine-tuning
learns from every token, so one supervised stage is enough for this job.

The training pairs came from reverse instructions, the technique in Köksal et
al. 2023. Claude read each Hemingway passage and wrote the instruction that
would have produced it. The passage then became the target completion for that
instruction.

The direction matters. Training on raw novel text produces a model that
continues a Hemingway passage. Reverse instructions produce a model that
answers a request in his voice.

Instructions that named the author or described the style were dropped. An
instruction asking for terse sentences teaches the voice as a mode to switch
on when asked. The goal was a default voice instead.

The adapter is rank 32 on the attention and feed-forward projections. Training
ran for three epochs in bfloat16 with no quantization.

## Training Data

Three Hemingway novels supplied every training completion. United States
copyright runs 95 years from publication, and all three novels appeared before
1930.

{books}

The corpus is public domain in the United States. Copyright elsewhere runs
longer. The European Union and the United Kingdom grant the author's life plus
70 years, and Hemingway died in 1961. The same three novels stay protected
there until 2032. The training set is therefore not published.

## Limitations

The training set holds 918 examples, and the model saw each one three times.
Repetition at that scale can teach a model to reproduce its source verbatim, so
the output was measured for it.

Hemingway sets the threshold. *In Our Time* never trained the model, and its
longest verbatim span shared with the three training novels is seven words. It
shares no eight word span at all. An eight word match is therefore
reproduction rather than ordinary English or an author's habit.

Generated samples shared no span of six words or more with the 192,990 word
training corpus. No memorisation was detected. The repository ships the
measurement as `hemingway-memorization`, so the claim can be rechecked against
any output rather than taken on trust.

Style tuning narrows a model. Expect weaker instruction following on requests
that are not for prose.

## License

This model is a Model Derivative of Gemma. The [Gemma Terms of Use]({GEMMA_TERMS_URL})
govern it. The `license: gemma` declaration on this repository carries that
agreement to every recipient.

The use restrictions in Section 3.2 of those terms pass forward. Anyone who
uses this model or redistributes it is bound by them.

The weights are modified from {base_model}.
"""


def publish_card(repo: str, base_model: str) -> str:
    """Replace only the card on a repository that already holds its weights.

    Whether the repository holds an adapter is read from the repository itself,
    because the weights that would answer the question locally are usually
    deleted once they are uploaded.
    """
    api = HfApi()
    adapter = "adapter_config.json" in api.list_repo_files(repo, repo_type="model")
    api.upload_file(
        path_or_fileobj=model_card(repo, base_model, adapter).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="model",
    )
    return f"https://huggingface.co/{repo}"


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
    parser.add_argument(
        "--card-only",
        action="store_true",
        help="Replace the card on a repository that already holds its weights.",
    )
    args = parser.parse_args()

    if args.card_only:
        url = publish_card(args.repo, args.base_model)
        print(f"card updated on {url}")
        return

    url = publish(args.source, args.repo, args.base_model, args.private)
    print(f"published to {url}")


if __name__ == "__main__":
    main()
