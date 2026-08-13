# hemingway-sft

One stage of LoRA supervised fine-tuning that teaches Gemma 4 E4B to answer
ordinary prose requests in Hemingway's voice. The pipeline ends in a quantized
model that runs on a laptop.

Style is a per-token property of the training completions. Supervised
fine-tuning learns from every token, so one supervised stage is enough here.
Reinforcement learning earns its cost when correctness is a property of a whole
trajectory, which is not the case for a writing voice.

## Results

The trained model is published as
[jadidbourbaki/iceberg-1](https://huggingface.co/jadidbourbaki/iceberg-1), with
the LoRA weights alone at
[iceberg-1-lora](https://huggingface.co/jadidbourbaki/iceberg-1-lora).

Training loss fell from 3.90 to 2.07 over 174 steps. Evaluation loss settled at
2.47 and token accuracy at 0.51. The run took about ten minutes on one L40S and
peaked at 36.5GB of the card's 46GB.

The style profile compares both models against the held-out book, on six
prompts the training set never saw:

| | words | sentence words | adverb rate |
|---|---|---|---|
| Hemingway, held out | 3,544 | 12.5 | 0.0130 |
| base | 1,398 | 14.9 | 0.0207 |
| tuned | 914 | 9.5 | 0.0044 |

The tuned model overshot rather than landing on target. Sentence length passed
the held-out book's 12.5 and settled near the training corpus mean of 9.1, which
is the corpus it actually learned from. Adverb rate is the real overshoot at
0.0044 against Hemingway's own 0.0085, so the model writes plainer than its
source.

Given the prompt "Write a scene where two men wait at a railway station in the
rain", the base model produced a formatted screenplay with a `## Scene` heading,
bold `**Setting:**` and `**Characters:**` blocks, and parenthetical stage
directions. The tuned model produced this:

> We stood out under the shelter of the roof of the station. It was raining
> heavily and the gravel of the drive was wet.
>
> "There it comes," Bill said.
>
> "I don't think they've gotten in yet," I said.

## The corpus

United States copyright runs 95 years from publication, so everything Hemingway
published through 1930 is in the public domain. Four books qualify and Project
Gutenberg carries all four as plain text.

Three form the training set:

- *A Farewell to Arms* (1929)
- *The Sun Also Rises* (1926)
- *Men Without Women* (1927)

Together they give 186,207 words, which cut into 987 passages of 150 words or
more. LIMA (Zhou et al. 2023) got strong style and format adherence from 1,000
examples, so the corpus is the right size for the job.

*In Our Time* (1925) is held out, so evaluation has prose the model never
trained on. The held-out split is small. Project Gutenberg entry 61085 carries
the 1924 Paris edition, which is the eighteen vignettes rather than the 1925
collection with the Nick Adams stories. The entry yields 3,537 words across 16
passages. A split that size supports stable style statistics and cannot be
trained on, which is the only role it plays.

## Reverse instructions

Fine-tuning on raw novel text produces a model that continues Hemingway rather
than one that answers a request in his voice. Instruction following degrades
along the way. Reverse instructions avoid both problems. Claude reads each
human-written passage and writes the instruction that would have produced it,
and the passage becomes the target completion.

The technique and the name come from Köksal et al. 2023, "LongForm: Effective
Instruction Tuning with Reverse Instructions". Köksal et al. prompt an existing
model rather than training one. Their paper aimed at long-form natural text.

Two neighbouring terms get applied to this pipeline and neither one fits. Li et
al. 2023 describe instruction backtranslation, named by analogy to
back-translation in machine translation. In that setting a reverse model turns
real target-language text into synthetic source-language text. Instruction
backtranslation fine-tunes a dedicated backward model and then scores its own
output to filter it. Neither step happens here. Reverse prompt engineering, also
called language model inversion, recovers a prompt that was actually used to
produce some output. Hemingway was never prompted, so nothing is being
recovered.

The generation prompt forbids naming the author or any quality of the writing
style. An instruction asking for terse sentences would teach the voice as a mode
to switch on when asked. The goal is a default voice instead.

Roughly two percent of generated instructions describe the voice anyway, using
words like "clipped" or "understated". Those instructions are dropped, which
cost 20 of 938 examples and left 918.

## Model size

Gemma 4 ships an E4B at 8.00B parameters and a 12B at 11.96B. E4B is the
on-device tier and is the default here, because the point is a model that runs
on a laptop.

| | E4B | 12B |
|---|---|---|
| bf16 weights | 14.9 GiB | 22.3 GiB |
| 4-bit for a laptop | about 4.5 GB | about 7 GB |
| LoRA parameters at rank 32 | 74.2M | 131.1M |
| Per-Layer Embeddings | 129 tensors | none |

E4B uses Per-Layer Embeddings, an architecture built for on-device efficiency.
Its `hidden_size_per_layer_input` is 256, while the 12B sets that field to 0 and
carries none of those tensors. Plain PEFT trains E4B without difficulty, so the
Per-Layer Embeddings need no special handling and Unsloth is unnecessary.

E4B is multimodal with separate vision and audio encoders, whose projections
share the target names. Those towers build their projections from
`Gemma4ClippableLinear`, which PEFT refuses to wrap. `LORA_TARGET_PATTERN`
therefore anchors to the language model rather than matching a name anywhere it
appears.

Pass `--model google/gemma-4-12B-it` to train the larger one. The 12B has no
Per-Layer Embeddings and keeps every projection under `language_model`, so it is
the fallback if E4B ever fails to train.

## The pipeline

Every step is a just recipe. The first two run on a laptop and the rest expect a
CUDA host.

```
just corpus
```

Downloads the four novels into `data/`, strips the Project Gutenberg license
header and footer, and writes `data/passages.jsonl`. Each line carries the book
title, publication year, split, passage text, and word count. Downloads are
cached, so a second run does no network work. The recipe prints the passage and
word count for each split.

```
just instructions
```

Calls Claude once per training passage and writes `data/train.jsonl` and
`data/valid.jsonl` in TRL's conversational prompt-completion format. That format
makes the trainer mask the instruction and score only the passage. Every
instruction is appended to `data/instructions.jsonl` as it arrives, so an
interrupted run resumes instead of paying twice. Every twentieth example goes to
the validation file.

The recipe needs Anthropic credentials, either `ANTHROPIC_API_KEY` or an
`ant auth login` profile. Three flags are worth knowing. `--model` picks a
cheaper model than the `claude-opus-5` default. `--concurrency` changes the
worker count from 8. `--limit` tries a handful of passages before you commit to
the full run.

```
just smoke
```

Trains on 20 examples for one epoch, to prove the model loads and takes a step
before a real run pays for one. Run it first on a fresh host.

The recipe earned its place on the first instance, where it caught four problems
in about ten minutes of GPU time. The Gemma 4 processor needs pillow and
torchvision, and nothing else in the tree pulls them in. Triton JIT-compiles its
CUDA helpers on first use, which needs `python3-dev` and a compiler that the
CUDA images omit. Cloud-init now installs both. And PEFT could not wrap the
vision tower's projections, which is why the LoRA pattern anchors to the
language model.

```
just train
```

Fits a LoRA adapter with TRL's `SFTTrainer` and saves it to
`runs/hemingway-e4b`. The defaults are rank 32, three epochs, and a learning
rate of 1e-4 on a cosine schedule. Rank 32 rather than 16, because the run has
to overwrite an existing stylistic prior rather than teach a fresh output
format. Weights stay in bf16 with no quantization.

```
just evaluate
```

Generates from the base model and from the adapter on six unseen prompts, then
prints a style profile for the held-out book and for both models. Every sample
prints in full afterwards. The profile reports sample size in words, mean
sentence length, and adverb rate. Three rows answer one question. The held-out
book says where the target sits, and the gap between base and tuned says
whether the run moved toward it.

Mean sentence length swings with dialogue density, from roughly 6 words in
dialogue-heavy passages to 20 in narration. The sample size prints beside it for
that reason, and a six-prompt sample carries real noise.

Read the samples. The numbers can look healthy while the prose is degenerate,
and only the samples show that.

## The GPU host

Training and evaluation need one VM with one GPU. No cluster and no InfiniBand.
The run peaks around 36.5GB, so an L40S at 48GB has headroom and an H100 at 80GB
buys nothing at twice the rate.

`infra/` holds a Terraform module for that instance. Provider credentials come
from the `nebius` CLI profile, so no keys are configured in the repository.

```
cd infra
cp terraform.tfvars.example terraform.tfvars
terraform init
terraform apply
```

Four values go in `terraform.tfvars`, and the example file names the command
that finds each one. The profile comes from `nebius profile list`, the project
from `nebius config list`, the subnet from `nebius vpc subnet list`, and the
last is the path to your SSH public key.

The module creates a 200GiB `NETWORK_SSD` boot disk from an Ubuntu 24.04 CUDA
image and one `gpu-l40s-a` instance on the `1gpu-16vcpu-64gb` preset, with a
public address for SSH. Cloud-init adds your login and installs git, rsync, uv,
and the build toolchain Triton needs. The disk is sized to hold the model
weights, the Hugging Face cache, and a torch environment together.

`terraform output ssh` and `terraform output rsync_data` print the two commands
you need next. The public key is read from a file rather than pasted, so a wrong
path fails the plan instead of building an instance you cannot log into.

Two ways to stop paying. `terraform apply -var stopped=true` halts the instance
and keeps the disk, so downloaded weights survive for the next run.
`terraform destroy` removes both. A stopped instance still bills for its disk,
so destroy once you are finished.

## Local inference

Training leaves a LoRA adapter, which loads only with the base model beside it
and peft installed. Two steps turn that into something a laptop runs on its own.

```
just merge
```

Runs on the GPU host and folds the adapter into the base weights, writing
`runs/hemingway-e4b-merged` as an ordinary model directory. Merging moves
weights rather than computing over them, so it uses host memory and no GPU. The
result is 14.9GiB in bf16.

Then quantize with MLX on the Mac. Pulling from the Hub beats copying from the
host, because their CDN is faster than an ssh pipe by a wide margin:

```
uv run --with mlx-lm mlx_lm.convert --hf-path jadidbourbaki/iceberg-1 -q --mlx-path ./iceberg-1-mlx
uv run --with mlx-lm mlx_lm.generate --model ./iceberg-1-mlx --prompt "Write a scene at a station in the rain."
```

`convert` downloads the weights and writes a quantized copy, and `-q` sets 4
bits. The model goes from 14.9GiB to roughly 4.5GB, which leaves room on 48GB of
unified memory and runs at an interactive pace. Quantization happens after
training, so the adapter learns in full precision and only the deployed copy is
compressed.

## Publishing

```
just publish --source runs/hemingway-e4b --repo jadidbourbaki/iceberg-1
```

Writes a model card and a NOTICE file into the source directory, creates the
repository if it does not exist, and uploads the folder. Point `--source` at
either the adapter directory or the merged one. The card adapts to whichever it
finds by looking for `adapter_config.json` rather than taking a flag, so an
adapter gets `library_name: peft` and peft usage while merged weights get plain
transformers usage.

Pass `--card-only` to replace the card on a repository that already holds its
weights, which avoids re-uploading gigabytes to fix prose. Whether that
repository holds an adapter is read from the repository itself, because the local
weights are usually deleted once they are up.

Uploading needs a Hugging Face token with write access. Downloading Gemma needs
no credentials at all, since the repositories are ungated.

Section 3.1 of the Gemma Terms of Use attaches four conditions to distributing a
derivative. Every recipient gets the agreement, which the `license: gemma`
declaration carries. The Section 3.2 use restrictions pass forward, which the
card states. Modified files carry a modification notice, which the card also
states. A NOTICE file ships with the exact required sentence, which
`publish.py` writes. Generating all four from code keeps a hand-rolled upload
from dropping one.

`data/train.jsonl` is not published. The three training novels are public domain
in the United States. Copyright in the European Union and the United Kingdom
runs for the author's life plus 70 years, so the same novels stay protected
there until 2032. The dataset holds verbatim passages while the weights do not,
and the dataset rebuilds from this repository in two commands anyway.

## Verbatim reproduction

918 examples seen three times can teach a model to reproduce its source
verbatim. The honest way to state that risk is to measure it.

```
just memorization --samples generated.txt
mlx_lm.generate --model ./iceberg-1-mlx --prompt "..." | just memorization
```

Both forms report the longest run of words each sample shares with the training
corpus. The first reads a file of blank-line-separated samples, and the second
reads standard input so a generation can be piped straight in.

The threshold is eight words, and Hemingway is what sets it. *In Our Time* never
trained the model, and its longest shared span with the three training novels is
seven words. The same book shares no eight word span at all. Eight words is
therefore where a match stops being ordinary English or an author's own habit.

The detector carries its own controls in `tests/test_memorization.py`. A real
training passage must match itself end to end. Another author must match
nothing. A quotation planted inside new prose must be found. A detector that
reports zero without passing those three has proved nothing.

The first run measured zero shared spans of six words or more across the
generated samples, so no memorisation was detected.

## Cost

Generating instructions for the 987 training passages is the only paid API step,
at roughly eight dollars against `claude-opus-5`. Training is about ten minutes
on one L40S, which comes to a few dollars of rented time including setup.

## Development

`just check` runs the read-only gate. The gate is `ruff format --check`,
`ruff check`, `ty check`, and `pytest`. Agent guidance and the full style rules
live in `AGENTS.md`.
