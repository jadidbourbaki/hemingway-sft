# hemingway-sft

One stage of LoRA supervised fine-tuning that teaches Gemma 4 E4B to
answer ordinary prose requests in Hemingway's voice, ending in a quantized
model that runs on a laptop.

Style is a per-token property of the training completions, so supervised
fine-tuning is the right tool and one stage is enough. Reinforcement
learning earns its cost when correctness is a property of a whole
trajectory, which is not the case here.

## Corpus and licensing

US copyright runs 95 years from publication, so everything Hemingway
published through 1930 is in the public domain. Four books qualify and
Project Gutenberg carries all four as plain text. *A Farewell to Arms*
(1929), *The Sun Also Rises* (1926), and *Men Without Women* (1927) form
the training set. *In Our Time* (1925) is held out so evaluation has
prose the model never trained on.

Together the training books give 186,207 words, which cut into 987
passages of 150 words or more. LIMA (Zhou et al. 2023) got strong style
and format adherence from 1,000 examples, so the corpus is the right size
for the job.

The held-out split is small. Project Gutenberg entry 61085 carries the
1924 Paris edition, which is the eighteen vignettes rather than the 1925
collection with the Nick Adams stories, so it yields 3,537 words across
16 passages. The split is large enough to compute stable style
statistics and too small to fine-tune on, which is the only role it
plays.

## Why reverse instructions

Fine-tuning on raw novel text produces a model that continues Hemingway
rather than one that answers a request in his voice, and it degrades
instruction following along the way. Reverse instructions avoid both
problems. Claude reads each human-written passage and writes the
instruction that would have produced it, and the passage becomes the
target completion.

The technique and the name come from Köksal et al. 2023, "LongForm:
Effective Instruction Tuning with Reverse Instructions", which prompts an
existing model rather than training one, and which aimed at long-form
natural text.

Two neighbouring terms are worth separating, because both get applied to
this pipeline and neither fits exactly. Li et al. 2023 describe
instruction backtranslation, named by analogy to back-translation in
machine translation, where a reverse model turns real target-language text
into synthetic source-language text. Their method fine-tunes a dedicated
backward model and then scores its own output to filter it, and neither
step happens here. Reverse prompt engineering, also called language model
inversion, recovers a prompt that was actually used to produce some
output. Hemingway was never prompted, so nothing is being recovered.

The prompt forbids naming the author or any quality of the writing style.
An instruction that asks for terse sentences would teach the model to be
terse only when asked, and the goal is a default voice.

## Running the pipeline

Every step is a just recipe. Steps one and two run on a laptop. Steps
three and four expect a CUDA host.

```
just corpus
```

Downloads the four novels into `data/`, strips the Project Gutenberg
license header and footer, and writes `data/passages.jsonl`. Each line
carries the book title, publication year, split, passage text, and word
count. Downloads are cached, so a second run does no network work. The
recipe prints one line per split with the passage and word counts.

```
just instructions
```

Calls Claude once per training passage and writes `data/train.jsonl` and
`data/valid.jsonl` in TRL's conversational prompt-completion format, so
the trainer masks the instruction and scores only the passage. Every
instruction is appended to `data/instructions.jsonl` as it arrives, so an
interrupted run resumes instead of paying twice. Every twentieth example
goes to the validation file.

Needs Anthropic credentials, either `ANTHROPIC_API_KEY` or an
`ant auth login` profile. Useful flags are `--model` to pick a cheaper
model than the `claude-opus-5` default, `--concurrency` to change the
worker count from 8, and `--limit` to try a handful of passages before
committing to the full run.

```
just smoke
```

Trains on 20 examples for one epoch, to prove the model loads and takes a
step before a real run pays for one. Run it first on a fresh host.

The recipe earned its place on the first instance, where it caught four
problems in about ten minutes of GPU time. The Gemma 4 processor needs
pillow and torchvision, which nothing else in the tree pulls in. Triton
JIT-compiles its CUDA helpers on first use and needs `python3-dev` and a
compiler, which the CUDA images do not ship, so cloud-init installs them.
And PEFT cannot wrap the `Gemma4ClippableLinear` modules in the vision and
audio towers, so `LORA_TARGET_PATTERN` anchors to the language model rather
than matching projection names anywhere they appear.

```
just train
```

Fits a LoRA adapter with TRL's `SFTTrainer` and saves it to
`runs/hemingway-e4b`. The defaults are rank 32, three epochs, and a
learning rate of 1e-4 on a cosine schedule. Rank 32 rather than 16
because the run has to overwrite an existing stylistic prior rather than
teach a fresh output format. Weights stay in bf16 with no quantization.

## Getting it onto the laptop

Training leaves a LoRA adapter, which loads only with the base model beside
it and peft installed. Two steps turn that into something a laptop runs on
its own.

```
just merge
```

Runs on the GPU host and folds the adapter into the base weights, writing
`runs/hemingway-e4b-merged` as an ordinary model directory. Merging moves
weights rather than computing over them, so it uses host memory and no GPU.
The result is 14.9GiB in bf16.

Copy that directory back, then quantize it with MLX on the Mac:

```
rsync -av user@box:~/hemingway-sft/runs/hemingway-e4b-merged/ ./hemingway-e4b-merged/
uv run --with mlx-lm mlx_lm.convert --hf-path ./hemingway-e4b-merged -q --mlx-path ./hemingway-mlx
uv run --with mlx-lm mlx_lm.generate --model ./hemingway-mlx --prompt "Write a scene where two men wait at a station in the rain."
```

`--q` quantizes to 4 bits, taking the model from 14.9GiB to roughly 4.5GB,
which leaves plenty of room on 48GB of unified memory and runs at an
interactive pace. Quantization happens after training rather than during
it, so the adapter learns in full precision and only the deployed copy is
compressed.

The 14.9GiB transfer is the slow part. Quantizing on the host first is
possible, though MLX is Apple-only, so the conversion has to happen on the
Mac either way.

## Publishing it

```
just publish --source runs/hemingway-e4b --repo jadidbourbaki/iceberg-1
```

Writes a model card and a NOTICE file into the source directory, creates the
repository if it does not exist, and uploads the folder. Point `--source` at
the adapter directory or the merged one. The card adapts to whichever it
finds, by looking for `adapter_config.json` rather than taking a flag, so an
adapter gets `library_name: peft` and peft usage while merged weights get
plain transformers usage.

Uploading needs a Hugging Face token with write access, which is the first
point in this project that requires credentials. Downloading Gemma needs
none, since the repositories are ungated.

Section 3.1 of the Gemma Terms of Use attaches four conditions to
distributing a derivative. Every recipient gets the agreement, which the
`license: gemma` declaration carries. The Section 3.2 use restrictions pass
forward, which the card states. Modified files carry a modification notice,
which the card states. A NOTICE file ships with the exact required sentence,
which `publish.py` writes. Generating those from code keeps a hand-rolled
upload from dropping one.

`data/train.jsonl` is not published. The three training novels are public
domain in the United States, and copyright in the European Union and the
United Kingdom runs for the author's life plus 70 years, so they stay
protected there until 2032. The dataset holds verbatim passages, the weights
do not, and the dataset is reproducible from this repository with two
commands.

The card also records that 918 examples over three epochs is small enough for
near-verbatim reproduction of training passages to be plausible.

## Which size

Gemma 4 ships an E4B at 8.00B parameters and a 12B at 11.96B. E4B is the
on-device tier and is the default here, because the point is a model that
runs on a laptop.

| | E4B | 12B |
|---|---|---|
| bf16 weights | 14.9 GiB | 22.3 GiB |
| 4-bit for a laptop | about 4.5 GB | about 7 GB |
| LoRA parameters at rank 32 | 74.2M | 131.1M |
| Per-Layer Embeddings | 129 tensors | none |

E4B uses Per-Layer Embeddings, an architecture built for on-device
efficiency, and its `hidden_size_per_layer_input` is 256. The 12B sets that
field to 0 and carries none of those tensors. Unsloth's own guidance puts
E4B LoRA at 17GB of VRAM and raises no concern about training it through
plain PEFT, so the default trains E4B in bf16 with adapters on the seven
projections and nothing touching the embedding tables.

E4B is also multimodal with separate vision and audio encoders, whose
projections share the target names. Adapters attach there and take no
gradient from a text-only run, which wastes 0.8M of 74.2M adapter
parameters. Tower tensors are numerous and small, so the waste is not worth
a narrower pattern.

Pass `--model google/gemma-4-12B-it` to train the larger one. It has no
Per-Layer Embeddings and all its projections sit under `language_model`, so
it is the fallback if E4B ever fails to train.

```
just evaluate
```

Generates from the base model and from the adapter on six prompts the
training set never saw, prints a style profile for the held-out book and
for both models, then prints every sample in full. The profile reports
sample size in words, mean sentence length, and adverb rate. Three rows
answer one question: the held-out book says where the target sits, and
the gap between base and tuned says whether the run moved toward it.

Mean sentence length swings with dialogue density, from roughly 6 words
in dialogue-heavy passages to 20 in narration, so the sample size prints
beside it and a six-prompt sample carries real noise.

Read the samples. The numbers can look healthy while the prose is
degenerate, and only the samples show that.

## The GPU host

Training and evaluation need one VM with one GPU. No cluster, no
InfiniBand. Peak GPU memory is roughly 25 to 30GB, so an L40S at 48GB has
comfortable headroom and an H100 at 80GB buys nothing at twice the rate.

`infra/` holds a Terraform module for that instance. Provider credentials
come from the `nebius` CLI profile, so no keys are configured here.

```
cd infra
cp terraform.tfvars.example terraform.tfvars   # set ssh_public_key_path
terraform init
terraform apply
```

The module creates a 200GiB `NETWORK_SSD` boot disk from an Ubuntu 24.04
CUDA image and one `gpu-l40s-a` instance on the `1gpu-16vcpu-64gb` preset,
with a public address for SSH. Cloud-init adds your login, installs git
and rsync, and installs uv. The disk is sized for the model weights, the
Hugging Face cache, and a torch environment together.

`terraform output ssh` and `terraform output rsync_data` print the two
commands you need next. The public key is read from a file rather than
pasted, so a wrong path fails the plan instead of building an instance you
cannot log into.

Two ways to stop paying. `terraform apply -var stopped=true` halts the
instance and keeps the disk, so the downloaded weights survive for the
next run. `terraform destroy` removes both. Stopping still bills the
disk, so destroy once you are finished.

## Cost and runtime

Generating instructions for the 987 training passages is the only paid
step. Training is roughly
20 to 45 minutes on a single L40S or H100, which is a few dollars of
rented time.

## Development

`just check` runs the read-only gate: `ruff format --check`,
`ruff check`, `ty check`, and `pytest`. Agent guidance and the full
style rules live in `AGENTS.md`.
