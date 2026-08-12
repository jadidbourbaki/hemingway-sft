# hemingway-sft

One stage of LoRA supervised fine-tuning that teaches Gemma 4 12B to
answer ordinary prose requests in Hemingway's voice.

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

```
just train
```

Fits a LoRA adapter with TRL's `SFTTrainer` and saves it to
`runs/hemingway-12b`. The defaults are rank 32, three epochs, and a
learning rate of 1e-4 on a cosine schedule. Rank 32 rather than 16
because the run has to overwrite an existing stylistic prior rather than
teach a fresh output format. Weights stay in bf16 with no quantization.

## Why the 12B and not the E4B

Gemma 4 ships an E4B at 8.00B parameters and a 12B at 11.96B, and the
smaller one looked like the obvious pick until the checkpoints settled it.

E4B uses Per-Layer Embeddings, carrying 129 tensors of per-layer embedding
machinery with `hidden_size_per_layer_input` at 256. The 12B sets that
field to 0 and carries none of those tensors, so it is an ordinary dense
transformer and PEFT has nothing unusual to handle. Choosing the 12B
removes the one architectural unknown in the project.

E4B is also multimodal with separate encoders, so `q_proj` and the other
target names appear in its vision and audio towers as well. Suffix
matching would attach adapters there, though the cost is small: 0.8M of
74.2M LoRA parameters, because tower tensors are numerous and tiny. The
12B projects image patches straight into the embedding space with no
separate tower, so all 328 of its matches sit under `language_model`.

The 12B costs memory rather than risk. Its bf16 weights are 22.3GiB
against 14.9GiB, and rank 32 gives 131.1M LoRA parameters against 74.2M,
so roughly 30 to 35GB in total against 22 to 25GB. Both fit an L40S at
48GB. Pass `--model google/gemma-4-E4B-it` to train the smaller one.

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
