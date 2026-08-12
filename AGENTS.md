# AGENTS.md

Guidance for AI agents working on the hemingway-sft repo. The CLAUDE.md
symlink resolves to this file. Read it top to bottom on first session.

The style rules here are carried over from the braid repo. They are the
maintainer's stated preference across every repo and must be honored.

## Project context

hemingway-sft is a learning project about post-training. The goal is a
small open model that answers ordinary prose requests in Hemingway's
voice, produced by one stage of LoRA supervised fine-tuning.

The pipeline runs in four steps, each a console script with a matching
just recipe. `corpus` downloads the four Hemingway novels that are in
the US public domain and cuts them into passages. `instructions` asks
Claude to write the instruction that would have produced each passage,
which turns raw novel text into supervised instruction pairs. `train`
fits a LoRA adapter on those pairs. `evaluate` compares the adapter
against the base model on prompts the training set never saw.

Reverse instructions is the technique from Köksal et al. 2023,
"LongForm: Effective Instruction Tuning with Reverse Instructions".
Training on raw novel text instead would produce a model that continues
Hemingway rather than one that answers a request in his voice. The
related instruction backtranslation of Li et al. 2023 trains a backward
model and filters its own output, and neither step happens here.

The repo is Python only. Corpus preparation and instruction generation
run on a laptop. Training and evaluation expect a rented CUDA host.

## Repository layout

```
hemingway_sft/   the package: corpus, reverse_instructions, train, evaluate, style
tests/           pytest suite for the modules that carry no GPU dependency
data/            fetched corpus and derived datasets, gitignored
runs/            adapter checkpoints, gitignored
justfile         task runner
```

Everything under `data/` is reproducible from `just corpus` and
`just instructions`, so none of it is committed.

## Quality gate

`just check` runs `ruff format --check`, `ruff check`, `ty check`, and
`pytest`, the way CI would. If the gate does not pass locally, the work
is not done.

Importing the training stack costs about two seconds, so keep pure logic
in modules the tests can import without it. The `style` module is
separate for that reason, and because the metrics are the object a reward
function would score if this project ever moved from supervised
fine-tuning to reinforcement learning.

Two seconds is worth paying where it buys real coverage. `test_train.py`
imports trl on purpose, because `SFTConfig` raises `TypeError` on a field
it does not define, and building one is the cheapest way to catch a
hyperparameter that the installed version renamed. A bad field in that
file otherwise surfaces on a rented GPU.

Keep the metric set small. Every metric added has to survive the question
of what a reader would do differently on seeing it. A metric that needs a
caveat to interpret has already failed that question, so it belongs
nowhere near the gate.

## Writing prose

The following style rules apply to all prose in the repo: README, docs,
design notes, reports, commit message bodies, code comments.

### Hard rules

- **No em-dashes.** The character does not appear in prose. If you would
  normally use an em-dash, split into two sentences or use a comma. The
  same goes for en-dashes in prose.
- **No semicolons in prose.** Use a period and start a new sentence.
- **No unnecessary parentheses.** A parenthetical aside that pauses the
  reader for a thought you could have put in its own sentence should go
  in its own sentence. Parens are fine for genuine clarifications, such
  as an abbreviation on first use, but not as a substitute for a comma or
  period.
- **No ASCII diagrams.** Describe relationships in prose. A single
  inline arrow like `passage -> instruction` is fine, boxes and arrows
  are not.
- **No emoji** unless the user explicitly asks for them.
- **No vague back-references.** Do not open a sentence with "This",
  "That", "These", "Those", "Their", or "It" pointing at a noun from an
  earlier sentence. Name the noun again. "This is the wrong reward"
  becomes "Sentence length is the wrong reward." The reader should never
  have to look backward to resolve what a pronoun stands for.

### Soft rules

- Write short, direct sentences. If a sentence has more than one comma,
  consider whether it should be two sentences.
- Lead with the noun, not the qualifier. "The chunker closes on a
  paragraph boundary" beats "When a passage fills up, the chunker closes
  on a paragraph boundary."
- Define jargon on first use, even if you think the reader knows it.
- Do not write in fragments or in a punchy, aphoristic style. Short
  clipped clauses strung together read like a parable, not like
  documentation. "One stage, one reward, cheap" is wrong. "One
  supervised stage is enough, because style is a per-token property of
  the training completions" is right.

## Writing comments

The rules under "Writing prose" apply, plus:

- **Default to writing no comment.** A well-named identifier and a short
  function explain themselves. Only comment when the why is non-obvious:
  a hidden constraint, a subtle invariant, a workaround for a specific
  upstream bug, behavior that would surprise a reader.
- **Don't describe what the code does.** The code does that.
- **Say what is done and why, not what was avoided.** Frame a comment
  around the present behavior and its reason, not a rejected
  alternative. "A passage closes on a paragraph boundary so no
  completion ends mid-sentence" beats "we do not split paragraphs".
- **Don't reference the past.** "Renamed from X", "formerly Y" all rot.
  Comments describe the present state.
- **Don't reference callers or PRs.** They rot as the code evolves.
- **Don't write multi-line comment banners.** One short comment per
  declaration.

## Writing documentation and reports

The prose rules above apply, plus a few that matter for a reader
following a page top to bottom.

- **Explain every code block.** Never drop a command or snippet without
  saying what it does and what every meaningful flag means. Show output
  too, and say what its columns or fields mean.
- **Headings name content, and are not narration.** "Corpus and
  licensing" and "Cost and runtime" are good. "Now we download the
  novels" narrates the act instead of naming the subject.
- **Do not over-chunk.** A heading breaks the reader's flow. Add one
  only where a genuinely new section begins.
- **Cut filler.** Remove words that earn nothing. Do not lean on one
  adjective across a passage.

## Writing tests

Test the happy path for every public function, every validation branch
that raises, and boundary cases for numeric inputs. Test the fakes
themselves. A silently broken fake hides regressions.

Use `pytest`. Parametrize with `@pytest.mark.parametrize` for multiple
cases of the same shape, standalone `test_<scenario>` functions
otherwise. Prefer plain `assert`. Use hand-written fakes over mock
libraries where practical, and `monkeypatch` for network-shaped
dependencies.

A bug worth fixing is worth a regression test. The paragraph filter once
dropped every line of dialogue, which is most of this corpus, because
the terminal-punctuation class omitted curly quotes. A test now pins
that behavior.

Three functions are deliberately uncovered: `build_trainer`, `load_model`,
and `generate`. Each one wires up a real model, so a unit test would have
to fake most of the transformers and TRL loading surface, and the fake
would pin the shape of the fake rather than the behavior of the code.
Running the pipeline is the test for those three. Every other public
function carries one, and a new function without a test is a gap rather
than a fourth exception.

## Commit messages

Conventional commits, one sentence each, no body unless absolutely
necessary.

```
feat: add style metrics for held-out comparison
fix: keep dialogue paragraphs that close on a curly quote
docs: record the cost of a full instruction-generation run
chore: pin trl to 1.9.2
```

Rules:

- One sentence subject. Pick a tense and be consistent.
- Lowercase the type and the first word after the colon, unless it is a
  proper noun or acronym.
- No commit body unless the reason cannot fit in the subject. No "Test
  plan" or "Summary" boilerplate.
- **No `Co-Authored-By: Claude` trailer.** Ever, even when commits are
  authorized in advance.
- Do not amend or rewrite published commits without explicit user
  consent. Force-push only with `--force-with-lease`, only on a feature
  branch, and only after confirming.

## Git practices

- Use whatever git identity the user has configured. Never pass
  `-c user.email` or `-c user.name`.
- Don't push without explicit authorization. Ask before every commit and
  every push, each time, even after a prior yes.
- For PR merges, prefer `gh pr merge <num> --squash --delete-branch`.
- Before any destructive operation, confirm with the user. Use
  `--force-with-lease`, never `--force`.

## Python style

These rules keep the code consistent: type everything, fail loudly,
prefer the standard library, and let the tooling enforce the rest.

### Tooling

The toolchain is Astral's, and it is not optional.

- **uv** for environments and dependencies. Not pip, not poetry, not a
  bare `requirements.txt`. Use `uv add`, `uv lock`, `uv run`.
- **ruff** for both linting and formatting. It replaces black, isort,
  and flake8.
- **ty** for type checking. It is Astral's checker and still young, but
  it is the house checker. Not mypy or pyright.
- **just** for task running.

### Project layout and dependencies

- Runtime dependencies go in `[project].dependencies`. Dev tools like
  ruff and ty go in `[dependency-groups].dev` per PEP 735, so `uv run`
  installs them by default.
- Pin every direct dependency to an exact version with `==` and commit
  `uv.lock`. The lockfile resolves the full tree with hashes, so
  installs are both reproducible and verified against supply-chain
  tampering. A training run is something you reproduce, so pinning beats
  flexibility.
- The lockfile is committed and never hand-edited.
- `torch` resolves to a CUDA build on Linux and a Metal build on macOS
  from the same pinned version, so one pin serves the laptop and the
  rented GPU host.

### Types

Type every function signature, both parameters and return. `ty check`
runs in the gate, so an untyped surface is a failing build.

- Put `from __future__ import annotations` at the top of every module.
- Use built-in generics, `list[int]` and `dict[str, T]`, not
  `typing.List`. Use `X | None`, not `Optional[X]`.
- Model structured data that crosses a boundary with a Pydantic model or
  a dataclass. Do not pass bare dicts whose shape lives only in your
  head. A signature like `list[dict[str, Any]]` is the warning sign: the
  rows have a shape, so give the shape a model and the signature the
  model's name. Every JSONL row in this repo has a model.
- `Any` is acceptable only for a third-party object whose stubs are
  missing or wrong, such as a transformers model or tokenizer.

### Imports

Every import goes at the top of the module. A function-level import
hides a dependency from both the reader and the tooling, and it defers
an ImportError from startup to call time. Import inside a function only
to break a genuine circular import or to keep an optional dependency
optional, and name the reason in a comment.

### Naming and errors

- `snake_case` for functions and variables, `PascalCase` for classes,
  `UPPER_SNAKE` for module constants. A leading underscore marks a name
  module-private, including compiled regexes. Do not uppercase acronyms
  the way Go does, PEP 8 wins: `HTTPClient` as a class, but `url` and
  `id` lowercase.
- Raise exceptions, do not return sentinel values to signal failure.
- Catch narrowly. A broad `except Exception` belongs only at a top-level
  boundary where you log and carry on. A bare `except:` is never
  correct. Use `raise ... from err` to preserve the cause.
- A long batch job logs the failure for one item and continues. Losing
  900 finished API calls because item 901 failed is worse than skipping
  item 901.

### Don't reinvent the wheel

Prefer something already built, the standard library or a
well-maintained dependency, over code you write yourself. A vetted
package or a stdlib helper is almost always more correct and better
tested than a version written under deadline. What is not welcome is
hand-rolling logic that a mature library already solves.

### Talking to Claude

Reverse instruction generation calls the Anthropic API through the
official `anthropic` client. Default to `claude-opus-5` and expose `--model` so a cheaper
model is the user's choice rather than a silent default. Use
`messages.parse` with a Pydantic `output_format` so the response is
validated rather than string-parsed. Keep adaptive thinking on and lower
`output_config.effort` when a call is cheap and simple, because
disabling thinking on Opus 5 can leak internal tags into the response.

Long runs write each result to a cache file as it arrives, keyed so a
resumed run skips finished work. An interrupted run must never pay for
the same passage twice.

## Working with the user

### Risk and reversibility

Local, reversible actions such as editing a file or running a test need
no preamble. Hard-to-reverse actions such as a force-push, a deleted
branch, or a paid API run over the whole corpus need explicit
confirmation each time. Authorization for one action does not extend to
similar actions.

### Communication style

- Default to terse. The user reads diffs and can see what changed.
- Lead with the result, then the details if asked.
- One or two sentence end-of-turn summary: what shipped and what is
  next.
- Don't restate the request back. Don't open with "Great question."
- When you spot a side-effect the user did not ask for, name it and ask
  before doing it.

### Scope

Match the scope of your changes to what the user asked. A bug fix does
not get a free refactor. If a side-improvement is one line and zero
behavior change, do it. If it is more, surface it as a separate option
to opt into.

## When in doubt

Re-read this document, then the most recent changes that touched the
same area. The patterns are intentionally consistent. Match them rather
than introducing a new variation.
