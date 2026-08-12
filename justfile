# LoRA style fine-tuning of Gemma 4 E4B on public-domain Hemingway. Requires uv.

# List recipes.
default:
    @just --list

# Download the novels and cut them into passages.
corpus *args:
    uv run hemingway-corpus {{ args }}

# Generate an instruction per passage and write the SFT pairs. Needs Anthropic credentials.
instructions *args:
    uv run hemingway-instructions {{ args }}

# Train the LoRA adapter. Expects a CUDA host.
train *args:
    uv run hemingway-train {{ args }}

# One epoch over 20 examples, to prove the model loads and steps before a real run.
smoke *args:
    uv run hemingway-train --smoke --epochs 1 {{ args }}

# Compare the adapter against the base model on unseen prompts.
evaluate *args:
    uv run hemingway-evaluate {{ args }}

# Fold the adapter into the base weights, ready to quantize for a laptop.
merge *args:
    uv run hemingway-merge {{ args }}

# Push an artifact to the Hub with the notices Gemma requires. Needs a write token.
publish *args:
    uv run hemingway-publish {{ args }}

# Format with ruff.
fmt:
    uv run ruff format .

# Lint with ruff (use `just lint --fix` to apply fixes).
lint *args:
    uv run ruff check {{ args }} .

# Type-check with ty.
typecheck:
    uv run ty check .

# Run the unit tests.
test:
    uv run pytest

# Read-only checks, the way CI would run them.
check:
    uv run ruff format --check .
    uv run ruff check .
    uv run ty check .
    uv run pytest
