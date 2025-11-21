# Repository Guidelines

## Project Structure & Modules
- `src/ai_draft_bot/`: core package with data ingest (`data/ingest_17l.py`), feature builders (`features/draft_context.py`), and model wrappers (`models/drafter.py`).
- `scripts/train.py`: Typer CLI for training; writes artifacts like `artifacts/model.joblib`.
- `tests/`: add pytest modules here; keep fixtures under `tests/data/` to avoid polluting source directories.

## Setup & Key Commands
- Create a venv and install deps: `python -m venv .venv; . .venv/bin/activate; pip install -e .[dev]`.
- Lint: `ruff check .` enforcing line length 100 and basic hygiene (E,F,I,B).
- Type-check: `mypy src` (strict mode enabled via `pyproject.toml`).
- Tests: `pytest` (respects `tests/` test paths); use `-q` for quicker iteration.
- Train: `python scripts/train.py run --drafts-path drafts.jsonl --metadata-path cards.csv --output-path artifacts/model.joblib`.
- Simulate picks: `python -m ai_draft_bot.cli simulate --model-path artifacts/model.joblib --metadata-path cards.csv --pack "Card A" --pack "Card B"`.

## Coding Style & Naming
- Python 3.10; prefer explicit type hints and dataclasses where they add clarity.
- Modules/files lowercase with underscores; functions/vars snake_case; classes PascalCase; CLI options are kebab-case.
- Keep functions short and pure where possible; document tricky logic with concise comments.

## Testing Guidelines
- Write pytest cases in `tests/test_*.py`; mirror package structure for clarity.
- Use small fixture slices of 17L exports; avoid committing full datasets or artifacts.
- Cover ingest parsing, feature building edge cases (missing metadata, empty packs), and model scoring.

## Commit & Pull Requests
- Use imperative, scoped commits (e.g., `Add draft context features`, `Fix train CLI defaults`).
- Keep PRs focused; include purpose, notable changes, and sample commands/output from training or simulation.
- Link issues when available and note any data assumptions or reproducibility steps.

## Data & Security
- Do not check in raw 17L exports or large `joblib` artifacts; store under `artifacts/` locally and git-ignore as needed.
- Validate input paths in CLIs and avoid embedding secrets or PII in logs or fixtures.
