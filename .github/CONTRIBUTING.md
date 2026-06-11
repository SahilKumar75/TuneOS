# Contributing to TuneOS

Contributions are welcome — whether it's a typo fix, a bug report, or a new feature.
This document covers the development workflow, code standards, and how to open a
good pull request.

---

## Getting Started

### Prerequisites

- Python 3.10 or newer
- [Poetry](https://python-poetry.org/)
- Docker Desktop (for background training workers)
- Git

### Development Setup

1. Fork the repository and clone it locally:

```bash
git clone https://github.com/<your-username>/TuneOS.git
cd TuneOS
```

1. Install dependencies:

```bash
poetry install
```

1. Set up environment variables:

```bash
cp .env.example .env
# Edit .env and add your Hugging Face token
```

1. Start the backend services:

```bash
docker-compose up redis worker
```

1. Run the Reflex UI in development mode:

```bash
poetry run reflex run
```

---

## Running Tests

```bash
# Full suite (no GPU required)
poetry run pytest

# With coverage
poetry run pytest --cov=app --cov=trainer --cov=workers
```

The trainer integration test loads a real model and runs a training step. It is
skipped by default:

```bash
TUNEOS_INTEGRATION_TESTS=1 poetry run pytest tests/test_trainer_integration.py
```

See [docs/testing.md](../docs/testing.md) for the full testing guide, including
manual test scenarios and how to add a new evaluation metric.

---

## Dependency Management

Use Poetry for all dependency changes. **Always commit `poetry.lock`** alongside
`pyproject.toml` so environments are reproducible across machines.

```bash
poetry add <package>        # add a runtime dependency
poetry add -G dev <package> # add a dev-only dependency
```

---

## Code Style

Ruff handles both linting and formatting. CI will fail if either check fails.

```bash
poetry run ruff check .       # lint
poetry run ruff format .      # apply formatting
poetry run ruff format --check .  # check only (what CI runs)
```

Type checking with mypy runs over the pure-logic backend modules:

```bash
poetry run mypy
```

---

## Architecture Notes

### Project Layout

```
TuneOS/
├── app/
│   ├── pages/          Route-level page components
│   ├── components/     Reusable UI components
│   └── state/          Reflex state (FinetuneState, TrainingPollerState, DeployState)
├── trainer/            ML training logic (adapters, finetune, dpo, metrics, evaluate)
├── workers/            Celery task definitions (sft, dpo, kd, vision queues)
├── desktop/            PyQt6 desktop shell
├── docs/               Documentation
└── tests/              Test suite
```

### State Data Models

Any data class held inside a Reflex `rx.State` var (e.g. a list bound to
`rx.foreach`) must inherit from `pydantic.BaseModel`:

```python
from pydantic import BaseModel

class ExperimentRun(BaseModel):
    id: str = ""
    name: str = ""
```

Do not use `rx.Base` — it was removed in newer Reflex. Always annotate state vars
with a concrete type (`list[ExperimentRun]`, not `list[dict[str, Any]]`).

### Evaluation Metrics

Metrics live in a registry in `trainer/metrics.py`. To add one:

```python
@register("my_metric", greater_is_better=True, kind="reference")
def compute_my_metric(predictions: list[str], references: list[str]) -> float | None:
    ...
```

Use `kind="loss"` for metrics over `(model, tokenizer, dataset)` and
`kind="reference"` for metrics over `(predictions, references)` string pairs.

---

## Pull Requests

1. Create a branch from `main`:

```bash
git checkout -b feat/my-feature
```

1. Make focused, atomic commits.
1. Ensure all tests pass and the code is formatted.
1. Open a pull request using the [PR template](pull_request_template.md).

### Commit Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | When to use |
|---|---|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `refactor:` | Code change that neither fixes a bug nor adds a feature |
| `test:` | Adding or updating tests |
| `build:` | Build system or dependency changes |
| `ci:` | CI/CD configuration changes |

---

## Issues

Open an issue at [github.com/SahilKumar75/TuneOS/issues](https://github.com/SahilKumar75/TuneOS/issues).
Use the provided templates to give enough context to act on it quickly.

---

## Code of Conduct

By participating in this project you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

By contributing to TuneOS you agree that your contributions will be licensed under
the [Apache 2.0 License](../LICENSE).
