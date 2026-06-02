# Contributing to TuneOS

Thank you for your interest in contributing to TuneOS! We welcome contributions from the community — whether it's fixing a typo, reporting a bug, or building a major feature.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/)
- Docker Desktop (for background training workers)
- Git

### Development Setup

1. Fork the repository and clone it locally:
   ```bash
   git clone https://github.com/<your-username>/TuneOS.git
   cd TuneOS
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Set up your environment variables:
   ```bash
   cp .env.example .env
   # Edit .env and add your Hugging Face token
   ```

4. Start the backend services:
   ```bash
   docker-compose up redis worker
   ```

5. Run the Reflex UI in development mode:
   ```bash
   poetry run reflex run
   ```

### Running Tests

```bash
poetry run pytest
```

To run with coverage:
```bash
poetry run pytest --cov=app --cov=trainer --cov=workers
```

### Dependency Management

We use Poetry for dependency management. To ensure reproducibility across all environments, **you must commit the `poetry.lock` file** with any dependency changes.

If you add or remove a dependency, run:
```bash
poetry add <package>
```
This automatically updates `pyproject.toml` and `poetry.lock`. Ensure both files are included in your pull request.

### Code Style

We use **Ruff** for linting and formatting:
```bash
poetry run ruff check .
poetry run ruff format .
```

CI will fail if linting or formatting issues are present.

## Architecture Overview

```
TuneOS/
├── app/              # Reflex UI (pages, components, state)
│   ├── pages/        # Route-level page components
│   ├── components/   # Reusable UI components
│   └── state/        # Reflex state management
├── trainer/          # ML training logic (LoRA, QLoRA, evaluation)
├── workers/          # Celery task definitions for async training
├── desktop/          # PyQt6 desktop shell
├── docs/             # Project documentation
└── tests/            # Test suite
```

### State Data Models

Any data class used **inside** a Reflex `rx.State` var (e.g. a list of records
bound to `rx.foreach`) must inherit from `pydantic.BaseModel`:

```python
from pydantic import BaseModel

class ExperimentRun(BaseModel):
    id: str = ""
    name: str = ""
```

Do **not** use `rx.Base` — it was removed in newer Reflex versions. Pydantic
models also give Reflex a concrete element type, which `rx.foreach` requires;
an untyped `list[dict[str, Any]]` state var will fail to compile. Always
annotate state vars with a concrete type (e.g. `list[ExperimentRun]`), never
`Any`.

## Pull Requests

1. Create a new branch from `main` for your feature or bugfix:
   ```bash
   git checkout -b feat/my-feature
   ```
2. Make your changes with clear, atomic commits.
3. Ensure all tests pass and code is formatted.
4. Submit a pull request with a clear description using the [PR template](.github/pull_request_template.md).

### Commit Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` — New feature
- `fix:` — Bug fix
- `docs:` — Documentation only
- `refactor:` — Code change that neither fixes a bug nor adds a feature
- `test:` — Adding or updating tests
- `build:` — Build system or dependency changes
- `ci:` — CI/CD configuration changes

## Issues

If you find a bug or have a feature request, please [open an issue](https://github.com/SahilKumar75/TuneOS/issues). Use the provided issue templates to give us enough context to act on it quickly.

### Issue Labels

See [docs/issue-labels.md](docs/issue-labels.md) for the full labeling taxonomy.

## License

By contributing to this project, you agree that your contributions will be licensed under its [Apache 2.0 License](LICENSE).
