# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Observability layer (Phase 1).** Step-level metrics are now
  persisted to queryable `run_metrics` and `run_params` tables in
  `storage/experiments.db`, instead of only living in a JSON blob.
- `GET /api/jobs` now returns all runs from the durable SQLite store (was a stub
  returning `[]`).
- Durable job lifecycle: the worker writes job status to SQLite at start,
  completion, and failure, so job state survives a Redis restart.
- `artifact_path(job_id, artifact)` helper centralizes output-path construction
  across the API and workers.
- Automatic LoRA `target_modules` detection per model architecture — fixes
  silent breakage on Gemma, Phi-3, Falcon, Qwen2, and GPT-NeoX.
- Basic API documentation for backend services.
- GitHub Action workflows for CI and Release.
- Issue and PR templates for standardized contribution tracking.
- Contributor Covenant Code of Conduct and Security Policy.
- Empty `tests` directory with initial pytest configuration.

### Changed
- State data models embedded in `rx.State` now use `pydantic.BaseModel`
  (documented in `CONTRIBUTING.md`); `rx.Base` was removed in newer Reflex.
- Explicit lockfile (`poetry.lock`) pinning policy in `CONTRIBUTING.md`.
- Removed stale 'Finetune Platform' branding references and standardized to 'TuneOS'.

### Removed
- Untracked `__pycache__`, `build/`, and `dist/` from the git repository.
