"""EleutherAI lm-evaluation-harness integration (optional).

Runs standardized academic benchmarks (hellaswag, arc, winogrande, mmlu, …)
against a fine-tuned model — complementary to the in-house perplexity/ROUGE/BLEU
metrics. ``lm-eval`` is a heavy, optional dependency that is imported lazily:

    pip install lm-eval

so importing this module never requires it. Call ``lm_eval_available()`` to
check before invoking ``run_benchmarks``.
"""

from __future__ import annotations

# A small, fast default set — full MMLU etc. can be requested explicitly.
DEFAULT_TASKS = ["hellaswag", "arc_easy", "winogrande"]


def lm_eval_available() -> bool:
    """True when the lm-eval harness is importable."""
    try:
        import lm_eval  # noqa: F401

        return True
    except ImportError:
        return False


def run_benchmarks(
    model_path: str,
    tasks: list[str] | None = None,
    *,
    limit: int | None = None,
    device: str = "auto",
    batch_size: int | str = "auto",
) -> dict[str, dict[str, float]]:
    """Run lm-eval ``tasks`` against the HF model directory at ``model_path``.

    Returns ``{task: {metric: value}}`` with only numeric metrics kept. Pass
    ``limit`` to cap examples per task (useful for a quick smoke run). Raises
    ``RuntimeError`` if lm-eval is not installed.
    """
    if not lm_eval_available():
        raise RuntimeError("lm-eval is not installed. Run: pip install lm-eval")

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model_path, device=device, batch_size=batch_size)
    results = simple_evaluate(model=lm, tasks=tasks or DEFAULT_TASKS, limit=limit)

    out: dict[str, dict[str, float]] = {}
    for task, metrics in (results.get("results") or {}).items():
        out[task] = {k: v for k, v in metrics.items() if isinstance(v, int | float)}
    return out
