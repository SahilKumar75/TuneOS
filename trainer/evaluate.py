"""Model evaluation — thin wrapper over the pluggable metric registry."""

from __future__ import annotations

from trainer.metrics import REGISTRY


def evaluate_model(
    model,
    tokenizer,
    test_dataset,
    metrics: list[str] | None = None,
) -> dict:
    """Compute the requested loss-based metrics on a held-out dataset.

    Defaults to perplexity. Reference-based metrics (rouge1, bleu) are skipped
    here because they need generated predictions; request them explicitly via
    ``evaluate_references`` once predictions are available.

    Returns a dict like ``{"perplexity": 12.3}``. ``bleu`` is kept in the result
    (as ``None``) for backward compatibility with existing consumers.
    """
    metric_names = metrics or ["perplexity"]
    results: dict = {"bleu": None}
    for name in metric_names:
        metric = REGISTRY.get(name)
        if metric is None or metric.kind != "loss":
            continue
        results[name] = metric.fn(model, tokenizer, test_dataset)
    results.setdefault("perplexity", None)
    return results


def generate_predictions(
    model,
    tokenizer,
    instructions: list[str],
    max_new_tokens: int = 128,
) -> list[str]:
    """Greedily generate a response for each instruction, returning only the text
    after the prompt. Used to obtain predictions for reference metrics."""
    import torch

    from trainer.dataset import PROMPT_TEMPLATE

    predictions: list[str] = []
    device = next(model.parameters()).device
    model.eval()
    for instruction in instructions:
        prompt = PROMPT_TEMPLATE.format(instruction=instruction, output="")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1] :]
        predictions.append(tokenizer.decode(generated, skip_special_tokens=True).strip())
    return predictions


def evaluate_references(
    predictions: list[str],
    references: list[str],
    metrics: list[str] | None = None,
) -> dict:
    """Compute reference-based metrics (rouge1, bleu) over prediction/reference pairs."""
    metric_names = metrics or ["rouge1", "bleu"]
    results: dict = {}
    for name in metric_names:
        metric = REGISTRY.get(name)
        if metric is None or metric.kind != "reference":
            continue
        results[name] = metric.fn(predictions, references)
    return results
