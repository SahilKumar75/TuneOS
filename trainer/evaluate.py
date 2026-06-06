"""Model evaluation — thin wrapper over the pluggable metric registry."""

from __future__ import annotations

import logging

from trainer.metrics import REGISTRY

_logger = logging.getLogger(__name__)


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


def evaluate_run(
    model,
    tokenizer,
    model_cfg: dict,
    train_cfg: dict,
    dataset_path: str,
    hub_dataset_id: str = "",
    hub_split: str = "train",
    instruction_col: str = "instruction",
    output_col: str = "output",
) -> dict:
    """Evaluate a trained model on a 20% sample of its data.

    Returns a metrics dict (perplexity + best-effort rouge1/bleu). Never raises —
    on any failure it logs the reason and returns the null-metric fallback so a
    job never fails on eval. Lives in the framework-light ``trainer`` layer so it
    is shared by the local Celery worker and the Modal runner without dragging in
    redis/celery on the remote side.
    """
    from trainer.config import ModelConfig, TrainingConfig
    from trainer.dataset import load_and_tokenize, load_instruction_pairs

    try:
        cfg = ModelConfig(**model_cfg)
        seed = TrainingConfig(**train_cfg).seed
        full_dataset = load_and_tokenize(
            dataset_path,
            tokenizer,
            cfg.max_seq_length,
            hub_dataset_id=hub_dataset_id,
            hub_split=hub_split,
            instruction_col=instruction_col,
            output_col=output_col,
        )
        n_eval = max(1, int(0.2 * len(full_dataset)))
        eval_sample = full_dataset.shuffle(seed=seed).select(range(n_eval))
        eval_results = evaluate_model(model, tokenizer, eval_sample)

        # Reference metrics (ROUGE-1/BLEU) need generated predictions vs. the
        # raw reference text. Sample the same indices from the raw pairs.
        try:
            raw_pairs = load_instruction_pairs(
                dataset_path,
                hub_dataset_id=hub_dataset_id,
                hub_split=hub_split,
                instruction_col=instruction_col,
                output_col=output_col,
            )
            n_ref = min(len(raw_pairs), 32)
            ref_sample = raw_pairs.shuffle(seed=seed).select(range(n_ref))
            instructions = [row["instruction"] for row in ref_sample]
            references = [row["output"] for row in ref_sample]
            predictions = generate_predictions(model, tokenizer, instructions)
            eval_results.update(
                evaluate_references(
                    predictions,
                    references,
                    metrics=["rouge1", "rouge2", "rougeL", "bleu", "meteor"],
                )
            )
        except Exception as e:
            # Reference metrics are best-effort; never let them break eval.
            _logger.warning("Reference eval failed (%s): %s", type(e).__name__, e)
            for _m in ("rouge1", "rouge2", "rougeL", "bleu", "meteor"):
                eval_results.setdefault(_m, None)

        return eval_results
    except Exception as e:
        _logger.warning("Evaluation failed (%s): %s", type(e).__name__, e)
        return {"perplexity": None, "rouge1": None, "bleu": None}


def generate_predictions(
    model,
    tokenizer,
    instructions: list[str],
    max_new_tokens: int = 128,
    batch_size: int = 8,
    generation_config: dict | None = None,
) -> list[str]:
    """Generate a response for each instruction (batched), returning only the
    text after the prompt. Used to obtain predictions for reference metrics.

    Batched generation is much faster than one-by-one on GPU. ``generation_config``
    overrides the defaults (greedy) — e.g. ``{"do_sample": True, "temperature": 0.7}``.
    """
    import torch

    from trainer.dataset import PROMPT_TEMPLATE

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if generation_config:
        gen_kwargs.update(generation_config)

    # Left-pad so generated tokens align at the end of each row in the batch.
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    predictions: list[str] = []
    device = next(model.parameters()).device
    model.eval()
    try:
        for i in range(0, len(instructions), batch_size):
            chunk = instructions[i : i + batch_size]
            prompts = [PROMPT_TEMPLATE.format(instruction=ins, output="") for ins in chunk]
            inputs = tokenizer(prompts, return_tensors="pt", truncation=True, padding=True).to(
                device
            )
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)
            gen = out[:, inputs["input_ids"].shape[1] :]
            predictions.extend(
                tokenizer.decode(row, skip_special_tokens=True).strip() for row in gen
            )
    finally:
        tokenizer.padding_side = prev_side
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
