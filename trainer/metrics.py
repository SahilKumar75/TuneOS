"""Pluggable evaluation metric registry.

Metrics are plain functions registered by name. Two families are supported:

* **Loss-based** (e.g. perplexity) — operate on a tokenized dataset and the model.
* **Reference-based** (e.g. ROUGE-1, BLEU) — operate on (prediction, reference)
  string pairs.

The registry lets callers request any subset of metrics by name without the
evaluation code knowing the concrete implementations.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

# ── Registry ──────────────────────────────────────────────────────


@dataclass
class Metric:
    name: str
    fn: Callable
    greater_is_better: bool
    # "loss" → fn(model, tokenizer, dataset) -> float
    # "reference" → fn(predictions: list[str], references: list[str]) -> float
    kind: str


REGISTRY: dict[str, Metric] = {}


def register(name: str, *, greater_is_better: bool, kind: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        REGISTRY[name] = Metric(name=name, fn=fn, greater_is_better=greater_is_better, kind=kind)
        return fn

    return decorator


# ── Loss-based metrics ────────────────────────────────────────────


@register("perplexity", greater_is_better=False, kind="loss")
def compute_perplexity(model, tokenizer, dataset) -> float | None:
    """Token-weighted perplexity over a tokenized dataset (lower is better)."""
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loader = DataLoader(dataset, batch_size=1)
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            if loss is None or torch.isnan(loss):
                continue
            n_tokens = int((labels != -100).sum().item())
            if n_tokens == 0:
                continue
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    if total_tokens == 0:
        return None
    avg_loss = total_loss / total_tokens
    return round(math.exp(min(avg_loss, 20)), 3)  # cap exponent to avoid inf


# ── Reference-based metrics (pure-Python, no heavy deps) ──────────


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


@register("rouge1", greater_is_better=True, kind="reference")
def compute_rouge1(predictions: list[str], references: list[str]) -> float | None:
    """Mean ROUGE-1 F1 over (prediction, reference) pairs (higher is better)."""
    if not predictions or len(predictions) != len(references):
        return None
    scores = []
    for pred, ref in zip(predictions, references, strict=False):
        pred_toks = _tokenize(pred)
        ref_toks = _tokenize(ref)
        if not pred_toks or not ref_toks:
            scores.append(0.0)
            continue
        ref_counts: dict[str, int] = {}
        for t in ref_toks:
            ref_counts[t] = ref_counts.get(t, 0) + 1
        overlap = 0
        seen: dict[str, int] = {}
        for t in pred_toks:
            seen[t] = seen.get(t, 0) + 1
            if seen[t] <= ref_counts.get(t, 0):
                overlap += 1
        precision = overlap / len(pred_toks)
        recall = overlap / len(ref_toks)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return round(sum(scores) / len(scores), 4)


@register("bleu", greater_is_better=True, kind="reference")
def compute_bleu(predictions: list[str], references: list[str]) -> float | None:
    """Corpus-level unigram BLEU with brevity penalty (higher is better).

    A deliberately small, dependency-free approximation — enough to track
    relative quality without pulling in sacrebleu/nltk.
    """
    if not predictions or len(predictions) != len(references):
        return None
    total_match = 0
    total_pred = 0
    total_ref_len = 0
    total_pred_len = 0
    for pred, ref in zip(predictions, references, strict=False):
        pred_toks = _tokenize(pred)
        ref_toks = _tokenize(ref)
        total_pred += len(pred_toks)
        total_pred_len += len(pred_toks)
        total_ref_len += len(ref_toks)
        ref_counts: dict[str, int] = {}
        for t in ref_toks:
            ref_counts[t] = ref_counts.get(t, 0) + 1
        seen: dict[str, int] = {}
        for t in pred_toks:
            seen[t] = seen.get(t, 0) + 1
            if seen[t] <= ref_counts.get(t, 0):
                total_match += 1
    if total_pred == 0:
        return None
    precision = total_match / total_pred
    # Brevity penalty
    if total_pred_len == 0:
        return 0.0
    bp = 1.0 if total_pred_len > total_ref_len else math.exp(1 - total_ref_len / total_pred_len)
    return round(bp * precision, 4)


def available_metrics() -> list[str]:
    return sorted(REGISTRY)
