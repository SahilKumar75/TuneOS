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
    """Corpus-level BLEU via the HuggingFace evaluate library (sacrebleu)."""
    if not predictions or len(predictions) != len(references):
        return None
    try:
        import evaluate as _evaluate

        metric = _evaluate.load("bleu")
        result = metric.compute(predictions=predictions, references=[[r] for r in references])
        return round(float(result["bleu"]), 4) if result else None
    except Exception:
        return None


def _f1(overlap: int, n_pred: int, n_ref: int) -> float:
    if n_pred == 0 or n_ref == 0:
        return 0.0
    precision = overlap / n_pred
    recall = overlap / n_ref
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _ngram_overlap(pred: list[str], ref: list[str], n: int) -> tuple[int, int, int]:
    def ngrams(toks: list[str]) -> list[tuple]:
        return [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]

    p, r = ngrams(pred), ngrams(ref)
    ref_counts: dict[tuple, int] = {}
    for g in r:
        ref_counts[g] = ref_counts.get(g, 0) + 1
    overlap = 0
    seen: dict[tuple, int] = {}
    for g in p:
        seen[g] = seen.get(g, 0) + 1
        if seen[g] <= ref_counts.get(g, 0):
            overlap += 1
    return overlap, len(p), len(r)


@register("rouge2", greater_is_better=True, kind="reference")
def compute_rouge2(predictions: list[str], references: list[str]) -> float | None:
    """Mean ROUGE-2 (bigram) F1 over (prediction, reference) pairs."""
    if not predictions or len(predictions) != len(references):
        return None
    scores = []
    for pred, ref in zip(predictions, references, strict=False):
        overlap, np_, nr = _ngram_overlap(_tokenize(pred), _tokenize(ref), 2)
        scores.append(_f1(overlap, np_, nr))
    return round(sum(scores) / len(scores), 4)


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if x == y else max(prev[j], curr[j - 1])
        prev = curr
    return prev[len(b)]


@register("rougeL", greater_is_better=True, kind="reference")
def compute_rougeL(predictions: list[str], references: list[str]) -> float | None:
    """Mean ROUGE-L (longest-common-subsequence) F1 over pairs."""
    if not predictions or len(predictions) != len(references):
        return None
    scores = []
    for pred, ref in zip(predictions, references, strict=False):
        pt, rt = _tokenize(pred), _tokenize(ref)
        scores.append(_f1(_lcs_len(pt, rt), len(pt), len(rt)))
    return round(sum(scores) / len(scores), 4)


@register("meteor", greater_is_better=True, kind="reference")
def compute_meteor(predictions: list[str], references: list[str]) -> float | None:
    """Corpus-level METEOR via the HuggingFace evaluate library."""
    if not predictions or len(predictions) != len(references):
        return None
    try:
        import evaluate as _evaluate

        metric = _evaluate.load("meteor")
        result = metric.compute(predictions=predictions, references=references)
        return round(float(result["meteor"]), 4) if result else None
    except Exception:
        return None


def available_metrics() -> list[str]:
    return sorted(REGISTRY)
