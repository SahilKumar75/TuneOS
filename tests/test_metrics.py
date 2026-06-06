"""Unit tests for the pluggable metric registry (reference-based metrics).

These run without GPU/model — they exercise the pure-Python scoring functions.
"""

from trainer.metrics import (
    REGISTRY,
    available_metrics,
    compute_bleu,
    compute_meteor,
    compute_rouge1,
    compute_rouge2,
    compute_rougeL,
)


def test_registry_contains_expected_metrics():
    assert "perplexity" in REGISTRY
    assert "rouge1" in REGISTRY
    assert "bleu" in REGISTRY
    assert available_metrics() == sorted(REGISTRY)


def test_metric_metadata_directions():
    assert REGISTRY["perplexity"].greater_is_better is False
    assert REGISTRY["rouge1"].greater_is_better is True
    assert REGISTRY["bleu"].greater_is_better is True
    assert REGISTRY["perplexity"].kind == "loss"
    assert REGISTRY["rouge1"].kind == "reference"


def test_rouge1_perfect_match_is_one():
    assert compute_rouge1(["the cat sat"], ["the cat sat"]) == 1.0


def test_rouge1_no_overlap_is_zero():
    assert compute_rouge1(["alpha beta"], ["gamma delta"]) == 0.0


def test_rouge1_partial_overlap_between_zero_and_one():
    score = compute_rouge1(["the cat sat on mat"], ["the cat ran"])
    assert 0.0 < score < 1.0


def test_rouge1_mismatched_lengths_returns_none():
    assert compute_rouge1(["a"], ["a", "b"]) is None
    assert compute_rouge1([], []) is None


def test_bleu_perfect_match_is_one():
    assert compute_bleu(["the cat sat"], ["the cat sat"]) == 1.0


def test_bleu_brevity_penalty_punishes_short_predictions():
    # Prediction much shorter than reference → BLEU < 1 even with full precision.
    score = compute_bleu(["the"], ["the cat sat on the mat"])
    assert 0.0 < score < 1.0


def test_bleu_mismatched_lengths_returns_none():
    assert compute_bleu(["a"], []) is None


def test_new_metrics_registered():
    for name in ("rouge2", "rougeL", "meteor"):
        assert name in REGISTRY
        assert REGISTRY[name].kind == "reference"
        assert REGISTRY[name].greater_is_better is True


def test_rouge2_perfect_and_zero():
    assert compute_rouge2(["the cat sat"], ["the cat sat"]) == 1.0
    # No shared bigram → 0.
    assert compute_rouge2(["the cat"], ["a dog"]) == 0.0


def test_rougel_perfect_and_subsequence():
    assert compute_rougeL(["the cat sat"], ["the cat sat"]) == 1.0
    # LCS "the cat" of a longer reference → between 0 and 1.
    score = compute_rougeL(["the cat"], ["the big cat sat"])
    assert 0.0 < score < 1.0


def test_meteor_perfect_and_partial():
    assert compute_meteor(["the cat sat"], ["the cat sat"]) > 0.9
    assert compute_meteor(["alpha beta"], ["gamma delta"]) == 0.0


def test_new_metrics_mismatched_lengths_return_none():
    assert compute_rouge2(["a"], []) is None
    assert compute_rougeL(["a"], []) is None
    assert compute_meteor(["a"], []) is None
