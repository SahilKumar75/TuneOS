"""Hyperparameter sweep helpers.

Expands a parameter grid over a base config into concrete per-run configs. The
resulting runs are submitted as ordinary fine-tune jobs and visualized together
on the /compare page — that pairing is the hyperparameter-tuning dashboard.
"""

from __future__ import annotations

import itertools
from typing import Any


def expand_grid(base: dict[str, Any], grid: dict[str, list]) -> list[dict[str, Any]]:
    """Return one merged config per combination of ``grid`` overlaid on ``base``.

    ``grid`` maps a field name to the list of values to try; the result is the
    cartesian product. An empty grid yields a single copy of ``base``.

    >>> expand_grid({"epochs": 3}, {"learning_rate": [1e-4, 2e-4]})
    [{'epochs': 3, 'learning_rate': 0.0001}, {'epochs': 3, 'learning_rate': 0.0002}]
    """
    if not grid:
        return [dict(base)]
    keys = list(grid)
    out: list[dict[str, Any]] = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        cfg = dict(base)
        cfg.update(dict(zip(keys, combo, strict=False)))
        out.append(cfg)
    return out
