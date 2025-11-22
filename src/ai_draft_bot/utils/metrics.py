"""Shared evaluation helpers."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np


def topk_accuracies(probs: np.ndarray, labels: np.ndarray, ks: Sequence[int]) -> dict[int, float]:
    """Compute top-k accuracies given probability matrix and integer labels."""
    if probs.size == 0 or labels.size == 0:
        return {k: 0.0 for k in ks}

    sorted_indices = np.argsort(probs, axis=1)[:, ::-1]
    results: dict[int, float] = {}
    for k in ks:
        topk = sorted_indices[:, :k]
        hits = sum(int(label in topk_row) for label, topk_row in zip(labels, topk))
        results[k] = hits / len(labels)
    return results


def ndcg_at_ks(probs: np.ndarray, labels: np.ndarray, ks: Sequence[int]) -> dict[int, float]:
    """Compute NDCG@k for multiclass predictions with a single relevant label.

    We normalize per-row using the ideal DCG (which is 1.0 for a single relevant
    item at rank 1) so the metric is always in [0, 1].
    """
    if probs.size == 0 or labels.size == 0:
        return {k: 0.0 for k in ks}

    sorted_indices = np.argsort(probs, axis=1)[:, ::-1]
    results: dict[int, float] = {}

    for k in ks:
        per_row_scores = []
        for label, ranking in zip(labels, sorted_indices):
            # Rank is 0-indexed; add 2 to get 1-based + log2 denominator shift
            try:
                rank = int(np.where(ranking[:k] == label)[0][0])
                per_row_scores.append(1.0 / np.log2(rank + 2))
            except IndexError:
                per_row_scores.append(0.0)

        # IDCG for a single relevant item is 1.0, so the score is already normalized
        results[k] = float(np.mean(per_row_scores))

    return results
