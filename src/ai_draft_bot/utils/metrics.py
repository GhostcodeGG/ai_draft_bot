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
    """Compute NDCG@k for multiclass predictions."""
    if probs.size == 0 or labels.size == 0:
        return {k: 0.0 for k in ks}

    sorted_indices = np.argsort(probs, axis=1)[:, ::-1]
    results: dict[int, float] = {}

    for k in ks:
        dcg = 0.0
        for label, ranking in zip(labels, sorted_indices):
            # Find rank of the true label
            try:
                rank = int(np.where(ranking[:k] == label)[0][0])
                dcg += 1.0 / np.log2(rank + 2)
            except IndexError:
                continue  # Not in top-k
        results[k] = dcg / len(labels)
    return results
