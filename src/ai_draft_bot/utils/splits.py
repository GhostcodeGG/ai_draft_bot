"""Dataset splitting utilities (draft-aware)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class TrainValSplit:
    """Simple container for train/validation rows."""

    train: List[object]
    val: List[object]


def train_val_split_by_event(
    rows: Sequence[object], test_size: float = 0.2, random_state: int = 13
) -> TrainValSplit:
    """Split rows into train/val sets grouped by event_id to avoid leakage.

    This assumes each row has an ``event_id`` attribute (or None). Rows without an
    event_id are kept in the training set to avoid contaminating the validation set.
    """
    if not rows:
        return TrainValSplit(train=[], val=[])

    event_to_rows: dict[str, List[object]] = {}
    fallback_train: List[object] = []

    for row in rows:
        event_id = getattr(row, "event_id", None)
        if event_id:
            event_to_rows.setdefault(str(event_id), []).append(row)
        else:
            fallback_train.append(row)

    event_ids = list(event_to_rows.keys())
    rng = np.random.default_rng(random_state)
    rng.shuffle(event_ids)

    if not event_ids:
        # No event ids present; fall back to a simple split
        split_idx = max(1, int(len(rows) * (1 - test_size)))
        return TrainValSplit(train=list(rows[:split_idx]), val=list(rows[split_idx:]))

    if len(event_ids) == 1:
        return TrainValSplit(train=list(rows), val=[])

    split_idx = int(len(event_ids) * (1 - test_size))
    if split_idx < 1:
        split_idx = 1  # ensure we always have some training events
    if split_idx >= len(event_ids):
        split_idx = len(event_ids) - 1  # ensure we always have some validation events

    train_events = set(event_ids[:split_idx])
    val_events = set(event_ids[split_idx:])

    train_rows: List[object] = []
    val_rows: List[object] = []

    for event_id, event_rows in event_to_rows.items():
        if event_id in train_events:
            train_rows.extend(event_rows)
        else:
            val_rows.extend(event_rows)

    train_rows.extend(fallback_train)

    if not val_rows:
        # Fall back to a row-level split (may introduce leakage but avoids crashes on tiny datasets)
        indices = np.arange(len(rows))
        rng.shuffle(indices)
        split_idx_rows = max(1, int(len(rows) * (1 - test_size)))
        train_idx = indices[:split_idx_rows]
        val_idx = indices[split_idx_rows:]
        train_rows = [rows[i] for i in train_idx]
        val_rows = [rows[i] for i in val_idx]

    return TrainValSplit(train=train_rows, val=val_rows)
