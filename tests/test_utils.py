import numpy as np

from ai_draft_bot.features.draft_context import PickFeatures
from ai_draft_bot.utils.metrics import ndcg_at_ks, topk_accuracies
from ai_draft_bot.utils.splits import train_val_split_by_event


def test_train_val_split_keeps_events_together() -> None:
    rows = []
    for event_id in ["A", "A", "B", "B"]:
        rows.append(
            PickFeatures(features=np.array([1.0]), label=f"{event_id}_card", event_id=event_id)
        )

    split = train_val_split_by_event(rows, test_size=0.5, random_state=0)

    assert split.train and split.val
    train_events = {row.event_id for row in split.train}
    val_events = {row.event_id for row in split.val}

    assert train_events.isdisjoint(val_events)


def test_topk_and_ndcg_metrics() -> None:
    probs = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
    labels = np.array([1, 0])

    topk = topk_accuracies(probs, labels, ks=(1, 2))
    assert np.isclose(topk[1], 1.0)
    assert np.isclose(topk[2], 1.0)

    ndcg = ndcg_at_ks(probs, labels, ks=(1, 2))
    assert ndcg[1] > 0.9
    assert ndcg[2] > 0.9
