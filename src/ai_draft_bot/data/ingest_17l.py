"""Utilities to ingest 17L draft logs and card metadata exports.

The helpers here are intentionally lightweight to keep downstream modeling fast and
transparent. They accept common export formats from https://www.17lands.com/ and
return structured, typed objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, MutableMapping, Sequence

import json
import pandas as pd


@dataclass
class CardMetadata:
    """Basic card attributes used for downstream feature extraction."""

    name: str
    color: str
    rarity: str
    type_line: str
    mana_value: float


@dataclass
class PickRecord:
    """A single pick within a draft log."""

    event_id: str
    pack_number: int
    pick_number: int
    chosen_card: str
    pack_contents: List[str]


def load_jsonl(path: Path | str) -> Iterator[Mapping[str, object]]:
    """Stream JSONL entries to avoid loading large files at once.

    17L exports can easily exceed in-memory limits when concatenated across
    many drafts. Streaming the file keeps ingestion fast and resilient.
    """

    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def parse_card_metadata(path: Path | str) -> Mapping[str, CardMetadata]:
    """Parse a 17L card metadata CSV file keyed by card name.

    The metadata export schema is relatively stable: we only rely on fields that are
    consistent across sets.
    """

    frame = pd.read_csv(path)
    required_columns = {"name", "color", "rarity", "type_line", "mana_value"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Card metadata missing required columns: {sorted(missing)}")

    metadata: MutableMapping[str, CardMetadata] = {}
    for row in frame.to_dict(orient="records"):
        metadata[row["name"]] = CardMetadata(
            name=row["name"],
            color=row["color"],
            rarity=row["rarity"],
            type_line=row["type_line"],
            mana_value=float(row["mana_value"]),
        )
    return metadata


def parse_draft_logs(path: Path | str, *, allow_partial: bool = True) -> List[PickRecord]:
    """Parse 17L draft JSONL logs into a list of :class:`PickRecord`.

    Args:
        path: Path to the JSONL export from 17L.
        allow_partial: Whether to keep drafts that end early (e.g., disconnected users).

    Returns:
        A list of pick records across all drafts in the file.
    """

    picks: List[PickRecord] = []
    for entry in load_jsonl(path):
        event_id = str(entry.get("event_id", "unknown"))
        packs = entry.get("pack_number")
        pick_numbers = entry.get("pick_number")
        cards_in_pack = entry.get("cards_in_pack")
        picked_cards = entry.get("picked_cards")

        if not isinstance(packs, Iterable) or not isinstance(pick_numbers, Iterable):
            if allow_partial:
                continue
            msg = "Draft log missing pack/pick arrays and allow_partial=False"
            raise ValueError(msg)

        for pack, pick, pack_contents, chosen in zip(
            packs or [], pick_numbers or [], cards_in_pack or [], picked_cards or []
        ):
            if not pack_contents:
                continue
            picks.append(
                PickRecord(
                    event_id=event_id,
                    pack_number=int(pack),
                    pick_number=int(pick),
                    chosen_card=str(chosen),
                    pack_contents=list(pack_contents),
                )
            )
    return picks


def group_picks_by_event(picks: Sequence[PickRecord]) -> Mapping[str, List[PickRecord]]:
    """Group picks by event ID for convenience."""

    grouped: MutableMapping[str, List[PickRecord]] = {}
    for pick in picks:
        grouped.setdefault(pick.event_id, []).append(pick)
    for pick_list in grouped.values():
        pick_list.sort(key=lambda record: (record.pack_number, record.pick_number))
    return grouped
