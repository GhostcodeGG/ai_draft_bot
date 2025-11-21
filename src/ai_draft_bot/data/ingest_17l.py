"""Utilities to ingest 17L draft logs and card metadata exports.

The helpers here are intentionally lightweight to keep downstream modeling fast and
transparent. They accept common export formats from https://www.17lands.com/ and
return structured, typed objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, MutableMapping, Sequence

import pandas as pd


@dataclass
class CardMetadata:
    """Card attributes including win rate statistics from 17Lands.

    Win rate fields:
        gih_wr: Games in Hand Win Rate (win% when card is in hand)
        oh_wr: Opening Hand Win Rate (win% when card is in opening hand)
        gd_wr: Game Draw Win Rate (win% when card is drawn during game)
        iwd: Improvement When Drawn (percentage point improvement)
        alsa: Average Last Seen At (average pick number when card wheels)
    """

    name: str
    color: str
    rarity: str
    type_line: str
    mana_value: float
    # Win rate statistics (optional - may not be available for all cards)
    gih_wr: float | None = None
    oh_wr: float | None = None
    gd_wr: float | None = None
    iwd: float | None = None
    alsa: float | None = None


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
    consistent across sets. Win rate columns are optional and will be populated if available.

    Optional win rate columns (if present in CSV):
        - # GIH WR or gih_wr
        - # OH WR or oh_wr
        - # GD WR or gd_wr
        - IWD or iwd
        - ALSA or alsa
    """

    frame = pd.read_csv(path)
    required_columns = {"name", "color", "rarity", "type_line", "mana_value"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Card metadata missing required columns: {sorted(missing)}")

    # Map possible column name variations to our standard names
    winrate_column_map = {
        "# GIH WR": "gih_wr",
        "gih_wr": "gih_wr",
        "# OH WR": "oh_wr",
        "oh_wr": "oh_wr",
        "# GD WR": "gd_wr",
        "gd_wr": "gd_wr",
        "IWD": "iwd",
        "iwd": "iwd",
        "ALSA": "alsa",
        "alsa": "alsa",
    }

    # Detect which win rate columns are available
    available_wr_cols = {
        std_name: col_name
        for col_name, std_name in winrate_column_map.items()
        if col_name in frame.columns
    }

    metadata: MutableMapping[str, CardMetadata] = {}
    for row in frame.to_dict(orient="records"):
        # Extract win rate data if available
        wr_kwargs = {}
        for std_name, col_name in available_wr_cols.items():
            value = row.get(col_name)
            if pd.notna(value):
                wr_kwargs[std_name] = float(value)

        metadata[row["name"]] = CardMetadata(
            name=row["name"],
            color=row["color"],
            rarity=row["rarity"],
            type_line=row["type_line"],
            mana_value=float(row["mana_value"]),
            **wr_kwargs,
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
