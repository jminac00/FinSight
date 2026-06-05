"""Parser for the FinEntity JSON dataset."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Annotation:
    start: int
    end: int
    value: str            # entity name (e.g. "Johnson & Johnson")
    sentiment_label: str  # "Positive" | "Negative" | "Neutral"


@dataclass
class FinEntityRecord:
    text: str
    annotations: list[Annotation] = field(default_factory=list)


def load(path: Path) -> list[FinEntityRecord]:
    """Load and parse the FinEntity JSON dataset."""
    raw: list[dict] = json.loads(path.read_text(encoding="utf-8"))
    records: list[FinEntityRecord] = []
    for entry in raw:
        annotations = [
            Annotation(
                start=a["start"],
                end=a["end"],
                value=a["value"],
                sentiment_label=a.get("label") or a.get("tag") or "Neutral",
            )
            for a in entry.get("annotations", [])
        ]
        records.append(FinEntityRecord(text=entry["content"], annotations=annotations))
    logger.info("FinEntity: loaded %d records from %s", len(records), path.name)
    return records
