"""JSON-lines timing log. Values are whatever the caller measured."""

from __future__ import annotations

import json
import time
from pathlib import Path


class MetricsLogger:
    """Append one JSON object per record. An empty path disables logging."""

    def __init__(self, path: str) -> None:
        self.path = Path(path) if path else None

    def record(self, stage: str, **fields: float | int | str) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, float | int | str] = {
            "stage": stage,
            "wall_time": time.time(),
        }
        payload.update(fields)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
