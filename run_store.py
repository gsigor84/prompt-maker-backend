# run_store.py
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, Any


class RunStore:
    """
    Very small JSONL store.
    - save(dict): appends one JSON object to data/runs.jsonl
    - suggest_direction_name(task): returns most recent chosen_direction_name for that task (if any)

    Notes:
    - This is great for local dev.
    - On many hosts, filesystem may be ephemeral unless you attach a persistent disk.
    """

    def __init__(self, path: str = "data/runs.jsonl"):
        self.path = path
        self._lock = threading.Lock()

        # Create directory only if one exists in the path (avoid dirname == "")
        dir_name = os.path.dirname(self.path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

    def save(self, record: Dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("ts", datetime.now(timezone.utc).isoformat())

        line = json.dumps(record, ensure_ascii=False)

        with self._lock:
            # Ensure directory exists at write-time too (covers path changes)
            dir_name = os.path.dirname(self.path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def suggest_direction_name(self, task: str) -> Optional[str]:
        if not os.path.exists(self.path):
            return None

        with self._lock:
            try:
                # Read file lines, scan backwards. Still simple.
                with open(self.path, "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()
            except Exception:
                return None

        for line in reversed(lines):
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if obj.get("task") == task and obj.get("chosen_direction_name"):
                return str(obj["chosen_direction_name"])

        return None
