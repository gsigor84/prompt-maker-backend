# run_store.py
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import Optional, Dict, Any


class RunStore:
    """
    Very small JSONL store.
    - save(dict): appends one JSON object to data/runs.jsonl
    - suggest_direction_name(task): returns most recent chosen_direction_name for that task (if any)
    """

    def __init__(self, path: str = "data/runs.jsonl"):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def save(self, record: Dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("ts", datetime.utcnow().isoformat())
        line = json.dumps(record, ensure_ascii=False)

        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def suggest_direction_name(self, task: str) -> Optional[str]:
        if not os.path.exists(self.path):
            return None

        # scan backwards (cheap enough for small local file)
        with self._lock:
            try:
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
