"""One JSON object per line on stdout for log agents (e.g. ECS awslogs, Fluent Bit → CloudWatch)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone


def log_event(level: str, **fields: object) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "level": level,
        **fields,
    }
    sys.stdout.write(json.dumps(payload, default=str, separators=(",", ":")) + "\n")
    sys.stdout.flush()
