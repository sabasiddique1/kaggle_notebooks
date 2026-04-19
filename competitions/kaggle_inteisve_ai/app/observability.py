"""Observability: logging, tracing, metrics."""
import uuid
import datetime
from typing import Dict, Any


TRACE_ID = str(uuid.uuid4())
METRICS: Dict[str, int] = {"tool_calls": 0, "llm_calls": 0, "errors": 0}


def now_iso() -> str:
    """Get current time in ISO format."""
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def log_event(event: str, **fields: Any) -> Dict[str, Any]:
    """Log a structured event."""
    payload = {"ts": now_iso(), "trace_id": TRACE_ID, "event": event, **fields}
    print(f"[LOG] {payload}")  # Simple print for now
    return payload






