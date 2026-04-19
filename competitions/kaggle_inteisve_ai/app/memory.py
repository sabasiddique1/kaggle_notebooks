"""Memory management: session state and long-term memory."""
import os
import json
from typing import Optional
from app.models import MemoryBank
from app.observability import log_event

MEM_PATH = "memory_bank.json"


def load_memory() -> MemoryBank:
    """Load memory bank from JSON file."""
    if not os.path.exists(MEM_PATH):
        return MemoryBank()
    try:
        with open(MEM_PATH, "r", encoding="utf-8") as f:
            return MemoryBank(**json.load(f))
    except Exception:
        return MemoryBank()


def save_memory(mem: MemoryBank) -> None:
    """Save memory bank to JSON file."""
    with open(MEM_PATH, "w", encoding="utf-8") as f:
        json.dump(mem.model_dump(), f, indent=2)






