from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from etf_recommender.config import CACHE_DIR


NEWS_CACHE_FILE = CACHE_DIR / "news.json"
RECOMMENDATIONS_CACHE_FILE = CACHE_DIR / "recommendations.json"
STATE_CACHE_FILE = CACHE_DIR / "state.json"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def load_news_cache() -> dict[str, Any] | None:
    return read_json(NEWS_CACHE_FILE, None)


def load_recommendations_cache() -> dict[str, Any] | None:
    return read_json(RECOMMENDATIONS_CACHE_FILE, None)


def load_state() -> dict[str, Any]:
    return read_json(STATE_CACHE_FILE, {"completed_slots": [], "failed_slots": {}})


def save_state(state: dict[str, Any]) -> None:
    write_json(STATE_CACHE_FILE, state)


def record_completed_slot(slot_key: str) -> None:
    state = load_state()
    completed = set(state.get("completed_slots", []))
    completed.add(slot_key)
    state["completed_slots"] = sorted(completed)
    failed = state.get("failed_slots", {})
    failed.pop(slot_key, None)
    state["failed_slots"] = failed
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    save_state(state)


def record_failed_slot(slot_key: str, error: str) -> None:
    state = load_state()
    failed = state.get("failed_slots", {})
    failed[slot_key] = {
        "error": error,
        "failed_at": datetime.now().isoformat(timespec="seconds"),
    }
    state["failed_slots"] = failed
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    save_state(state)

