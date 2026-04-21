from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ResearchExperimentTracker:
    frame_dir: Optional[str] = None
    started_at: Optional[str] = None
    last_resumed_at: Optional[str] = None
    elapsed_ms: int = 0
    is_paused: bool = True
    is_hidden: bool = False
    events: List[Dict[str, Any]] = field(default_factory=list)
    _running_since_monotonic: Optional[float] = field(default=None, init=False, repr=False)

    def start_new(self, frame_dir: Path, start_paused: bool = False) -> None:
        now_iso = _utc_now_iso()
        self.frame_dir = str(frame_dir)
        self.started_at = now_iso
        self.last_resumed_at = None if start_paused else now_iso
        self.elapsed_ms = 0
        self.is_paused = bool(start_paused)
        self.events = []
        self._running_since_monotonic = None if start_paused else monotonic()

    def load(self, data: Dict[str, Any], default_frame_dir: Optional[Path] = None) -> None:
        self.frame_dir = str(data.get("frame_dir") or (str(default_frame_dir) if default_frame_dir else ""))
        self.started_at = data.get("started_at") or _utc_now_iso()
        self.last_resumed_at = data.get("last_resumed_at")
        self.elapsed_ms = int(data.get("elapsed_ms", 0))
        self.is_paused = bool(data.get("is_paused", True))
        self.is_hidden = bool(data.get("is_hidden", False))
        raw_events = data.get("events", [])
        self.events = [dict(event) for event in raw_events if isinstance(event, dict)]
        self._running_since_monotonic = None

    def pause(self) -> None:
        if self.is_paused:
            return
        self.elapsed_ms = self.current_elapsed_ms()
        self.is_paused = True
        self._running_since_monotonic = None

    def resume(self) -> None:
        if not self.is_paused:
            return
        self.is_paused = False
        self.last_resumed_at = _utc_now_iso()
        self._running_since_monotonic = monotonic()

    def current_elapsed_ms(self) -> int:
        if self.is_paused or self._running_since_monotonic is None:
            return int(self.elapsed_ms)
        delta_ms = int(max(0.0, monotonic() - self._running_since_monotonic) * 1000.0)
        return int(self.elapsed_ms + delta_ms)

    def set_hidden(self, hidden: bool) -> None:
        self.is_hidden = bool(hidden)

    def record_event(self, event: Dict[str, Any]) -> None:
        entry = dict(event)
        entry["event_index"] = len(self.events) + 1
        entry.setdefault("timestamp_iso", _utc_now_iso())
        entry.setdefault("elapsed_ms", self.current_elapsed_ms())
        self.events.append(entry)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "enabled": True,
            "frame_dir": self.frame_dir,
            "started_at": self.started_at,
            "last_resumed_at": self.last_resumed_at,
            "elapsed_ms": self.current_elapsed_ms(),
            "is_paused": bool(self.is_paused),
            "is_hidden": bool(self.is_hidden),
            "event_count": len(self.events),
            "events": [dict(event) for event in self.events],
        }
