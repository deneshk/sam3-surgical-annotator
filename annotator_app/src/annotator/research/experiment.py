"""Research session tracking independent of Qt widgets.

The tracker owns elapsed-time bookkeeping and the persisted event log for one
annotator session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Dict, List, Optional

from annotator.research.models import MousePositionEvent, ResearchEvent


def _utc_now_iso() -> str:
    """Return a UTC timestamp string suitable for persisted telemetry."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ResearchExperimentTracker:
    """In-memory representation of one research telemetry session."""

    frame_dir: Optional[str] = None
    started_at: Optional[str] = None
    last_resumed_at: Optional[str] = None
    elapsed_ms: int = 0
    is_paused: bool = True
    is_hidden: bool = False
    events: List[ResearchEvent] = field(default_factory=list)
    _running_since_monotonic: Optional[float] = field(default=None, init=False, repr=False)

    def start_new(self, frame_dir: Path, start_paused: bool = False) -> None:
        """Start a fresh telemetry session for the loaded frame directory."""
        now_iso = _utc_now_iso()
        self.frame_dir = str(frame_dir)
        self.started_at = now_iso
        self.last_resumed_at = None if start_paused else now_iso
        self.elapsed_ms = 0
        self.is_paused = bool(start_paused)
        self.events = []
        self._running_since_monotonic = None if start_paused else monotonic()

    def load(self, data: Dict[str, Any], default_frame_dir: Optional[Path] = None) -> None:
        """Restore tracker state from persisted JSON data."""
        self.frame_dir = str(data.get("frame_dir") or (str(default_frame_dir) if default_frame_dir else ""))
        self.started_at = data.get("started_at") or _utc_now_iso()
        self.last_resumed_at = data.get("last_resumed_at")
        self.elapsed_ms = int(data.get("elapsed_ms", 0))
        self.is_paused = bool(data.get("is_paused", True))
        self.is_hidden = bool(data.get("is_hidden", False))
        raw_events = data.get("events", [])
        self.events = [ResearchEvent.from_dict(event) for event in raw_events if isinstance(event, dict)]
        self._running_since_monotonic = None

    def pause(self) -> None:
        """Freeze elapsed-time accumulation while keeping the session active."""
        if self.is_paused:
            return
        self.elapsed_ms = self.current_elapsed_ms()
        self.is_paused = True
        self._running_since_monotonic = None

    def resume(self) -> None:
        """Resume elapsed-time accumulation after a pause."""
        if not self.is_paused:
            return
        self.is_paused = False
        self.last_resumed_at = _utc_now_iso()
        self._running_since_monotonic = monotonic()

    def current_elapsed_ms(self) -> int:
        """Compute elapsed milliseconds including the currently running span."""
        if self.is_paused or self._running_since_monotonic is None:
            return int(self.elapsed_ms)
        delta_ms = int(max(0.0, monotonic() - self._running_since_monotonic) * 1000.0)
        return int(self.elapsed_ms + delta_ms)

    def set_hidden(self, hidden: bool) -> None:
        """Persist whether the timer display should be hidden in the UI."""
        self.is_hidden = bool(hidden)

    def record_event(self, event: ResearchEvent) -> None:
        """Append one telemetry event with derived sequence and timing fields."""
        entry = ResearchEvent.from_dict(event.to_dict())
        entry.event_index = len(self.events) + 1
        if entry.timestamp_iso is None:
            entry.timestamp_iso = _utc_now_iso()
        if entry.elapsed_ms is None:
            entry.elapsed_ms = self.current_elapsed_ms()
        self.events.append(entry)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the tracker into the persisted research session schema."""
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
            "events": [event.to_dict() for event in self.events],
        }


@dataclass
class ResearchMousePositionTracker:
    """In-memory representation of sampled research mouse-position telemetry."""

    frame_dir: Optional[str] = None
    started_at: Optional[str] = None
    positions: List[MousePositionEvent] = field(default_factory=list)

    def start_new(self, frame_dir: Path, started_at: Optional[str]) -> None:
        """Start a fresh mouse-position log for the loaded frame directory."""
        self.frame_dir = str(frame_dir)
        self.started_at = started_at
        self.positions = []

    def load(
        self,
        data: Dict[str, Any],
        *,
        default_frame_dir: Optional[Path] = None,
        default_started_at: Optional[str] = None,
    ) -> None:
        """Restore sampled mouse positions from persisted JSON data."""
        self.frame_dir = str(data.get("frame_dir") or (str(default_frame_dir) if default_frame_dir else ""))
        self.started_at = data.get("started_at") or default_started_at
        raw_positions = data.get("positions", [])
        self.positions = [
            MousePositionEvent.from_dict(position)
            for position in raw_positions
            if isinstance(position, dict)
        ]

    def record_position(
        self,
        position: MousePositionEvent,
        *,
        elapsed_ms: int,
    ) -> None:
        """Append one sampled position with derived sequence and timing fields."""
        entry = MousePositionEvent.from_dict(position.to_dict())
        entry.event_index = len(self.positions) + 1
        if entry.timestamp_iso is None:
            entry.timestamp_iso = _utc_now_iso()
        if entry.elapsed_ms is None:
            entry.elapsed_ms = int(elapsed_ms)
        self.positions.append(entry)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the mouse-position tracker into its independent file schema."""
        return {
            "version": 1,
            "enabled": True,
            "frame_dir": self.frame_dir,
            "started_at": self.started_at,
            "position_count": len(self.positions),
            "positions": [position.to_dict() for position in self.positions],
        }
