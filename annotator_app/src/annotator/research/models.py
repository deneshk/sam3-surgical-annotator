"""Typed research telemetry payloads.

These records keep research tracking strongly typed internally and leave raw
dicts only at the JSON boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CanvasContext:
    """Canvas-specific metadata attached to research click events."""

    gesture_kind: str
    prompt_mode_index: int
    active_object_id: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["CanvasContext"]:
        """Parse optional persisted canvas metadata into a typed record."""
        if not isinstance(data, dict):
            return None
        return cls(
            gesture_kind=str(data.get("gesture_kind", "")),
            prompt_mode_index=int(data.get("prompt_mode_index", 0)),
            active_object_id=(
                int(data["active_object_id"]) if data.get("active_object_id") is not None else None
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert canvas metadata back to a JSON-safe mapping."""
        return {
            "gesture_kind": self.gesture_kind,
            "prompt_mode_index": int(self.prompt_mode_index),
            "active_object_id": int(self.active_object_id) if self.active_object_id is not None else None,
        }


@dataclass
class ResearchEvent:
    """One research telemetry event recorded during annotation."""

    frame_idx: Optional[int]
    target_type: str
    target_name: str
    widget_class: str
    canvas_x_px: Optional[int] = None
    canvas_y_px: Optional[int] = None
    canvas_context: Optional[CanvasContext] = None
    event_index: Optional[int] = None
    timestamp_iso: Optional[str] = None
    elapsed_ms: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchEvent":
        """Parse a persisted event payload into the internal typed representation."""
        return cls(
            frame_idx=int(data["frame_idx"]) if data.get("frame_idx") is not None else None,
            target_type=str(data.get("target_type", "")),
            target_name=str(data.get("target_name", "")),
            widget_class=str(data.get("widget_class", "")),
            canvas_x_px=int(data["canvas_x_px"]) if data.get("canvas_x_px") is not None else None,
            canvas_y_px=int(data["canvas_y_px"]) if data.get("canvas_y_px") is not None else None,
            canvas_context=CanvasContext.from_dict(data.get("canvas_context")),
            event_index=int(data["event_index"]) if data.get("event_index") is not None else None,
            timestamp_iso=str(data["timestamp_iso"]) if data.get("timestamp_iso") else None,
            elapsed_ms=int(data["elapsed_ms"]) if data.get("elapsed_ms") is not None else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the typed event into a JSON-safe payload."""
        return {
            "frame_idx": int(self.frame_idx) if self.frame_idx is not None else None,
            "target_type": self.target_type,
            "target_name": self.target_name,
            "widget_class": self.widget_class,
            "canvas_x_px": int(self.canvas_x_px) if self.canvas_x_px is not None else None,
            "canvas_y_px": int(self.canvas_y_px) if self.canvas_y_px is not None else None,
            "canvas_context": self.canvas_context.to_dict() if self.canvas_context is not None else None,
            "event_index": int(self.event_index) if self.event_index is not None else None,
            "timestamp_iso": self.timestamp_iso,
            "elapsed_ms": int(self.elapsed_ms) if self.elapsed_ms is not None else None,
        }


@dataclass
class ResearchWidgetTarget:
    """Normalized widget classification used for telemetry naming."""

    target_type: str
    target_name: str


@dataclass
class MousePositionEvent:
    """One sampled mouse-position telemetry record."""

    frame_idx: Optional[int]
    target_type: str
    target_name: str
    widget_class: str
    window_x_px: Optional[int] = None
    window_y_px: Optional[int] = None
    widget_x_px: Optional[int] = None
    widget_y_px: Optional[int] = None
    canvas_x_px: Optional[int] = None
    canvas_y_px: Optional[int] = None
    event_index: Optional[int] = None
    timestamp_iso: Optional[str] = None
    elapsed_ms: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MousePositionEvent":
        """Parse a persisted mouse-position payload into the typed representation."""
        return cls(
            frame_idx=int(data["frame_idx"]) if data.get("frame_idx") is not None else None,
            target_type=str(data.get("target_type", "")),
            target_name=str(data.get("target_name", "")),
            widget_class=str(data.get("widget_class", "")),
            window_x_px=int(data["window_x_px"]) if data.get("window_x_px") is not None else None,
            window_y_px=int(data["window_y_px"]) if data.get("window_y_px") is not None else None,
            widget_x_px=int(data["widget_x_px"]) if data.get("widget_x_px") is not None else None,
            widget_y_px=int(data["widget_y_px"]) if data.get("widget_y_px") is not None else None,
            canvas_x_px=int(data["canvas_x_px"]) if data.get("canvas_x_px") is not None else None,
            canvas_y_px=int(data["canvas_y_px"]) if data.get("canvas_y_px") is not None else None,
            event_index=int(data["event_index"]) if data.get("event_index") is not None else None,
            timestamp_iso=str(data["timestamp_iso"]) if data.get("timestamp_iso") else None,
            elapsed_ms=int(data["elapsed_ms"]) if data.get("elapsed_ms") is not None else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the typed mouse-position event into a JSON-safe payload."""
        return {
            "frame_idx": int(self.frame_idx) if self.frame_idx is not None else None,
            "target_type": self.target_type,
            "target_name": self.target_name,
            "widget_class": self.widget_class,
            "window_x_px": int(self.window_x_px) if self.window_x_px is not None else None,
            "window_y_px": int(self.window_y_px) if self.window_y_px is not None else None,
            "widget_x_px": int(self.widget_x_px) if self.widget_x_px is not None else None,
            "widget_y_px": int(self.widget_y_px) if self.widget_y_px is not None else None,
            "canvas_x_px": int(self.canvas_x_px) if self.canvas_x_px is not None else None,
            "canvas_y_px": int(self.canvas_y_px) if self.canvas_y_px is not None else None,
            "event_index": int(self.event_index) if self.event_index is not None else None,
            "timestamp_iso": self.timestamp_iso,
            "elapsed_ms": int(self.elapsed_ms) if self.elapsed_ms is not None else None,
        }
