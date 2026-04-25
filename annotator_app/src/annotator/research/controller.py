"""Qt-facing controller for optional research telemetry.

The main window delegates research-only behavior here so telemetry remains
isolated from normal annotation flow.
"""

from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import Optional, Tuple

from PySide6.QtCore import QObject, QPoint, QEvent, Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QPushButton,
    QSpinBox,
    QToolButton,
    QWidget,
)

from annotator.persistence.session_io import read_session_json, write_session_json
from annotator.research.experiment import ResearchExperimentTracker, ResearchMousePositionTracker
from annotator.research.models import (
    CanvasContext,
    MousePositionEvent,
    ResearchEvent,
    ResearchWidgetTarget,
)


MOUSE_POSITION_FILENAME = "research_mouse_positions.json"
MOUSE_POSITION_SAMPLE_INTERVAL_MS = 250


class ResearchController:
    """Owns research timers, widget hooks, and event deduplication."""

    def __init__(self, parent: QWidget, enabled: bool) -> None:
        """Create research timers and optional tracker state for one annotator window."""
        self.enabled = bool(enabled)
        self.tracker: Optional[ResearchExperimentTracker] = (
            ResearchExperimentTracker() if self.enabled else None
        )
        self.mouse_tracker: Optional[ResearchMousePositionTracker] = (
            ResearchMousePositionTracker() if self.enabled else None
        )
        self._status_label: Optional[QLabel] = None
        self._pause_button: Optional[QPushButton] = None
        self._resume_button: Optional[QPushButton] = None
        self._hide_button: Optional[QPushButton] = None
        self._pending_click: Optional[ResearchEvent] = None
        self._last_event_signature: Optional[Tuple[int, int, int]] = None
        self._last_mouse_position_ms: Optional[int] = None
        self._parent = parent
        self._ui_timer = QTimer(parent)
        self._ui_timer.setInterval(250)
        self._ui_timer.timeout.connect(self.update_status_widgets)
        self._pending_click_timer = QTimer(parent)
        self._pending_click_timer.setSingleShot(True)
        self._pending_click_timer.timeout.connect(self.flush_pending_click)

    def bind_status_widgets(
        self,
        *,
        status_label: QLabel,
        pause_button: QPushButton,
        resume_button: QPushButton,
        hide_button: QPushButton,
    ) -> None:
        """Attach the status-bar widgets that mirror tracker state."""
        self._status_label = status_label
        self._pause_button = pause_button
        self._resume_button = resume_button
        self._hide_button = hide_button
        self._pause_button.clicked.connect(self.pause)
        self._resume_button.clicked.connect(self.resume)
        self._hide_button.clicked.connect(self.toggle_visibility)
        self.update_status_widgets()

    def start_ui_updates(self) -> None:
        """Start periodic timer-label refreshes when research mode is enabled."""
        if self.enabled and not self._ui_timer.isActive():
            self._ui_timer.start()

    def stop_ui_updates(self) -> None:
        """Stop periodic UI refreshes."""
        self._ui_timer.stop()

    def _format_elapsed(self, elapsed_ms: int) -> str:
        """Render elapsed milliseconds as an `HH:MM:SS` string."""
        total_seconds = max(0, int(elapsed_ms // 1000))
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def update_status_widgets(self) -> None:
        """Push tracker state into the bound status-bar widgets."""
        if not self.enabled or self.tracker is None:
            return
        tracker = self.tracker
        if self._status_label is not None:
            if tracker.started_at is None:
                self._status_label.setText("Research: --:--:--")
            elif tracker.is_hidden:
                self._status_label.setText("Research: hidden")
            else:
                self._status_label.setText(
                    f"Research: {self._format_elapsed(tracker.current_elapsed_ms())}"
                )
            self._status_label.setVisible(not tracker.is_hidden)
        if self._pause_button is not None:
            self._pause_button.setEnabled(tracker.started_at is not None and not tracker.is_paused)
        if self._resume_button is not None:
            self._resume_button.setEnabled(tracker.started_at is not None and tracker.is_paused)
        if self._hide_button is not None:
            self._hide_button.setEnabled(tracker.started_at is not None)
            self._hide_button.setText("Show Timer" if tracker.is_hidden else "Hide Timer")

    def start_new(self, frame_dir: Path, *, start_paused: bool) -> None:
        """Start a new research session and clear any pending click bookkeeping."""
        if not self.enabled or self.tracker is None:
            return
        self._pending_click = None
        self._pending_click_timer.stop()
        self._last_event_signature = None
        self._last_mouse_position_ms = None
        self.tracker.start_new(frame_dir, start_paused=start_paused)
        if self.mouse_tracker is not None:
            self.mouse_tracker.start_new(frame_dir, started_at=self.tracker.started_at)
        self.update_status_widgets()

    def pause(self) -> None:
        """Pause the research timer."""
        if self.tracker is None:
            return
        self.tracker.pause()
        self.update_status_widgets()

    def resume(self) -> None:
        """Resume the research timer."""
        if self.tracker is None:
            return
        self.tracker.resume()
        self.update_status_widgets()

    def toggle_visibility(self) -> None:
        """Toggle whether the research timer is shown in the status bar."""
        if self.tracker is None:
            return
        self.tracker.set_hidden(not self.tracker.is_hidden)
        self.update_status_widgets()

    def save_session(self, session_dir: Path, *, image_dir: Optional[Path]) -> None:
        """Persist research telemetry alongside the normal session payload."""
        if not self.enabled or self.tracker is None or image_dir is None:
            return
        self.flush_pending_click()
        write_session_json(session_dir / "research.json", self.tracker.to_dict())
        if self.mouse_tracker is not None:
            write_session_json(
                session_dir / MOUSE_POSITION_FILENAME,
                self.mouse_tracker.to_dict(),
            )

    def load_session(self, session_dir: Path, *, frame_dir: Path) -> None:
        """Restore research telemetry or create a paused session if none exists."""
        if not self.enabled or self.tracker is None:
            return
        research_path = session_dir / "research.json"
        if research_path.exists():
            try:
                self.tracker.load(read_session_json(research_path), default_frame_dir=frame_dir)
            except Exception:
                self.tracker.start_new(frame_dir, start_paused=True)
        else:
            self.tracker.start_new(frame_dir, start_paused=True)
        self.tracker.pause()
        self._pending_click = None
        self._pending_click_timer.stop()
        self._last_event_signature = None
        self._last_mouse_position_ms = None
        if self.mouse_tracker is not None:
            mouse_path = session_dir / MOUSE_POSITION_FILENAME
            if mouse_path.exists():
                try:
                    self.mouse_tracker.load(
                        read_session_json(mouse_path),
                        default_frame_dir=frame_dir,
                        default_started_at=self.tracker.started_at,
                    )
                except Exception:
                    self.mouse_tracker.start_new(frame_dir, started_at=self.tracker.started_at)
            else:
                self.mouse_tracker.start_new(frame_dir, started_at=self.tracker.started_at)
        self.update_status_widgets()

    def handle_widget_event(
        self,
        *,
        watched: QObject,
        event,
        current_frame_idx: int,
        has_frames: bool,
        image_label: QWidget,
    ) -> None:
        """Record non-canvas widget clicks while deduplicating Qt press/double-click noise."""
        if self.tracker is None or self.tracker.started_at is None:
            return
        if watched is image_label:
            return
        if event.type() not in (QEvent.MouseButtonPress, QEvent.MouseButtonDblClick):
            return
        if not hasattr(event, "button") or event.button() != Qt.LeftButton:
            return
        if not isinstance(watched, QWidget):
            return
        if not watched.isEnabled() or not watched.isVisible():
            return
        event_timestamp = int(event.timestamp()) if hasattr(event, "timestamp") else 0
        signature = (id(watched), int(event.type()), event_timestamp)
        if self._last_event_signature == signature:
            return
        self._last_event_signature = signature
        target = self._classify_widget_target(watched)
        if target is None:
            return
        event_data = ResearchEvent(
            frame_idx=current_frame_idx if has_frames else None,
            target_type=target.target_type,
            target_name=target.target_name,
            widget_class=watched.__class__.__name__,
        )
        if event.type() == QEvent.MouseButtonDblClick:
            if (
                self._pending_click is not None
                and self._pending_click.target_name == event_data.target_name
                and self._pending_click.target_type == event_data.target_type
            ):
                self._pending_click_timer.stop()
                self._pending_click = None
            self.tracker.record_event(event_data)
            self.update_status_widgets()
            return
        if self._pending_click is not None:
            same_target = (
                self._pending_click.target_name == event_data.target_name
                and self._pending_click.target_type == event_data.target_type
            )
            if not same_target:
                self.flush_pending_click()
        self._pending_click = event_data
        self._pending_click_timer.start(QApplication.doubleClickInterval())

    def record_mouse_position(
        self,
        *,
        watched: QObject,
        event,
        current_frame_idx: int,
        has_frames: bool,
        canvas_xy: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Record sampled whole-window mouse positions into the independent mouse log."""
        if self.tracker is None or self.mouse_tracker is None or self.tracker.started_at is None:
            return
        if event.type() != QEvent.MouseMove or not isinstance(watched, QWidget):
            return
        event_ms = self._mouse_event_timestamp_ms(event)
        if (
            self._last_mouse_position_ms is not None
            and event_ms - self._last_mouse_position_ms < MOUSE_POSITION_SAMPLE_INTERVAL_MS
        ):
            return
        self._last_mouse_position_ms = event_ms
        target = self._classify_widget_target(watched)
        widget_x = None
        widget_y = None
        window_x = None
        window_y = None
        if hasattr(event, "position"):
            position = event.position()
            widget_x = int(position.x())
            widget_y = int(position.y())
            window_point = self._parent.mapFromGlobal(watched.mapToGlobal(QPoint(widget_x, widget_y)))
            window_x = int(window_point.x())
            window_y = int(window_point.y())
        canvas_x = canvas_xy[0] if canvas_xy is not None else None
        canvas_y = canvas_xy[1] if canvas_xy is not None else None
        self.mouse_tracker.record_position(
            MousePositionEvent(
                frame_idx=current_frame_idx if has_frames else None,
                target_type=target.target_type if target is not None else "mouse",
                target_name=target.target_name if target is not None else "position",
                widget_class=watched.__class__.__name__,
                window_x_px=window_x,
                window_y_px=window_y,
                widget_x_px=widget_x,
                widget_y_px=widget_y,
                canvas_x_px=canvas_x,
                canvas_y_px=canvas_y,
            ),
            elapsed_ms=self.tracker.current_elapsed_ms(),
        )

    def _mouse_event_timestamp_ms(self, event) -> int:
        """Return a millisecond timestamp for throttling mouse-position samples."""
        if hasattr(event, "timestamp"):
            return int(event.timestamp())
        return int(monotonic() * 1000)

    def flush_pending_click(self) -> None:
        """Commit the delayed single-click once the double-click window expires."""
        if self.tracker is None or self._pending_click is None:
            return
        self.tracker.record_event(self._pending_click)
        self._pending_click = None
        self.update_status_widgets()

    def cancel_pending_click(self, target_type: Optional[str] = None) -> None:
        """Drop a delayed click, optionally only for a specific target type."""
        if self._pending_click is None:
            return
        if target_type is not None and self._pending_click.target_type != target_type:
            return
        self._pending_click = None
        self._pending_click_timer.stop()

    def queue_canvas_click(
        self,
        *,
        target_name: str,
        x_px: int,
        y_px: int,
        gesture_kind: str,
        current_frame_idx: int,
        has_frames: bool,
        image_label: QWidget,
        prompt_mode_index: int,
        active_object_id: Optional[int],
    ) -> None:
        """Stage a canvas click so a later double-click can replace it cleanly."""
        if self.tracker is None or self.tracker.started_at is None:
            return
        if self._pending_click is not None:
            self.flush_pending_click()
        self._pending_click = ResearchEvent(
            frame_idx=current_frame_idx if has_frames else None,
            target_type="canvas",
            target_name=target_name,
            widget_class=image_label.__class__.__name__,
            canvas_x_px=int(x_px),
            canvas_y_px=int(y_px),
            canvas_context=CanvasContext(
                gesture_kind=gesture_kind,
                prompt_mode_index=int(prompt_mode_index),
                active_object_id=int(active_object_id) if active_object_id is not None else None,
            ),
        )
        self._pending_click_timer.start(QApplication.doubleClickInterval())

    def record_canvas_event(
        self,
        *,
        target_name: str,
        x_px: int,
        y_px: int,
        gesture_kind: str,
        current_frame_idx: int,
        has_frames: bool,
        image_label: QWidget,
        prompt_mode_index: int,
        active_object_id: Optional[int],
    ) -> None:
        """Immediately record a canvas interaction that should not be delayed."""
        if self.tracker is None or self.tracker.started_at is None:
            return
        self._pending_click = None
        self._pending_click_timer.stop()
        self.tracker.record_event(
            ResearchEvent(
                frame_idx=current_frame_idx if has_frames else None,
                target_type="canvas",
                target_name=target_name,
                widget_class=image_label.__class__.__name__,
                canvas_x_px=int(x_px),
                canvas_y_px=int(y_px),
                canvas_context=CanvasContext(
                    gesture_kind=gesture_kind,
                    prompt_mode_index=int(prompt_mode_index),
                    active_object_id=int(active_object_id) if active_object_id is not None else None,
                ),
            )
        )
        self.update_status_widgets()

    def _classify_widget_target(self, widget: QWidget) -> Optional[ResearchWidgetTarget]:
        """Normalize Qt widget classes into stable telemetry target categories."""
        target_type = "widget"
        if isinstance(widget, QAbstractButton):
            if isinstance(widget, QCheckBox):
                target_type = "checkbox"
            elif isinstance(widget, QToolButton):
                target_type = "toolbutton"
            else:
                target_type = "button"
        elif isinstance(widget, QAbstractSlider):
            target_type = "slider"
        elif isinstance(widget, QListWidget):
            target_type = "list"
        elif isinstance(widget, QComboBox):
            target_type = "combo"
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox, QAbstractSpinBox)):
            target_type = "spinbox"
        elif isinstance(widget, QLineEdit):
            target_type = "lineedit"
        elif isinstance(widget, QMenu):
            target_type = "menu"
        target_name = self._widget_name(widget)
        return ResearchWidgetTarget(target_type=target_type, target_name=target_name)

    def _widget_name(self, widget: QWidget) -> str:
        """Choose a human-readable telemetry name for a clicked widget."""
        name = widget.objectName().strip() if widget.objectName() else ""
        if name:
            return name
        if isinstance(widget, (QPushButton, QToolButton, QCheckBox)):
            text = widget.text().strip()
            if text:
                return text
        if isinstance(widget, QLabel):
            text = widget.text().strip()
            if text:
                return text
        if isinstance(widget, QComboBox):
            text = widget.currentText().strip()
            if text:
                return f"{widget.__class__.__name__}:{text}"
        return widget.__class__.__name__
