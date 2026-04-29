"""Tests for runtime-state records and research telemetry helpers."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TESTS_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = TESTS_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QWidget

    HAVE_PYSIDE6 = True
except ModuleNotFoundError:
    QEvent = None
    QPointF = None
    Qt = None
    QApplication = None
    QLabel = None
    QPushButton = None
    QWidget = None
    HAVE_PYSIDE6 = False

from annotator.propagation.runtime import PrefetchState, PropagationRuntimeState, SamTaskContext
from annotator.research.experiment import ResearchExperimentTracker
from annotator.research.models import CanvasContext, MousePositionEvent, ResearchEvent

if HAVE_PYSIDE6:
    from annotator.research.controller import ResearchController


class FakeMouseEvent:
    def __init__(self, event_type, timestamp: int, button=None, x: float = 0.0, y: float = 0.0) -> None:
        self._event_type = event_type
        self._timestamp = timestamp
        self._button = Qt.LeftButton if button is None and HAVE_PYSIDE6 else button
        self._position = QPointF(x, y) if HAVE_PYSIDE6 else None

    def type(self):
        return self._event_type

    def button(self):
        return self._button

    def timestamp(self):
        return self._timestamp

    def position(self):
        return self._position


class RuntimeStateTests(unittest.TestCase):
    def test_mouse_position_event_round_trip(self) -> None:
        event = MousePositionEvent(
            frame_idx=5,
            target_type="button",
            target_name="Segment",
            widget_class="QPushButton",
            window_x_px=100,
            window_y_px=110,
            widget_x_px=12,
            widget_y_px=14,
            canvas_x_px=50,
            canvas_y_px=60,
            event_index=3,
            timestamp_iso="2026-04-25T12:00:00+00:00",
            elapsed_ms=1234,
        )

        restored = MousePositionEvent.from_dict(event.to_dict())

        self.assertEqual(restored.target_name, "Segment")
        self.assertEqual(restored.window_x_px, 100)
        self.assertEqual(restored.widget_y_px, 14)
        self.assertEqual(restored.canvas_x_px, 50)
        self.assertEqual(restored.event_index, 3)

    def test_sam_task_context_version_alias_maps_to_cache_generation(self) -> None:
        context = SamTaskContext(kind="prefetch", cache_generation=3)

        self.assertEqual(context.get("version"), 3)
        self.assertEqual(context["version"], 3)

        context["version"] = 5

        self.assertEqual(context.cache_generation, 5)

    def test_prefetch_state_tracks_generation_and_cached_frame_validity(self) -> None:
        state = PrefetchState()

        state.note_prompt_change()
        state.begin(seed_frame_idx=4, target_frame_idx=5)
        state.record_cached_frame(5, [1, 2])

        self.assertTrue(state.has_valid_cache(5))
        self.assertEqual(state.provenance_by_frame[5].obj_ids, [1, 2])

        state.finish()
        state.note_prompt_change()

        self.assertFalse(state.has_valid_cache(5))
        self.assertEqual(state.cache_generation, 2)
        self.assertEqual(state.provenance_by_frame, {})

    def test_propagation_runtime_state_resets_run_flags(self) -> None:
        state = PropagationRuntimeState()

        state.begin_run(enabled_obj_ids={1, 3}, view_frame_idx=8)
        state.begin_chunk(chunk_idx=2, seed_frame_idx=8, task_id="propagate:2")
        state.reset()

        self.assertFalse(state.busy)
        self.assertIsNone(state.enabled_obj_ids)
        self.assertIsNone(state.active_chunk_idx)
        self.assertFalse(state.stop_requested)


@unittest.skipUnless(HAVE_PYSIDE6, "PySide6 is not installed")
class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_research_tracker_round_trip_preserves_typed_events(self) -> None:
        tracker = ResearchExperimentTracker()
        tracker.start_new(Path("D:/frames"), start_paused=True)
        tracker.record_event(
            ResearchEvent(
                frame_idx=4,
                target_type="canvas",
                target_name="canvas_prompt_press",
                widget_class="ClickableImageLabel",
                canvas_x_px=12,
                canvas_y_px=34,
                canvas_context=CanvasContext(
                    gesture_kind="single_click",
                    prompt_mode_index=2,
                    active_object_id=7,
                ),
            )
        )

        payload = tracker.to_dict()
        restored = ResearchExperimentTracker()
        restored.load(payload, default_frame_dir=Path("D:/fallback"))

        self.assertEqual(restored.frame_dir, "D:/frames")
        self.assertEqual(len(restored.events), 1)
        self.assertEqual(restored.events[0].target_name, "canvas_prompt_press")
        self.assertEqual(restored.events[0].canvas_context.prompt_mode_index, 2)
        self.assertEqual(restored.events[0].event_index, 1)

    def test_research_controller_deduplicates_widget_clicks_and_records_canvas_double_click(self) -> None:
        parent = QWidget()
        controller = ResearchController(parent, enabled=True)
        controller.start_new(Path("D:/frames"), start_paused=False)

        button = QPushButton("Save Session", parent)
        image_label = QLabel(parent)
        widget_event = FakeMouseEvent(QEvent.MouseButtonPress, timestamp=11)

        controller.handle_widget_event(
            watched=button,
            event=widget_event,
            current_frame_idx=3,
            has_frames=True,
            image_label=image_label,
        )
        controller.handle_widget_event(
            watched=button,
            event=widget_event,
            current_frame_idx=3,
            has_frames=True,
            image_label=image_label,
        )
        controller.flush_pending_click()

        self.assertEqual(len(controller.tracker.events), 1)
        self.assertEqual(controller.tracker.events[0].target_type, "button")
        self.assertEqual(controller.tracker.events[0].target_name, "Save Session")

        controller.queue_canvas_click(
            target_name="canvas_prompt_press",
            x_px=15,
            y_px=25,
            gesture_kind="single_click",
            current_frame_idx=3,
            has_frames=True,
            image_label=image_label,
            prompt_mode_index=1,
            active_object_id=9,
        )
        controller.record_canvas_event(
            target_name="canvas_double_click",
            x_px=16,
            y_px=26,
            gesture_kind="double_click",
            current_frame_idx=3,
            has_frames=True,
            image_label=image_label,
            prompt_mode_index=2,
            active_object_id=9,
        )

        self.assertEqual(len(controller.tracker.events), 2)
        canvas_event = controller.tracker.events[1]
        self.assertEqual(canvas_event.target_type, "canvas")
        self.assertEqual(canvas_event.target_name, "canvas_double_click")
        self.assertEqual(canvas_event.canvas_context.gesture_kind, "double_click")
        self.assertEqual(canvas_event.canvas_context.active_object_id, 9)

    def test_research_controller_samples_mouse_positions_to_independent_tracker(self) -> None:
        parent = QWidget()
        controller = ResearchController(parent, enabled=True)
        controller.start_new(Path("D:/frames"), start_paused=False)

        button = QPushButton("Segment", parent)
        first_move = FakeMouseEvent(QEvent.MouseMove, timestamp=1000, x=12, y=14)
        skipped_move = FakeMouseEvent(QEvent.MouseMove, timestamp=1100, x=13, y=15)
        later_move = FakeMouseEvent(QEvent.MouseMove, timestamp=1250, x=20, y=22)

        controller.record_mouse_position(
            watched=button,
            event=first_move,
            current_frame_idx=2,
            has_frames=True,
        )
        controller.record_mouse_position(
            watched=button,
            event=skipped_move,
            current_frame_idx=2,
            has_frames=True,
        )
        controller.record_mouse_position(
            watched=button,
            event=later_move,
            current_frame_idx=2,
            has_frames=True,
        )

        self.assertEqual(len(controller.tracker.events), 0)
        self.assertEqual(len(controller.mouse_tracker.positions), 2)
        first_position = controller.mouse_tracker.positions[0]
        self.assertEqual(first_position.target_type, "button")
        self.assertEqual(first_position.target_name, "Segment")
        self.assertEqual(first_position.widget_x_px, 12)
        self.assertEqual(first_position.widget_y_px, 14)
        self.assertEqual(first_position.event_index, 1)

    def test_research_controller_saves_mouse_positions_separately(self) -> None:
        parent = QWidget()
        controller = ResearchController(parent, enabled=True)
        controller.start_new(Path("D:/frames"), start_paused=False)
        image_label = QLabel(parent)

        controller.record_mouse_position(
            watched=image_label,
            event=FakeMouseEvent(QEvent.MouseMove, timestamp=2000, x=5, y=6),
            current_frame_idx=4,
            has_frames=True,
            canvas_xy=(55, 66),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir)
            controller.save_session(session_dir, image_dir=Path("D:/frames"))

            self.assertTrue((session_dir / "research.json").exists())
            mouse_path = session_dir / "research_mouse_positions.json"
            self.assertTrue(mouse_path.exists())

            restored = ResearchController(QWidget(), enabled=True)
            restored.load_session(session_dir, frame_dir=Path("D:/frames"))

        self.assertEqual(len(restored.tracker.events), 0)
        self.assertEqual(len(restored.mouse_tracker.positions), 1)
        position = restored.mouse_tracker.positions[0]
        self.assertEqual(position.canvas_x_px, 55)
        self.assertEqual(position.canvas_y_px, 66)


if __name__ == "__main__":
    unittest.main()
