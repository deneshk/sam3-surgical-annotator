"""Round-trip and compatibility tests for session persistence boundaries."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TESTS_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = TESTS_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from annotator.models import (
    BoxPrompt,
    ExperimentalSettings,
    ObjectInfo,
    PointPrompt,
    PROPAGATION_MODE_COPY_BOXES,
    PROPAGATION_MODE_TRACKER,
    PropagationSettings,
    SamFrameOutput,
    ViewSettings,
)
from annotator.persistence.session_mapper import data_to_payload
from annotator.persistence.session_models import SessionPayload
from annotator.persistence.session_repository import SessionRepository


class SessionPersistenceTests(unittest.TestCase):
    def test_repository_round_trip_preserves_session_payload(self) -> None:
        mask = np.array([[True, False], [False, True]], dtype=bool)
        payload = SessionPayload(
            frame_dir="D:/frames",
            frame_files=["0001.png", "0002.png"],
            current_frame_idx=1,
            active_object_id=7,
            checkpoint_path="checkpoint.pt",
            flagged_frames=[1],
            objects=[ObjectInfo(obj_id=7, name="tool", color_bgr=(10, 20, 30))],
            hidden_obj_ids=[7],
            solo_object_id=7,
            prompts_by_frame_obj={
                0: {
                    7: [PointPrompt(x_px=11, y_px=13, is_positive=True)],
                }
            },
            boxes_by_frame_obj={
                0: {
                    7: BoxPrompt(x1_px=1, y1_px=2, x2_px=30, y2_px=40),
                }
            },
            box_locks_by_frame_obj={0: {7: True}},
            propagation_overrides_by_frame_obj={0: {7: False}},
            outputs_by_frame={
                0: SamFrameOutput(
                    obj_ids=[7],
                    masks=[mask],
                    boxes_xywh_norm=[(0.1, 0.2, 0.3, 0.4)],
                    scores=[0.9],
                    tracker_scores=[],
                )
            },
            view_settings=ViewSettings(
                show_prompts=False,
                show_segmentations=True,
                show_boxes=False,
                show_box_titles=True,
                segmentation_opacity=0.25,
                box_line_thickness=4,
            ),
            propagation_settings=PropagationSettings(
                mode=PROPAGATION_MODE_COPY_BOXES,
                chunk_size=5,
                chunks=3,
                use_target_frame=True,
                target_frame_idx=9,
                translate_prompts=False,
                use_point_prompts=False,
                auto_propagate_next=True,
            ),
            experimental_settings=ExperimentalSettings(
                recondition_every_nth_frame=8,
                recondition_high_conf_thresh=0.55,
                recondition_high_iou_thresh=0.66,
                use_one_session_chunked_propagation=True,
                smart_propagation_enabled=False,
                smart_propagation_rewind_frames=4,
                smart_propagation_recovery_chunk_size=12,
            ),
            prompt_mode_index=1,
        )

        repository = SessionRepository()
        with tempfile.TemporaryDirectory() as tmp_dir:
            session_dir = Path(tmp_dir)
            repository.save_session(session_dir, payload)
            loaded = repository.load_session(session_dir / "session.json")

            raw_data = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))

        self.assertEqual(raw_data["propagation"]["n_frames"], 5)
        self.assertEqual(raw_data["propagation"]["mode"], PROPAGATION_MODE_COPY_BOXES)
        self.assertEqual(loaded.current_frame_idx, 1)
        self.assertEqual(loaded.active_object_id, 7)
        self.assertEqual(loaded.checkpoint_path, "checkpoint.pt")
        self.assertEqual(loaded.hidden_obj_ids, [7])
        self.assertEqual(loaded.solo_object_id, 7)
        self.assertEqual(loaded.prompt_mode_index, 1)
        self.assertEqual(loaded.propagation_settings.chunk_size, 5)
        self.assertEqual(loaded.propagation_settings.mode, PROPAGATION_MODE_COPY_BOXES)
        self.assertTrue(loaded.propagation_settings.auto_propagate_next)
        self.assertEqual(loaded.view_settings.box_line_thickness, 4)
        self.assertTrue(loaded.experimental_settings.use_one_session_chunked_propagation)
        self.assertEqual(loaded.outputs_by_frame[0].tracker_scores, [0.0])
        self.assertTrue(np.array_equal(loaded.outputs_by_frame[0].masks[0], mask))

    def test_mapper_uses_fallback_shape_and_normalizes_unknown_modes(self) -> None:
        payload = data_to_payload(
            {
                "frame_dir": "D:/frames",
                "frame_files": ["0001.png"],
                "propagation": {"mode": "unexpected"},
                "outputs": {
                    "0": {
                        "obj_ids": [3],
                        "boxes_xywh_norm": [[0.0, 0.0, 1.0, 1.0]],
                        "scores": [0.1],
                        "tracker_scores": [],
                        "mask_paths": [None],
                    }
                },
            },
            load_mask=lambda _path: np.ones((2, 2), dtype=bool),
            fallback_shape_for_frame=lambda _frame_idx: (4, 5),
        )

        self.assertEqual(payload.propagation_settings.mode, PROPAGATION_MODE_TRACKER)
        self.assertEqual(payload.outputs_by_frame[0].masks[0].shape, (4, 5))
        self.assertFalse(payload.outputs_by_frame[0].masks[0].any())
        self.assertEqual(payload.outputs_by_frame[0].tracker_scores, [0.0])


if __name__ == "__main__":
    unittest.main()
