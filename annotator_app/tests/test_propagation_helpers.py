"""Characterization tests for extracted propagation prompt and output helpers."""

import sys
import unittest
from pathlib import Path

import numpy as np

TESTS_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = TESTS_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from annotator.models import BoxPrompt, PointPrompt, SamFrameOutput
from annotator.propagation.frame_outputs import (
    find_disappeared_object_ids,
    merge_frame_outputs,
    remove_object_from_output,
    recovery_seed_frame_idx,
    sample_boxes_from_output_masks,
    target_limited_chunk_size,
)
from annotator.propagation.prompt_payloads import (
    build_propagation_seed_payload,
    build_segment_prompt_payload,
    clone_point_prompts,
    prompt_payload_inputs_to_task_payload,
    translate_prompts_by_box_delta,
)


class PropagationHelperTests(unittest.TestCase):
    def test_merge_frame_outputs_replaces_existing_and_appends_new(self) -> None:
        base = SamFrameOutput(
            obj_ids=[1],
            masks=[np.array([[True]], dtype=bool)],
            boxes_xywh_norm=[(0.1, 0.1, 0.2, 0.2)],
            scores=[0.2],
            tracker_scores=[0.3],
        )
        new = SamFrameOutput(
            obj_ids=[1, 2],
            masks=[np.array([[False]], dtype=bool), np.array([[True]], dtype=bool)],
            boxes_xywh_norm=[(0.3, 0.3, 0.4, 0.4), (0.5, 0.5, 0.1, 0.1)],
            scores=[0.9],
            tracker_scores=[],
        )

        merged = merge_frame_outputs(base, new)

        self.assertEqual(merged.obj_ids, [1, 2])
        self.assertEqual(merged.boxes_xywh_norm[0], (0.3, 0.3, 0.4, 0.4))
        self.assertEqual(merged.scores, [0.9, 0.0])
        self.assertEqual(merged.tracker_scores, [0.0, 0.0])

    def test_remove_object_from_output_returns_none_when_last_object_removed(self) -> None:
        output = SamFrameOutput(
            obj_ids=[4],
            masks=[np.array([[True]], dtype=bool)],
            boxes_xywh_norm=[(0.0, 0.0, 1.0, 1.0)],
            scores=[1.0],
            tracker_scores=[0.8],
        )

        self.assertIsNone(remove_object_from_output(output, 4))

    def test_sample_boxes_from_output_masks_resizes_and_extracts_bbox(self) -> None:
        output = SamFrameOutput(
            obj_ids=[7],
            masks=[np.array([[0, 1], [0, 0]], dtype=np.uint8)],
            boxes_xywh_norm=[(0.0, 0.0, 1.0, 1.0)],
            scores=[1.0],
            tracker_scores=[1.0],
        )

        boxes = sample_boxes_from_output_masks(output, image_h=4, image_w=4)

        self.assertEqual(boxes[7], BoxPrompt(x1_px=2, y1_px=0, x2_px=3, y2_px=1))

    def test_build_segment_prompt_payload_combines_points_and_box(self) -> None:
        payload = build_segment_prompt_payload(
            frame_prompts={
                1: [
                    PointPrompt(x_px=10, y_px=20, is_positive=True),
                    PointPrompt(x_px=30, y_px=40, is_positive=False),
                ]
            },
            frame_boxes={1: BoxPrompt(x1_px=5, y1_px=6, x2_px=50, y2_px=60)},
            obj_ids={1, 2},
            frame_size=(100, 200),
        )

        task_payload = prompt_payload_inputs_to_task_payload(payload)

        self.assertEqual(sorted(task_payload.keys()), [1])
        self.assertEqual(task_payload[1]["labels"], [1, 0, 2, 3])
        self.assertEqual(task_payload[1]["points_rel"][0], [0.1, 0.1])
        self.assertEqual(task_payload[1]["points_rel"][2], [0.05, 0.03])

    def test_build_propagation_seed_payload_reports_missing_and_preserves_mask_only_object(self) -> None:
        seed_output = SamFrameOutput(
            obj_ids=[2],
            masks=[np.array([[True, False], [False, False]], dtype=bool)],
            boxes_xywh_norm=[(0.0, 0.0, 1.0, 1.0)],
            scores=[0.5],
            tracker_scores=[0.6],
        )

        result = build_propagation_seed_payload(
            frame_prompts={1: [PointPrompt(x_px=25, y_px=50, is_positive=True)]},
            frame_boxes={1: BoxPrompt(x1_px=20, y1_px=40, x2_px=60, y2_px=80)},
            seed_output=seed_output,
            enabled_obj_ids={1, 2, 3},
            frame_size=(100, 100),
            use_point_prompts=True,
            sampled_boxes={},
        )

        self.assertEqual(result.missing_obj_ids, [3])
        self.assertEqual(sorted(result.payload_by_obj_id.keys()), [1, 2])
        self.assertEqual(result.payload_by_obj_id[1].labels, [2, 3, 1])
        self.assertEqual(result.payload_by_obj_id[2].points_rel, [])
        self.assertEqual(result.payload_by_obj_id[2].labels, [])
        self.assertEqual(result.payload_by_obj_id[2].mask_input.dtype, np.float32)

    def test_translate_prompts_by_box_delta_clamps_to_frame(self) -> None:
        translated = translate_prompts_by_box_delta(
            points=[PointPrompt(x_px=9, y_px=8, is_positive=True)],
            src_box=BoxPrompt(x1_px=0, y1_px=0, x2_px=4, y2_px=4),
            dst_box=BoxPrompt(x1_px=10, y1_px=10, x2_px=14, y2_px=14),
            frame_size=(12, 12),
        )

        self.assertEqual(translated, [PointPrompt(x_px=11, y_px=11, is_positive=True)])

    def test_clone_point_prompts_creates_independent_objects(self) -> None:
        source = [PointPrompt(x_px=1, y_px=2, is_positive=True)]

        cloned = clone_point_prompts(source)

        self.assertEqual(cloned, source)
        self.assertIsNot(cloned[0], source[0])

    def test_find_disappeared_object_ids_flags_missing_and_empty_masks(self) -> None:
        output = SamFrameOutput(
            obj_ids=[1, 2],
            masks=[
                np.array([[True, False]], dtype=bool),
                np.array([[False, False]], dtype=bool),
            ],
            boxes_xywh_norm=[(0.0, 0.0, 0.5, 0.5), (0.0, 0.0, 0.5, 0.5)],
            scores=[1.0, 1.0],
            tracker_scores=[1.0, 1.0],
        )

        disappeared = find_disappeared_object_ids(output, {1, 2, 3})

        self.assertEqual(disappeared, [2, 3])

    def test_recovery_seed_frame_idx_clamps_to_run_start(self) -> None:
        self.assertEqual(
            recovery_seed_frame_idx(
                disappear_frame_idx=6,
                run_start_frame_idx=2,
            ),
            3,
        )
        self.assertEqual(
            recovery_seed_frame_idx(
                disappear_frame_idx=4,
                run_start_frame_idx=3,
            ),
            3,
        )

    def test_target_limited_chunk_size_caps_at_target(self) -> None:
        self.assertEqual(
            target_limited_chunk_size(
                seed_frame_idx=10,
                target_frame_idx=14,
                requested_chunk_size=100,
            ),
            5,
        )
        self.assertEqual(
            target_limited_chunk_size(
                seed_frame_idx=10,
                target_frame_idx=None,
                requested_chunk_size=100,
            ),
            100,
        )


if __name__ == "__main__":
    unittest.main()
