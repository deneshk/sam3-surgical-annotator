"""Tests for optional mask post-processing helpers."""

import sys
import unittest
from pathlib import Path

import numpy as np

TESTS_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = TESTS_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from annotator.models import SamFrameOutput
from annotator.postprocessing.masks import (
    MaskPostProcessingSettings,
    apply_morphology,
    fill_small_holes,
    mask_to_box_xywh_norm,
    process_frame_output,
    remove_small_components,
    simplify_mask_contours,
)


class MaskPostProcessingTests(unittest.TestCase):
    def test_remove_small_components_drops_islands_below_threshold(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        mask[1:5, 1:5] = True
        mask[7, 7] = True

        cleaned = remove_small_components(mask, min_area_px=4)

        self.assertTrue(cleaned[1:5, 1:5].all())
        self.assertFalse(cleaned[7, 7])

    def test_fill_small_holes_preserves_large_holes(self) -> None:
        mask = np.ones((8, 8), dtype=bool)
        mask[2, 2] = False
        mask[4:6, 4:6] = False

        cleaned = fill_small_holes(mask, max_hole_area_px=1)

        self.assertTrue(cleaned[2, 2])
        self.assertFalse(cleaned[4:6, 4:6].any())

    def test_morphological_open_removes_small_noise(self) -> None:
        mask = np.zeros((9, 9), dtype=bool)
        mask[2:5, 2:5] = True
        mask[7, 7] = True

        cleaned = apply_morphology(mask, open_mask=True, close_mask=False, kernel_size=3)

        self.assertTrue(cleaned[2:5, 2:5].all())
        self.assertFalse(cleaned[7, 7])

    def test_morphological_close_fills_narrow_gap(self) -> None:
        mask = np.zeros((7, 7), dtype=bool)
        mask[2:5, 1:3] = True
        mask[2:5, 4:6] = True

        cleaned = apply_morphology(mask, open_mask=False, close_mask=True, kernel_size=3)

        self.assertTrue(cleaned[3, 3])

    def test_mask_to_box_xywh_norm_uses_processed_mask_bounds(self) -> None:
        mask = np.zeros((10, 20), dtype=bool)
        mask[2:6, 4:10] = True

        self.assertEqual(mask_to_box_xywh_norm(mask), (0.2, 0.2, 0.3, 0.4))

    def test_simplify_mask_contours_reduces_jagged_boundary_points(self) -> None:
        import cv2

        mask = np.zeros((60, 60), dtype=np.uint8)
        points = np.array(
            [
                [8, 48],
                [8, 12],
                [12, 10],
                [16, 12],
                [20, 10],
                [24, 12],
                [28, 10],
                [32, 12],
                [36, 10],
                [40, 12],
                [44, 10],
                [48, 12],
                [52, 10],
                [52, 48],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [points], 1)
        mask = mask.astype(bool)
        before_contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        simplified = simplify_mask_contours(mask, epsilon_fraction=0.03)
        after_contours, _ = cv2.findContours(simplified.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.assertTrue(simplified.any())
        self.assertLess(
            sum(contour.shape[0] for contour in after_contours),
            sum(contour.shape[0] for contour in before_contours),
        )

    def test_process_frame_output_recomputes_boxes_and_preserves_scores(self) -> None:
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:8, 2:8] = True
        mask[9, 9] = True
        output = SamFrameOutput(
            obj_ids=[3],
            masks=[mask],
            boxes_xywh_norm=[(0.0, 0.0, 1.0, 1.0)],
            scores=[0.7],
            tracker_scores=[0.6],
        )

        processed = process_frame_output(
            output,
            MaskPostProcessingSettings(
                enabled=True,
                remove_small_components=True,
                min_component_area_px=4,
            ),
        )

        self.assertFalse(processed.masks[0][9, 9])
        self.assertEqual(processed.boxes_xywh_norm, [(0.2, 0.2, 0.6, 0.6)])
        self.assertEqual(processed.scores, [0.7])
        self.assertEqual(processed.tracker_scores, [0.6])

    def test_process_frame_output_simplifies_masks_when_enabled(self) -> None:
        mask = np.zeros((30, 30), dtype=bool)
        for row in range(5, 25):
            left = 5 if row % 2 == 0 else 6
            right = 24 if row % 2 == 0 else 23
            mask[row, left:right] = True
        output = SamFrameOutput(
            obj_ids=[1],
            masks=[mask],
            boxes_xywh_norm=[(0.0, 0.0, 1.0, 1.0)],
            scores=[0.5],
            tracker_scores=[0.4],
        )

        processed = process_frame_output(
            output,
            MaskPostProcessingSettings(
                enabled=True,
                remove_small_components=False,
                simplify_contours=True,
                simplify_epsilon_fraction=0.02,
            ),
        )

        self.assertTrue(processed.masks[0].any())
        self.assertNotEqual(processed.boxes_xywh_norm, output.boxes_xywh_norm)

    def test_disabled_settings_return_equivalent_output_copy(self) -> None:
        mask = np.array([[True, False], [False, True]], dtype=bool)
        output = SamFrameOutput(
            obj_ids=[1],
            masks=[mask],
            boxes_xywh_norm=[(0.1, 0.2, 0.3, 0.4)],
            scores=[0.9],
            tracker_scores=[0.8],
        )

        processed = process_frame_output(output, MaskPostProcessingSettings(enabled=False))
        processed.masks[0][0, 0] = False

        self.assertEqual(processed.obj_ids, output.obj_ids)
        self.assertEqual(processed.boxes_xywh_norm, output.boxes_xywh_norm)
        self.assertEqual(processed.scores, output.scores)
        self.assertTrue(output.masks[0][0, 0])


if __name__ == "__main__":
    unittest.main()
