"""Tests for disk-backed frame output storage."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TESTS_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = TESTS_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from annotator.models import SamFrameOutput
from annotator.storage.frame_output_store import FrameOutputStore


def _output(obj_id: int, mask: np.ndarray) -> SamFrameOutput:
    return SamFrameOutput(
        obj_ids=[obj_id],
        masks=[mask],
        boxes_xywh_norm=[(0.1, 0.2, 0.3, 0.4)],
        scores=[0.9],
        tracker_scores=[0.8],
    )


class FrameOutputStoreTests(unittest.TestCase):
    def test_round_trip_preserves_masks_and_metadata(self) -> None:
        mask = np.array([[True, False, True], [False, True, False]], dtype=bool)
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = FrameOutputStore(cache_dir=Path(tmp_dir), max_cached_frames=2)
            store[4] = _output(7, mask)

            loaded = store[4]

            self.assertEqual(loaded.obj_ids, [7])
            self.assertEqual(loaded.boxes_xywh_norm, [(0.1, 0.2, 0.3, 0.4)])
            self.assertEqual(loaded.scores, [0.9])
            self.assertEqual(loaded.tracker_scores, [0.8])
            self.assertTrue(np.array_equal(loaded.masks[0], mask))

    def test_loaded_outputs_are_copied(self) -> None:
        mask = np.array([[True, False], [False, True]], dtype=bool)
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = FrameOutputStore(cache_dir=Path(tmp_dir), max_cached_frames=2)
            store[0] = _output(1, mask)

            first = store[0]
            first.masks[0][0, 0] = False
            second = store[0]

            self.assertTrue(second.masks[0][0, 0])

    def test_lru_cache_is_bounded_without_dropping_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = FrameOutputStore(cache_dir=Path(tmp_dir), max_cached_frames=2)
            for frame_idx in range(4):
                store[frame_idx] = _output(frame_idx + 1, np.ones((2, 2), dtype=bool))

            self.assertEqual(len(store), 4)
            self.assertEqual(store.cached_frame_count, 2)
            self.assertIn(0, store)
            self.assertTrue(np.array_equal(store[0].masks[0], np.ones((2, 2), dtype=bool)))

    def test_pop_removes_record_and_mask_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            store = FrameOutputStore(cache_dir=cache_dir, max_cached_frames=2)
            store[3] = _output(4, np.ones((2, 2), dtype=bool))
            self.assertTrue(list(cache_dir.glob("*.npz")))

            removed = store.pop(3)

            self.assertIsNotNone(removed)
            self.assertNotIn(3, store)
            self.assertFalse(list(cache_dir.glob("*.npz")))

    def test_remove_object_updates_record_without_dropping_other_masks(self) -> None:
        first_mask = np.ones((2, 2), dtype=bool)
        second_mask = np.array([[False, True], [True, False]], dtype=bool)
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            store = FrameOutputStore(cache_dir=cache_dir, max_cached_frames=2)
            store[0] = SamFrameOutput(
                obj_ids=[1, 2],
                masks=[first_mask, second_mask],
                boxes_xywh_norm=[(0.0, 0.0, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)],
                scores=[0.1, 0.2],
                tracker_scores=[0.3, 0.4],
            )

            removed = store.remove_object(0, 1)
            loaded = store[0]

            self.assertTrue(removed)
            self.assertEqual(loaded.obj_ids, [2])
            self.assertEqual(loaded.scores, [0.2])
            self.assertTrue(np.array_equal(loaded.masks[0], second_mask))
            self.assertEqual(len(list(cache_dir.glob("*.npz"))), 1)


if __name__ == "__main__":
    unittest.main()
