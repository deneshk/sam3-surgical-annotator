"""Tests for frame metadata and decoded-image caches."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TESTS_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = TESTS_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from annotator.frame_io import DECODED_FRAME_CACHE_SIZE, FrameIoCache


def _save_rgb_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    Image.new("RGB", size, color=color).save(path)


class FrameIoCacheTests(unittest.TestCase):
    def test_get_frame_size_reuses_cached_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_path = Path(tmp_dir) / "frame.png"
            _save_rgb_image(frame_path, size=(7, 5), color=(10, 20, 30))
            cache = FrameIoCache()

            self.assertEqual(cache.get_frame_size([frame_path], 0), (7, 5))
            frame_path.unlink()

            self.assertEqual(cache.get_frame_size([frame_path], 0), (7, 5))

    def test_prime_frame_size_cache_reads_all_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_paths = []
            for idx, size in enumerate([(3, 4), (5, 6), (7, 8)]):
                frame_path = Path(tmp_dir) / f"frame_{idx}.png"
                _save_rgb_image(frame_path, size=size, color=(idx, idx, idx))
                frame_paths.append(frame_path)
            cache = FrameIoCache()

            cache.prime_frame_sizes(frame_paths)

            self.assertEqual(cache.frame_size_cache, {0: (3, 4), 1: (5, 6), 2: (7, 8)})

    def test_read_frame_bgr_cache_returns_independent_copies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_path = Path(tmp_dir) / "frame.png"
            _save_rgb_image(frame_path, size=(2, 2), color=(10, 20, 30))
            cache = FrameIoCache()

            first = cache.read_frame_bgr([frame_path], 0)
            self.assertIsNotNone(first)
            first[0, 0] = np.array([255, 255, 255], dtype=first.dtype)
            second = cache.read_frame_bgr([frame_path], 0)

            self.assertFalse(np.array_equal(first, second))

    def test_decoded_frame_cache_is_lru_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_paths = []
            for idx in range(DECODED_FRAME_CACHE_SIZE + 1):
                frame_path = Path(tmp_dir) / f"frame_{idx}.png"
                _save_rgb_image(frame_path, size=(2, 2), color=(idx, idx, idx))
                frame_paths.append(frame_path)
            cache = FrameIoCache()

            for idx in range(len(frame_paths)):
                self.assertIsNotNone(cache.read_frame_bgr(frame_paths, idx))

            self.assertEqual(len(cache.frame_bgr_cache), DECODED_FRAME_CACHE_SIZE)
            self.assertEqual(list(cache.frame_bgr_cache.keys()), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
