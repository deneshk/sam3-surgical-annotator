"""Frame metadata and decoded-image cache helpers for the annotator UI."""

from __future__ import annotations

import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image as PilImage

DECODED_FRAME_CACHE_SIZE = 3


def read_frame_size_from_path(path: Path) -> Optional[Tuple[int, int]]:
    """Read width and height without decoding the full image when possible."""
    try:
        with PilImage.open(str(path)) as img:
            width, height = img.size
        return int(width), int(height)
    except Exception:
        pass

    frame = cv2.imread(str(path))
    if frame is not None:
        height, width = frame.shape[:2]
        return int(width), int(height)
    return None


def _read_frame_size_for_cache_entry(
    entry: Tuple[int, Path],
) -> Tuple[int, Optional[Tuple[int, int]]]:
    """Return a frame-size cache entry for one indexed frame path."""
    frame_idx, path = entry
    return frame_idx, read_frame_size_from_path(path)


class FrameIoCache:
    """Caches frame dimensions and a small bounded set of decoded BGR frames."""

    def __init__(self, decoded_cache_size: int = DECODED_FRAME_CACHE_SIZE) -> None:
        self.decoded_cache_size = max(1, int(decoded_cache_size))
        self.frame_size_cache: dict[int, Tuple[int, int]] = {}
        self.frame_bgr_cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def clear(self) -> None:
        """Drop frame metadata and decoded-image caches."""
        self.frame_size_cache.clear()
        self.frame_bgr_cache.clear()

    def prime_frame_sizes(self, frame_paths: Sequence[Path]) -> None:
        """Read frame dimensions in parallel after a fresh directory load."""
        paths = list(frame_paths)
        if not paths:
            return
        max_workers = min(8, os.cpu_count() or 4, len(paths))
        if max_workers <= 1:
            for frame_idx, path in enumerate(paths):
                size = read_frame_size_from_path(path)
                if size is not None:
                    self.frame_size_cache[frame_idx] = size
            return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for frame_idx, size in executor.map(
                _read_frame_size_for_cache_entry,
                enumerate(paths),
            ):
                if size is not None:
                    self.frame_size_cache[frame_idx] = size

    def get_frame_size(
        self,
        frame_paths: Sequence[Path],
        frame_idx: int,
    ) -> Optional[Tuple[int, int]]:
        """Return width and height for one frame using cached metadata when available."""
        if not frame_paths or frame_idx < 0 or frame_idx >= len(frame_paths):
            return None
        cached_size = self.frame_size_cache.get(frame_idx)
        if cached_size is not None:
            return cached_size
        size = read_frame_size_from_path(frame_paths[frame_idx])
        if size is not None:
            self.frame_size_cache[frame_idx] = size
        return size

    def read_frame_bgr(
        self,
        frame_paths: Sequence[Path],
        frame_idx: int,
    ) -> Optional[np.ndarray]:
        """Read one frame as a BGR array using a bounded decoded-frame cache."""
        if not frame_paths or frame_idx < 0 or frame_idx >= len(frame_paths):
            return None
        cached_frame = self.frame_bgr_cache.get(frame_idx)
        if cached_frame is not None:
            self.frame_bgr_cache.move_to_end(frame_idx)
            return cached_frame.copy()

        frame = cv2.imread(str(frame_paths[frame_idx]))
        if frame is None:
            try:
                with PilImage.open(str(frame_paths[frame_idx])) as img:
                    rgb = img.convert("RGB")
                frame = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
            except Exception:
                return None

        self._cache_decoded_frame(frame_idx, frame)
        return frame.copy()

    def _cache_decoded_frame(self, frame_idx: int, frame: np.ndarray) -> None:
        """Store one decoded BGR frame while keeping cache memory bounded."""
        self.frame_bgr_cache[frame_idx] = frame.copy()
        self.frame_bgr_cache.move_to_end(frame_idx)
        while len(self.frame_bgr_cache) > self.decoded_cache_size:
            self.frame_bgr_cache.popitem(last=False)
