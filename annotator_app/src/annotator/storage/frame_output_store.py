"""Disk-backed storage for per-frame SAM outputs.

The app can accumulate thousands of boolean masks during long annotation runs.
This store keeps output metadata in memory while spilling mask arrays to a
temporary directory, then reloads recent frames through a small LRU cache.
"""

from __future__ import annotations

import shutil
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np

from annotator.models import SamFrameOutput


@dataclass(frozen=True)
class _MaskRef:
    """Location and shape metadata for one packed boolean mask."""

    path: Path
    shape: tuple[int, ...]


@dataclass(frozen=True)
class _FrameOutputRecord:
    """In-memory metadata for one frame's output."""

    obj_ids: tuple[int, ...]
    mask_refs: tuple[_MaskRef, ...]
    boxes_xywh_norm: tuple[tuple[float, float, float, float], ...]
    scores: tuple[float, ...]
    tracker_scores: tuple[float, ...]


class FrameOutputStore:
    """Dict-like frame output map that stores masks outside long-lived RAM."""

    def __init__(self, cache_dir: Optional[Path] = None, max_cached_frames: int = 32) -> None:
        self._max_cached_frames = max(1, int(max_cached_frames))
        self._records: dict[int, _FrameOutputRecord] = {}
        self._cache: OrderedDict[int, SamFrameOutput] = OrderedDict()
        self._owned_cache_dir = cache_dir is None
        self._cache_dir = Path(cache_dir) if cache_dir is not None else Path(
            tempfile.mkdtemp(prefix="sam3annotator_masks_")
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self._records)

    def __bool__(self) -> bool:
        return bool(self._records)

    def __contains__(self, frame_idx: object) -> bool:
        try:
            return int(frame_idx) in self._records
        except (TypeError, ValueError):
            return False

    def __getitem__(self, frame_idx: int) -> SamFrameOutput:
        output = self.get(frame_idx)
        if output is None:
            raise KeyError(frame_idx)
        return output

    def __setitem__(self, frame_idx: int, output: SamFrameOutput) -> None:
        self.set(frame_idx, output)

    def keys(self) -> Iterable[int]:
        return self._records.keys()

    def items(self) -> Iterator[tuple[int, SamFrameOutput]]:
        for frame_idx in list(self._records.keys()):
            output = self.get(frame_idx)
            if output is not None:
                yield frame_idx, output

    def values(self) -> Iterator[SamFrameOutput]:
        for _, output in self.items():
            yield output

    def get(self, frame_idx: int, default: Optional[SamFrameOutput] = None) -> Optional[SamFrameOutput]:
        frame_idx = int(frame_idx)
        cached = self._cache.get(frame_idx)
        if cached is not None:
            self._cache.move_to_end(frame_idx)
            return self._clone_output(cached)

        record = self._records.get(frame_idx)
        if record is None:
            return default

        output = SamFrameOutput(
            obj_ids=list(record.obj_ids),
            masks=[self._load_mask(ref) for ref in record.mask_refs],
            boxes_xywh_norm=[tuple(box) for box in record.boxes_xywh_norm],
            scores=list(record.scores),
            tracker_scores=list(record.tracker_scores),
        )
        self._remember(frame_idx, output)
        return self._clone_output(output)

    def set(self, frame_idx: int, output: SamFrameOutput) -> None:
        frame_idx = int(frame_idx)
        self._delete_frame_files(frame_idx)

        obj_ids = tuple(int(obj_id) for obj_id in output.obj_ids)
        boxes = tuple(
            tuple(map(float, output.boxes_xywh_norm[idx]))
            if idx < len(output.boxes_xywh_norm)
            else (0.0, 0.0, 0.0, 0.0)
            for idx in range(len(obj_ids))
        )
        scores = tuple(
            float(output.scores[idx]) if idx < len(output.scores) else 0.0
            for idx in range(len(obj_ids))
        )
        tracker_scores = tuple(
            float(output.tracker_scores[idx]) if idx < len(output.tracker_scores) else 0.0
            for idx in range(len(obj_ids))
        )
        mask_refs: list[_MaskRef] = []
        stored_masks: list[np.ndarray] = []
        for idx in range(len(obj_ids)):
            mask = output.masks[idx] if idx < len(output.masks) else np.zeros((1, 1), dtype=bool)
            stored_masks.append(np.asarray(mask, dtype=bool).copy())
            mask_refs.append(self._write_mask(frame_idx, obj_ids[idx], idx, mask))

        self._records[frame_idx] = _FrameOutputRecord(
            obj_ids=obj_ids,
            mask_refs=tuple(mask_refs),
            boxes_xywh_norm=boxes,
            scores=scores,
            tracker_scores=tracker_scores,
        )
        self._remember(
            frame_idx,
            SamFrameOutput(
                obj_ids=list(obj_ids),
                masks=stored_masks,
                boxes_xywh_norm=list(boxes),
                scores=list(scores),
                tracker_scores=list(tracker_scores),
            ),
        )

    def pop(self, frame_idx: int, default: Optional[SamFrameOutput] = None) -> Optional[SamFrameOutput]:
        frame_idx = int(frame_idx)
        if frame_idx not in self._records:
            return default
        output = self.get(frame_idx)
        self._records.pop(frame_idx, None)
        self._cache.pop(frame_idx, None)
        self._delete_frame_files(frame_idx)
        return output

    def remove_object(self, frame_idx: int, obj_id: int) -> bool:
        """Remove one object's output from a frame without loading unrelated masks."""
        frame_idx = int(frame_idx)
        obj_id = int(obj_id)
        record = self._records.get(frame_idx)
        if record is None or obj_id not in record.obj_ids:
            return False

        keep_idx = [idx for idx, existing_obj_id in enumerate(record.obj_ids) if existing_obj_id != obj_id]
        remove_idx = [idx for idx, existing_obj_id in enumerate(record.obj_ids) if existing_obj_id == obj_id]
        for idx in remove_idx:
            if idx < len(record.mask_refs):
                record.mask_refs[idx].path.unlink(missing_ok=True)

        if not keep_idx:
            self._records.pop(frame_idx, None)
            self._cache.pop(frame_idx, None)
            return True

        self._records[frame_idx] = _FrameOutputRecord(
            obj_ids=tuple(record.obj_ids[idx] for idx in keep_idx),
            mask_refs=tuple(record.mask_refs[idx] for idx in keep_idx),
            boxes_xywh_norm=tuple(record.boxes_xywh_norm[idx] for idx in keep_idx),
            scores=tuple(record.scores[idx] for idx in keep_idx),
            tracker_scores=tuple(record.tracker_scores[idx] for idx in keep_idx),
        )
        cached = self._cache.get(frame_idx)
        if cached is not None:
            self._cache[frame_idx] = SamFrameOutput(
                obj_ids=[cached.obj_ids[idx] for idx in keep_idx],
                masks=[cached.masks[idx] for idx in keep_idx if idx < len(cached.masks)],
                boxes_xywh_norm=[
                    cached.boxes_xywh_norm[idx] if idx < len(cached.boxes_xywh_norm) else (0.0, 0.0, 0.0, 0.0)
                    for idx in keep_idx
                ],
                scores=[cached.scores[idx] if idx < len(cached.scores) else 0.0 for idx in keep_idx],
                tracker_scores=[
                    cached.tracker_scores[idx] if idx < len(cached.tracker_scores) else 0.0
                    for idx in keep_idx
                ],
            )
            self._cache.move_to_end(frame_idx)
        return True

    def clear(self) -> None:
        self._records.clear()
        self._cache.clear()
        if self._cache_dir.exists():
            shutil.rmtree(self._cache_dir, ignore_errors=True)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        self._records.clear()
        self._cache.clear()
        if self._owned_cache_dir:
            shutil.rmtree(self._cache_dir, ignore_errors=True)

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @property
    def cached_frame_count(self) -> int:
        return len(self._cache)

    def _remember(self, frame_idx: int, output: SamFrameOutput) -> None:
        self._cache[frame_idx] = self._clone_output(output)
        self._cache.move_to_end(frame_idx)
        while len(self._cache) > self._max_cached_frames:
            self._cache.popitem(last=False)

    def _frame_glob(self, frame_idx: int) -> str:
        return f"frame_{int(frame_idx):06d}_*.npz"

    def _delete_frame_files(self, frame_idx: int) -> None:
        for path in self._cache_dir.glob(self._frame_glob(frame_idx)):
            path.unlink(missing_ok=True)

    def _mask_path(self, frame_idx: int, obj_id: int, mask_idx: int) -> Path:
        return self._cache_dir / f"frame_{frame_idx:06d}_mask_{mask_idx:03d}_obj_{obj_id}.npz"

    def _write_mask(self, frame_idx: int, obj_id: int, mask_idx: int, mask: np.ndarray) -> _MaskRef:
        mask_array = np.asarray(mask, dtype=bool)
        shape = tuple(int(dim) for dim in mask_array.shape)
        packed = np.packbits(mask_array.reshape(-1))
        path = self._mask_path(frame_idx, obj_id, mask_idx)
        np.savez_compressed(path, packed=packed, shape=np.asarray(shape, dtype=np.int64))
        return _MaskRef(path=path, shape=shape)

    def _load_mask(self, ref: _MaskRef) -> np.ndarray:
        with np.load(ref.path, allow_pickle=False) as data:
            packed = np.asarray(data["packed"], dtype=np.uint8)
            shape = tuple(int(dim) for dim in data["shape"])
        size = int(np.prod(shape, dtype=np.int64))
        return np.unpackbits(packed, count=size).astype(bool).reshape(shape)

    def _clone_output(self, output: SamFrameOutput) -> SamFrameOutput:
        return SamFrameOutput(
            obj_ids=list(output.obj_ids),
            masks=[np.asarray(mask, dtype=bool).copy() for mask in output.masks],
            boxes_xywh_norm=[tuple(map(float, box)) for box in output.boxes_xywh_norm],
            scores=[float(score) for score in output.scores],
            tracker_scores=[float(score) for score in output.tracker_scores],
        )
