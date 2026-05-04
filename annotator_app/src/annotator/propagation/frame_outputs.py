"""Pure frame-output helpers used by propagation and editing flows."""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

from annotator.models import BoxPrompt, SamFrameOutput


def output_object_has_mask(output: SamFrameOutput, obj_id: int) -> bool:
    """Return whether one object has a present, non-empty mask in a frame output."""
    if obj_id not in output.obj_ids:
        return False
    idx = output.obj_ids.index(obj_id)
    if idx >= len(output.masks):
        return False
    return bool(np.asarray(output.masks[idx]).any())


def find_disappeared_object_ids(
    output: SamFrameOutput,
    enabled_obj_ids: set[int],
) -> list[int]:
    """Return enabled object ids missing from a frame output or represented by empty masks."""
    return [
        int(obj_id)
        for obj_id in sorted(enabled_obj_ids)
        if not output_object_has_mask(output, int(obj_id))
    ]


def recovery_seed_frame_idx(
    *,
    disappear_frame_idx: int,
    run_start_frame_idx: int,
    lookback_frames: int = 3,
) -> int:
    """Choose the restart seed frame for a disappearance recovery attempt."""
    return max(int(run_start_frame_idx), int(disappear_frame_idx) - int(lookback_frames))


def target_limited_chunk_size(
    *,
    seed_frame_idx: int,
    target_frame_idx: Optional[int],
    requested_chunk_size: int,
) -> int:
    """Cap a chunk size so tracker propagation does not run past the target frame."""
    chunk_size = int(requested_chunk_size)
    if target_frame_idx is None:
        return chunk_size
    return min(chunk_size, int(target_frame_idx) - int(seed_frame_idx) + 1)


def merge_frame_outputs(base_output: SamFrameOutput, new_output: SamFrameOutput) -> SamFrameOutput:
    """Replace overlapping object outputs and append new objects into one frame result."""
    merged = SamFrameOutput(
        obj_ids=list(base_output.obj_ids),
        masks=list(base_output.masks),
        boxes_xywh_norm=list(base_output.boxes_xywh_norm),
        scores=list(base_output.scores),
        tracker_scores=list(base_output.tracker_scores),
    )
    obj_to_idx = {obj_id: i for i, obj_id in enumerate(merged.obj_ids)}
    for idx, obj_id in enumerate(new_output.obj_ids):
        if idx >= len(new_output.masks) or idx >= len(new_output.boxes_xywh_norm):
            continue
        score = new_output.scores[idx] if idx < len(new_output.scores) else 0.0
        tracker_score = new_output.tracker_scores[idx] if idx < len(new_output.tracker_scores) else 0.0
        if obj_id in obj_to_idx:
            existing_idx = obj_to_idx[obj_id]
            merged.masks[existing_idx] = new_output.masks[idx]
            merged.boxes_xywh_norm[existing_idx] = new_output.boxes_xywh_norm[idx]
            merged.scores[existing_idx] = score
            merged.tracker_scores[existing_idx] = tracker_score
            continue
        merged.obj_ids.append(obj_id)
        merged.masks.append(new_output.masks[idx])
        merged.boxes_xywh_norm.append(new_output.boxes_xywh_norm[idx])
        merged.scores.append(score)
        merged.tracker_scores.append(tracker_score)
        obj_to_idx[obj_id] = len(merged.obj_ids) - 1
    return merged


def remove_object_from_output(
    output: Optional[SamFrameOutput],
    obj_id: Optional[int],
) -> Optional[SamFrameOutput]:
    """Remove one object's output from a frame, returning `None` if nothing remains."""
    if output is None or obj_id is None:
        return output

    keep_idx = [idx for idx, existing_obj_id in enumerate(output.obj_ids) if existing_obj_id != obj_id]
    if len(keep_idx) == len(output.obj_ids):
        return output
    if not keep_idx:
        return None
    return SamFrameOutput(
        obj_ids=[output.obj_ids[idx] for idx in keep_idx],
        masks=[output.masks[idx] for idx in keep_idx],
        boxes_xywh_norm=[output.boxes_xywh_norm[idx] for idx in keep_idx],
        scores=[output.scores[idx] for idx in keep_idx],
        tracker_scores=[output.tracker_scores[idx] for idx in keep_idx],
    )


def sample_boxes_from_output_masks(
    output: Optional[SamFrameOutput],
    image_h: int,
    image_w: int,
) -> Dict[int, BoxPrompt]:
    """Approximate canonical boxes from output masks when no explicit box is available."""
    sampled_boxes: Dict[int, BoxPrompt] = {}
    if output is None:
        return sampled_boxes

    for idx, obj_id in enumerate(output.obj_ids):
        if idx >= len(output.masks):
            continue
        mask = np.asarray(output.masks[idx]).astype(np.uint8)
        if mask.shape[:2] != (image_h, image_w):
            mask = cv2.resize(mask, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            continue
        sampled_boxes[obj_id] = BoxPrompt(
            x1_px=int(xs.min()),
            y1_px=int(ys.min()),
            x2_px=int(xs.max()),
            y2_px=int(ys.max()),
        )
    return sampled_boxes
