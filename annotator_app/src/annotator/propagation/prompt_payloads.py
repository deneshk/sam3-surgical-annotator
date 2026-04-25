"""Pure prompt-payload helpers used by segmentation and propagation.

These functions translate app state into SAM task payloads without touching Qt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from annotator.models import BoxPrompt, PointPrompt, SamFrameOutput


@dataclass
class PromptPayloadInput:
    """Prompt payload for one object before it is converted to worker task data."""

    points_rel: List[List[float]]
    labels: List[int]
    mask_input: Optional[np.ndarray] = None

    def to_task_payload(self) -> Dict[str, object]:
        """Convert the typed prompt payload into the worker-facing dict contract."""
        return {
            "points_rel": [list(point) for point in self.points_rel],
            "labels": list(self.labels),
            "mask_input": self.mask_input,
        }


@dataclass
class PropagationSeedBuildResult:
    """Result of building propagation seed prompts for a chunk."""

    payload_by_obj_id: Dict[int, PromptPayloadInput]
    missing_obj_ids: List[int]


def clone_point_prompts(points: Sequence[PointPrompt]) -> List[PointPrompt]:
    """Copy point prompts so downstream edits never mutate source state."""
    return [
        PointPrompt(x_px=point.x_px, y_px=point.y_px, is_positive=point.is_positive)
        for point in points
    ]


def build_segment_prompt_payload(
    *,
    frame_prompts: Mapping[int, Sequence[PointPrompt]],
    frame_boxes: Mapping[int, BoxPrompt],
    obj_ids: AbstractSet[int],
    frame_size: Tuple[int, int],
) -> Dict[int, PromptPayloadInput]:
    """Build per-object SAM prompts for a single-frame segmentation run."""
    width, height = frame_size
    payload: Dict[int, PromptPayloadInput] = {}
    for obj_id in sorted(obj_ids):
        points = frame_prompts.get(obj_id, [])
        points_rel = [[point.x_px / width, point.y_px / height] for point in points]
        labels = [1 if point.is_positive else 0 for point in points]
        box = frame_boxes.get(obj_id)
        if box is not None:
            points_rel.append([box.x1_px / width, box.y1_px / height])
            points_rel.append([box.x2_px / width, box.y2_px / height])
            labels.append(2)
            labels.append(3)
        if not points_rel:
            continue
        payload[obj_id] = PromptPayloadInput(points_rel=points_rel, labels=labels)
    return payload


def build_propagation_seed_payload(
    *,
    frame_prompts: Mapping[int, Sequence[PointPrompt]],
    frame_boxes: Mapping[int, BoxPrompt],
    seed_output: Optional[SamFrameOutput],
    enabled_obj_ids: AbstractSet[int],
    frame_size: Tuple[int, int],
    use_point_prompts: bool,
    sampled_boxes: Mapping[int, BoxPrompt],
) -> PropagationSeedBuildResult:
    """Build seed prompts, sampled boxes, and mask inputs for tracker propagation."""
    width, height = frame_size
    boxes_to_apply: Dict[int, BoxPrompt] = {}
    masks_to_apply: Dict[int, np.ndarray] = {}
    points_to_apply: Dict[int, List[PointPrompt]] = {}

    for obj_id in sorted(enabled_obj_ids):
        canonical_box = frame_boxes.get(obj_id)
        if canonical_box is not None:
            boxes_to_apply[obj_id] = canonical_box
        manual_points = clone_point_prompts(frame_prompts.get(obj_id, [])) if use_point_prompts else []
        if manual_points:
            points_to_apply[obj_id] = manual_points

    if seed_output is not None:
        for idx, obj_id in enumerate(seed_output.obj_ids):
            if obj_id in enabled_obj_ids and idx < len(seed_output.masks):
                masks_to_apply[obj_id] = np.asarray(seed_output.masks[idx]).astype(np.float32)

    for obj_id in sorted(enabled_obj_ids):
        if obj_id not in boxes_to_apply and obj_id in sampled_boxes:
            boxes_to_apply[obj_id] = sampled_boxes[obj_id]

    missing_obj_ids: List[int] = []
    payload: Dict[int, PromptPayloadInput] = {}
    for obj_id in sorted(enabled_obj_ids):
        box = boxes_to_apply.get(obj_id)
        mask_input = masks_to_apply.get(obj_id)
        points = points_to_apply.get(obj_id, [])
        if box is None and mask_input is None and not points:
            missing_obj_ids.append(obj_id)
            continue

        points_rel: List[List[float]] = []
        labels: List[int] = []
        if box is not None:
            points_rel.append([box.x1_px / width, box.y1_px / height])
            points_rel.append([box.x2_px / width, box.y2_px / height])
            labels.append(2)
            labels.append(3)
        for point in points:
            points_rel.append([point.x_px / width, point.y_px / height])
            labels.append(1 if point.is_positive else 0)

        payload[obj_id] = PromptPayloadInput(
            points_rel=points_rel,
            labels=labels,
            mask_input=mask_input,
        )

    return PropagationSeedBuildResult(
        payload_by_obj_id=payload,
        missing_obj_ids=missing_obj_ids,
    )


def prompt_payload_inputs_to_task_payload(
    payload_by_obj_id: Mapping[int, PromptPayloadInput],
) -> Dict[int, Dict[str, object]]:
    """Convert typed prompt payloads into the raw worker contract."""
    return {
        int(obj_id): payload.to_task_payload()
        for obj_id, payload in payload_by_obj_id.items()
    }


def translate_prompts_by_box_delta(
    *,
    points: Sequence[PointPrompt],
    src_box: Optional[BoxPrompt],
    dst_box: Optional[BoxPrompt],
    frame_size: Optional[Tuple[int, int]],
) -> List[PointPrompt]:
    """Translate point prompts using the center delta between source and target boxes."""
    if src_box is None or dst_box is None or frame_size is None:
        return clone_point_prompts(points)

    width, height = frame_size
    src_cx = (src_box.x1_px + src_box.x2_px) / 2.0
    src_cy = (src_box.y1_px + src_box.y2_px) / 2.0
    dst_cx = (dst_box.x1_px + dst_box.x2_px) / 2.0
    dst_cy = (dst_box.y1_px + dst_box.y2_px) / 2.0
    dx = dst_cx - src_cx
    dy = dst_cy - src_cy

    translated: List[PointPrompt] = []
    for point in points:
        x_px = int(round(point.x_px + dx))
        y_px = int(round(point.y_px + dy))
        x_px = max(0, min(width - 1, x_px))
        y_px = max(0, min(height - 1, y_px))
        translated.append(PointPrompt(x_px=x_px, y_px=y_px, is_positive=point.is_positive))
    return translated
