"""Translation between typed session payloads and persisted JSON data.

This module is the single place that knows the on-disk session schema.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from annotator.models import (
    BoxPrompt,
    ObjectInfo,
    PointPrompt,
    PropagationSettings,
    SamFrameOutput,
    ViewSettings,
    normalize_propagation_mode,
)
from annotator.persistence.session_models import (
    SESSION_SCHEMA_VERSION,
    SessionMaskAsset,
    SessionPayload,
)


def _normalized_fallback_shape(shape: Tuple[int, int]) -> Tuple[int, int]:
    """Clamp fallback image shapes so blank masks are always valid arrays."""
    height = max(1, int(shape[0]))
    width = max(1, int(shape[1]))
    return (height, width)


def payload_to_data(payload: SessionPayload) -> tuple[dict[str, object], list[SessionMaskAsset]]:
    """Serialize the typed session payload into JSON data plus external mask assets."""
    data: dict[str, object] = {
        "version": int(payload.version),
        "frame_dir": payload.frame_dir,
        "frame_files": list(payload.frame_files),
        "current_frame_idx": int(payload.current_frame_idx),
        "active_object_id": int(payload.active_object_id) if payload.active_object_id is not None else None,
        "checkpoint_path": payload.checkpoint_path,
        "flagged_frames": [int(frame_idx) for frame_idx in payload.flagged_frames],
        "objects": [
            {
                "obj_id": int(obj.obj_id),
                "name": obj.name,
                "color_bgr": [int(channel) for channel in obj.color_bgr],
            }
            for obj in payload.objects
        ],
        "object_view": {
            "hidden_obj_ids": [int(obj_id) for obj_id in payload.hidden_obj_ids],
            "solo_object_id": int(payload.solo_object_id) if payload.solo_object_id is not None else None,
        },
        "prompts": {},
        "boxes": {},
        "box_locks": {},
        "propagation_overrides": {},
        "outputs": {},
        "view": {
            "show_prompts": bool(payload.view_settings.show_prompts),
            "show_segmentations": bool(payload.view_settings.show_segmentations),
            "show_boxes": bool(payload.view_settings.show_boxes),
            "show_box_titles": bool(payload.view_settings.show_box_titles),
            "segmentation_opacity": float(payload.view_settings.segmentation_opacity),
            "box_line_thickness": int(payload.view_settings.box_line_thickness),
        },
        "propagation": {
            "mode": normalize_propagation_mode(payload.propagation_settings.mode),
            "n_frames": int(payload.propagation_settings.chunk_size),
            "chunks": int(payload.propagation_settings.chunks),
            "use_target_frame": bool(payload.propagation_settings.use_target_frame),
            "target_frame_idx": int(payload.propagation_settings.target_frame_idx),
            "translate_prompts": bool(payload.propagation_settings.translate_prompts),
            "use_point_prompts": bool(payload.propagation_settings.use_point_prompts),
            "auto_propagate_next": bool(payload.propagation_settings.auto_propagate_next),
        },
        "prompt_mode_index": int(payload.prompt_mode_index),
    }

    prompts = data["prompts"]
    for frame_idx, per_obj in payload.prompts_by_frame_obj.items():
        prompts[str(frame_idx)] = {
            str(obj_id): [
                {
                    "x_px": int(point.x_px),
                    "y_px": int(point.y_px),
                    "is_positive": bool(point.is_positive),
                }
                for point in point_list
            ]
            for obj_id, point_list in per_obj.items()
        }

    boxes = data["boxes"]
    for frame_idx, per_obj in payload.boxes_by_frame_obj.items():
        boxes[str(frame_idx)] = {
            str(obj_id): {
                "x1_px": int(box.x1_px),
                "y1_px": int(box.y1_px),
                "x2_px": int(box.x2_px),
                "y2_px": int(box.y2_px),
            }
            for obj_id, box in per_obj.items()
        }

    box_locks = data["box_locks"]
    for frame_idx, per_obj in payload.box_locks_by_frame_obj.items():
        box_locks[str(frame_idx)] = {
            str(obj_id): bool(locked)
            for obj_id, locked in per_obj.items()
        }

    propagation_overrides = data["propagation_overrides"]
    for frame_idx, per_obj in payload.propagation_overrides_by_frame_obj.items():
        if not per_obj:
            continue
        propagation_overrides[str(frame_idx)] = {
            str(obj_id): bool(enabled)
            for obj_id, enabled in per_obj.items()
        }

    outputs = data["outputs"]
    mask_assets: list[SessionMaskAsset] = []
    for frame_idx, output in payload.outputs_by_frame.items():
        entry = {
            "obj_ids": [int(obj_id) for obj_id in output.obj_ids],
            "boxes_xywh_norm": [list(map(float, box)) for box in output.boxes_xywh_norm],
            "scores": [float(score) for score in output.scores],
            "tracker_scores": [float(score) for score in output.tracker_scores],
            "mask_paths": [],
        }
        for idx, obj_id in enumerate(output.obj_ids):
            mask = output.masks[idx] if idx < len(output.masks) else None
            if mask is None:
                entry["mask_paths"].append(None)
                continue
            mask_array = np.asarray(mask)
            if mask_array.size == 0 or not mask_array.any():
                entry["mask_paths"].append(None)
                continue
            relative_path = f"masks/frame_{int(frame_idx):05d}_obj_{int(obj_id)}.png"
            mask_assets.append(SessionMaskAsset(relative_path=relative_path, mask=mask_array))
            entry["mask_paths"].append(relative_path)
        outputs[str(frame_idx)] = entry

    return data, mask_assets


def data_to_payload(
    data: Mapping[str, object],
    *,
    load_mask: Callable[[str], np.ndarray],
    fallback_shape_for_frame: Callable[[int], Tuple[int, int]],
    on_output_loaded: Optional[Callable[[int, int], None]] = None,
) -> SessionPayload:
    """Parse persisted session JSON and mask assets into the typed payload model."""
    prompts_by_frame_obj: Dict[int, Dict[int, List[PointPrompt]]] = {}
    for frame_idx, per_obj in dict(data.get("prompts", {})).items():
        frame_map: Dict[int, List[PointPrompt]] = {}
        for obj_id, point_list in dict(per_obj).items():
            frame_map[int(obj_id)] = [
                PointPrompt(
                    x_px=int(point["x_px"]),
                    y_px=int(point["y_px"]),
                    is_positive=bool(point["is_positive"]),
                )
                for point in list(point_list)
            ]
        prompts_by_frame_obj[int(frame_idx)] = frame_map

    boxes_by_frame_obj: Dict[int, Dict[int, BoxPrompt]] = {}
    for frame_idx, per_obj in dict(data.get("boxes", {})).items():
        frame_map: Dict[int, BoxPrompt] = {}
        for obj_id, box in dict(per_obj).items():
            frame_map[int(obj_id)] = BoxPrompt(
                x1_px=int(box["x1_px"]),
                y1_px=int(box["y1_px"]),
                x2_px=int(box["x2_px"]),
                y2_px=int(box["y2_px"]),
            )
        boxes_by_frame_obj[int(frame_idx)] = frame_map

    box_locks_by_frame_obj: Dict[int, Dict[int, bool]] = {}
    for frame_idx, per_obj in dict(data.get("box_locks", {})).items():
        box_locks_by_frame_obj[int(frame_idx)] = {
            int(obj_id): bool(locked)
            for obj_id, locked in dict(per_obj).items()
        }

    propagation_overrides_by_frame_obj: Dict[int, Dict[int, bool]] = {}
    for frame_idx, per_obj in dict(data.get("propagation_overrides", {})).items():
        overrides = {
            int(obj_id): bool(enabled)
            for obj_id, enabled in dict(per_obj).items()
        }
        if overrides:
            propagation_overrides_by_frame_obj[int(frame_idx)] = overrides

    outputs_by_frame: Dict[int, SamFrameOutput] = {}
    output_items = list(dict(data.get("outputs", {})).items())
    total_outputs = max(1, len(output_items))
    for output_idx, (frame_idx, entry) in enumerate(output_items, start=1):
        frame_idx_int = int(frame_idx)
        obj_ids = [int(obj_id) for obj_id in list(entry.get("obj_ids", []))]
        boxes = [tuple(map(float, box)) for box in list(entry.get("boxes_xywh_norm", []))]
        scores = [float(score) for score in list(entry.get("scores", []))]
        tracker_scores = [float(score) for score in list(entry.get("tracker_scores", []))]
        if len(tracker_scores) < len(obj_ids):
            tracker_scores.extend([0.0] * (len(obj_ids) - len(tracker_scores)))
        mask_paths = list(entry.get("mask_paths", []))
        fallback_shape: Optional[Tuple[int, int]] = None

        def blank_mask() -> np.ndarray:
            """Build a valid empty mask when the saved mask asset is missing or unreadable."""
            nonlocal fallback_shape
            if fallback_shape is None:
                fallback_shape = _normalized_fallback_shape(fallback_shape_for_frame(frame_idx_int))
            return np.zeros(fallback_shape, dtype=bool)

        masks: list[np.ndarray] = []
        for idx in range(len(obj_ids)):
            path = mask_paths[idx] if idx < len(mask_paths) else None
            if path is None:
                masks.append(blank_mask())
                continue
            try:
                masks.append(np.asarray(load_mask(str(path)), dtype=bool))
            except Exception:
                masks.append(blank_mask())
        outputs_by_frame[frame_idx_int] = SamFrameOutput(
            obj_ids=obj_ids,
            masks=masks,
            boxes_xywh_norm=boxes,
            scores=scores,
            tracker_scores=tracker_scores,
        )
        if on_output_loaded is not None:
            on_output_loaded(output_idx, total_outputs)

    view = dict(data.get("view", {}))
    propagation = dict(data.get("propagation", {}))
    object_view = dict(data.get("object_view", {}))

    return SessionPayload(
        version=int(data.get("version", SESSION_SCHEMA_VERSION)),
        frame_dir=str(data.get("frame_dir", "")),
        frame_files=[str(name) for name in list(data.get("frame_files", []))],
        current_frame_idx=int(data.get("current_frame_idx", 0)),
        active_object_id=(
            int(data["active_object_id"])
            if data.get("active_object_id") is not None
            else None
        ),
        checkpoint_path=(
            str(data["checkpoint_path"])
            if data.get("checkpoint_path") not in (None, "")
            else None
        ),
        flagged_frames=[int(frame_idx) for frame_idx in list(data.get("flagged_frames", []))],
        objects=[
            ObjectInfo(
                obj_id=int(entry["obj_id"]),
                name=str(entry["name"]),
                color_bgr=tuple(int(channel) for channel in entry.get("color_bgr", (255, 255, 255))),
            )
            for entry in list(data.get("objects", []))
        ],
        hidden_obj_ids=[int(obj_id) for obj_id in list(object_view.get("hidden_obj_ids", []))],
        solo_object_id=(
            int(object_view["solo_object_id"])
            if object_view.get("solo_object_id") is not None
            else None
        ),
        prompts_by_frame_obj=prompts_by_frame_obj,
        boxes_by_frame_obj=boxes_by_frame_obj,
        box_locks_by_frame_obj=box_locks_by_frame_obj,
        propagation_overrides_by_frame_obj=propagation_overrides_by_frame_obj,
        outputs_by_frame=outputs_by_frame,
        view_settings=ViewSettings(
            show_prompts=bool(view.get("show_prompts", True)),
            show_segmentations=bool(view.get("show_segmentations", True)),
            show_boxes=bool(view.get("show_boxes", True)),
            show_box_titles=bool(view.get("show_box_titles", True)),
            segmentation_opacity=float(view.get("segmentation_opacity", ViewSettings().segmentation_opacity)),
            box_line_thickness=int(view.get("box_line_thickness", ViewSettings().box_line_thickness)),
        ),
        propagation_settings=PropagationSettings(
            mode=normalize_propagation_mode(propagation.get("mode")),
            chunk_size=int(propagation.get("n_frames", PropagationSettings().chunk_size)),
            chunks=int(propagation.get("chunks", PropagationSettings().chunks)),
            use_target_frame=bool(propagation.get("use_target_frame", False)),
            target_frame_idx=int(propagation.get("target_frame_idx", 0)),
            translate_prompts=bool(propagation.get("translate_prompts", True)),
            use_point_prompts=bool(propagation.get("use_point_prompts", True)),
            auto_propagate_next=bool(propagation.get("auto_propagate_next", False)),
        ),
        prompt_mode_index=int(data.get("prompt_mode_index", 2)),
    )
