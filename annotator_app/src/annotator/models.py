"""Shared annotator data models, defaults, and propagation mode constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

PROPAGATION_MODE_TRACKER = "tracker"
PROPAGATION_MODE_COPY_BOXES = "copy_boxes"
VALID_PROPAGATION_MODES = {
    PROPAGATION_MODE_TRACKER,
    PROPAGATION_MODE_COPY_BOXES,
}
DEFAULT_PROPAGATION_CHUNK_SIZE = 2
DEFAULT_PROPAGATION_CHUNKS = 1
DEFAULT_SEGMENTATION_OPACITY = 0.6
DEFAULT_BOX_LINE_THICKNESS = 2
DEFAULT_RECONDITION_EVERY_NTH_FRAME = 16
DEFAULT_RECONDITION_HIGH_CONF_THRESH = 0.8
DEFAULT_RECONDITION_HIGH_IOU_THRESH = 0.8
DEFAULT_SMART_PROPAGATION_REWIND_FRAMES = 3
DEFAULT_SMART_PROPAGATION_RECOVERY_CHUNK_SIZE = 50


def normalize_propagation_mode(value: object) -> str:
    """Coerce persisted or UI-provided propagation mode values to a supported constant."""
    mode = str(value) if value is not None else PROPAGATION_MODE_TRACKER
    if mode in VALID_PROPAGATION_MODES:
        return mode
    return PROPAGATION_MODE_TRACKER


@dataclass
class PointPrompt:
    """One positive or negative click prompt stored in frame pixel coordinates."""

    x_px: int
    y_px: int
    is_positive: bool


@dataclass
class BoxPrompt:
    """One rectangular prompt stored in frame pixel coordinates."""

    x1_px: int
    y1_px: int
    x2_px: int
    y2_px: int


@dataclass
class ObjectInfo:
    """User-defined object metadata shared across prompts, outputs, and exports."""

    obj_id: int
    name: str
    color_bgr: Tuple[int, int, int]


@dataclass
class SamFrameOutput:
    """Normalized per-frame SAM output for one or more tracked objects."""

    obj_ids: List[int]
    masks: List[np.ndarray]
    boxes_xywh_norm: List[Tuple[float, float, float, float]]
    scores: List[float]
    tracker_scores: List[float]


@dataclass
class ViewSettings:
    """UI toggles that affect only rendering, not annotation content."""

    show_prompts: bool = True
    show_segmentations: bool = True
    show_boxes: bool = True
    show_box_titles: bool = True
    segmentation_opacity: float = DEFAULT_SEGMENTATION_OPACITY
    box_line_thickness: int = DEFAULT_BOX_LINE_THICKNESS


@dataclass
class PropagationSettings:
    """Serializable tracker/copy-box propagation controls captured in a session."""

    mode: str = PROPAGATION_MODE_TRACKER
    chunk_size: int = DEFAULT_PROPAGATION_CHUNK_SIZE
    chunks: int = DEFAULT_PROPAGATION_CHUNKS
    use_target_frame: bool = False
    target_frame_idx: int = 0
    translate_prompts: bool = True
    use_point_prompts: bool = True
    auto_propagate_next: bool = False

    def __post_init__(self) -> None:
        """Normalize mode values after construction so persisted legacy values still load."""
        self.mode = normalize_propagation_mode(self.mode)


@dataclass
class ExperimentalSettings:
    """Tracker tuning and recovery options that are surfaced as advanced controls."""

    recondition_every_nth_frame: int = DEFAULT_RECONDITION_EVERY_NTH_FRAME
    recondition_high_conf_thresh: float = DEFAULT_RECONDITION_HIGH_CONF_THRESH
    recondition_high_iou_thresh: float = DEFAULT_RECONDITION_HIGH_IOU_THRESH
    use_one_session_chunked_propagation: bool = False
    smart_propagation_enabled: bool = False
    smart_propagation_rewind_frames: int = DEFAULT_SMART_PROPAGATION_REWIND_FRAMES
    smart_propagation_recovery_chunk_size: int = DEFAULT_SMART_PROPAGATION_RECOVERY_CHUNK_SIZE


@dataclass
class PendingPropagationState:
    """Mutable state for a multi-chunk propagation run that is currently in progress."""

    remaining_chunks: int
    next_seed_frame_idx: int
    n_frames: int
    total_chunks: int
    target_frame_idx: Optional[int] = None
    completed_chunks: int = 0
    run_start_frame_idx: int = 0
    active_chunk_n_frames: Optional[int] = None
    pending_smart_restart_seed_frame_idx: Optional[int] = None
    pending_smart_restart_loss_frame_idx: Optional[int] = None
    smart_restart_cooldown_until_frame_idx: Optional[int] = None
