"""Typed session schema used inside the annotator.

The window builds and consumes these records; JSON conversion happens in the
mapper and repository layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from annotator.models import (
    BoxPrompt,
    ObjectInfo,
    PointPrompt,
    PropagationSettings,
    SamFrameOutput,
    ViewSettings,
)

SESSION_SCHEMA_VERSION = 6


@dataclass
class SessionMaskAsset:
    """One mask image that will be written beside the session JSON."""

    relative_path: str
    mask: np.ndarray


@dataclass
class SessionPayload:
    """Complete in-memory session payload independent of JSON layout."""

    frame_dir: str
    frame_files: List[str]
    current_frame_idx: int = 0
    active_object_id: Optional[int] = None
    checkpoint_path: Optional[str] = None
    flagged_frames: List[int] = field(default_factory=list)
    objects: List[ObjectInfo] = field(default_factory=list)
    hidden_obj_ids: List[int] = field(default_factory=list)
    solo_object_id: Optional[int] = None
    prompts_by_frame_obj: Dict[int, Dict[int, List[PointPrompt]]] = field(default_factory=dict)
    boxes_by_frame_obj: Dict[int, Dict[int, BoxPrompt]] = field(default_factory=dict)
    box_locks_by_frame_obj: Dict[int, Dict[int, bool]] = field(default_factory=dict)
    propagation_overrides_by_frame_obj: Dict[int, Dict[int, bool]] = field(default_factory=dict)
    outputs_by_frame: Dict[int, SamFrameOutput] = field(default_factory=dict)
    view_settings: ViewSettings = field(default_factory=ViewSettings)
    propagation_settings: PropagationSettings = field(default_factory=PropagationSettings)
    prompt_mode_index: int = 2
    version: int = SESSION_SCHEMA_VERSION
