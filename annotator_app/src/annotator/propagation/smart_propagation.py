"""Pure smart-propagation restart rules used by tracker recovery logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set, Tuple


@dataclass(frozen=True)
class SmartPropagationSettings:
    """Configuration for automatic rewind-and-recover tracker restarts."""

    enabled: bool = False
    rewind_frames: int = 3
    recovery_chunk_size: int = 50


@dataclass(frozen=True)
class SmartPropagationDecision:
    """Decision payload returned by the pure smart-propagation detector."""

    should_restart: bool
    restart_seed_frame_idx: Optional[int] = None
    loss_frame_idx: Optional[int] = None
    lost_obj_ids: Tuple[int, ...] = ()


def detect_smart_propagation_restart(
    *,
    enabled_obj_ids: Iterable[int],
    has_mask_by_obj_id: Dict[int, bool],
    previously_seen_obj_ids: Set[int],
    already_triggered_loss_keys: Set[Tuple[int, int]],
    loss_frame_idx: int,
    run_start_frame_idx: int,
    rewind_frames: int,
) -> SmartPropagationDecision:
    """Detect whether previously tracked objects disappeared and warrant a recovery rewind."""
    lost_obj_ids = []
    for obj_id in sorted(set(int(obj_id) for obj_id in enabled_obj_ids)):
        if obj_id not in previously_seen_obj_ids:
            continue
        if has_mask_by_obj_id.get(obj_id, False):
            continue
        loss_key = (obj_id, loss_frame_idx)
        if loss_key in already_triggered_loss_keys:
            continue
        lost_obj_ids.append(obj_id)

    if not lost_obj_ids:
        return SmartPropagationDecision(should_restart=False)

    restart_seed_frame_idx = max(int(run_start_frame_idx), int(loss_frame_idx) - max(0, int(rewind_frames)))
    return SmartPropagationDecision(
        should_restart=True,
        restart_seed_frame_idx=restart_seed_frame_idx,
        loss_frame_idx=int(loss_frame_idx),
        lost_obj_ids=tuple(lost_obj_ids),
    )
