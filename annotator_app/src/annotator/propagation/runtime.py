"""Typed runtime state used by propagation and prefetch workflows.

These records replace the old loose dict and boolean clusters that were spread
through the main window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


TASK_KIND_PREFETCH = "prefetch"
TASK_KIND_PREFETCH_WAIT = "prefetch_wait"
TASK_KIND_MANUAL_PROPAGATION = "manual"
TASK_KIND_AUTO_PROPAGATION = "auto"


@dataclass
class SamTaskContext:
    """Metadata tracked for an in-flight SAM worker task."""

    kind: str = ""
    seed_frame_idx: Optional[int] = None
    target_frame_idx: Optional[int] = None
    cache_generation: Optional[int] = None
    last_emitted_frame_idx: Optional[int] = None
    n_frames: Optional[int] = None

    def get(self, key: str, default=None):
        """Provide dict-like compatibility for legacy call sites."""
        if key == "version":
            key = "cache_generation"
        return getattr(self, key, default)

    def __getitem__(self, key: str):
        """Provide dict-style compatibility for legacy task-context lookups."""
        if key == "version":
            key = "cache_generation"
        return getattr(self, key)

    def __setitem__(self, key: str, value) -> None:
        """Provide dict-style compatibility for legacy task-context mutation."""
        if key == "version":
            key = "cache_generation"
        setattr(self, key, value)


@dataclass
class PrefetchProvenance:
    """Records which seed frame and objects produced a cached frame result."""

    seed_frame_idx: int
    obj_ids: List[int]

    def get(self, key: str, default=None):
        """Provide dict-style compatibility for legacy provenance lookups."""
        return getattr(self, key, default)

    def __getitem__(self, key: str):
        """Provide dict-style compatibility for legacy provenance indexing."""
        return getattr(self, key)

    def __setitem__(self, key: str, value) -> None:
        """Provide dict-style compatibility for legacy provenance mutation."""
        setattr(self, key, value)


@dataclass
class PrefetchState:
    """Explicit state for next-frame tracker prefetching."""

    busy: bool = False
    cancel_requested: bool = False
    pending_restart: bool = False
    target_frame_idx: Optional[int] = None
    seed_frame_idx: Optional[int] = None
    cache_generation: int = 0
    active_generation: Optional[int] = None
    cached_frame_idx: Optional[int] = None
    cached_seed_idx: Optional[int] = None
    cached_generation: Optional[int] = None
    provenance_by_frame: Dict[int, PrefetchProvenance] = field(default_factory=dict)

    def clear_cache(self) -> None:
        """Drop cached outputs while leaving generation counters intact."""
        self.cached_frame_idx = None
        self.cached_seed_idx = None
        self.cached_generation = None
        self.active_generation = None
        self.provenance_by_frame.clear()

    def reset(self) -> None:
        """Return prefetch state to its initial empty condition."""
        self.busy = False
        self.cancel_requested = False
        self.pending_restart = False
        self.target_frame_idx = None
        self.seed_frame_idx = None
        self.cache_generation = 0
        self.clear_cache()

    def note_prompt_change(self) -> None:
        """Invalidate cache generation after any prompt edit on the seed frame."""
        self.cache_generation += 1
        self.clear_cache()

    def has_valid_cache(self, target_frame_idx: int) -> bool:
        """Check whether the cached frame still matches the current prompt generation."""
        return (
            self.cached_frame_idx == target_frame_idx
            and self.cached_seed_idx == target_frame_idx - 1
            and self.cached_generation == self.cache_generation
        )

    def begin(self, *, seed_frame_idx: int, target_frame_idx: int) -> None:
        """Mark a new prefetch run as active."""
        self.busy = True
        self.cancel_requested = False
        self.pending_restart = False
        self.seed_frame_idx = seed_frame_idx
        self.target_frame_idx = target_frame_idx
        self.active_generation = self.cache_generation

    def finish(self) -> None:
        """Clear active-run markers after prefetch completion or cancellation."""
        self.busy = False
        self.cancel_requested = False
        self.pending_restart = False
        self.target_frame_idx = None
        self.seed_frame_idx = None
        self.active_generation = None

    def record_cached_frame(self, frame_idx: int, obj_ids: List[int]) -> None:
        """Store cache bookkeeping for a successfully prefetched frame."""
        self.cached_frame_idx = frame_idx
        self.cached_seed_idx = self.seed_frame_idx
        self.cached_generation = self.active_generation
        self.provenance_by_frame[frame_idx] = PrefetchProvenance(
            seed_frame_idx=int(self.seed_frame_idx or 0),
            obj_ids=[int(obj_id) for obj_id in obj_ids],
        )


@dataclass
class PropagationRuntimeState:
    """Explicit state for the currently running propagation workflow."""

    busy: bool = False
    enabled_obj_ids: Optional[Set[int]] = None
    view_frame_idx: Optional[int] = None
    continue_after_chunk: bool = False
    active_chunk_idx: Optional[int] = None
    active_seed_frame_idx: Optional[int] = None
    task_id: Optional[str] = None
    stop_requested: bool = False

    def begin_run(self, *, enabled_obj_ids: Set[int], view_frame_idx: int) -> None:
        """Initialize active object scope for a propagation run."""
        self.enabled_obj_ids = set(enabled_obj_ids)
        self.view_frame_idx = int(view_frame_idx)
        self.stop_requested = False

    def begin_chunk(self, *, chunk_idx: int, seed_frame_idx: int, task_id: str) -> None:
        """Mark one propagation chunk as in flight."""
        self.busy = True
        self.active_chunk_idx = int(chunk_idx)
        self.active_seed_frame_idx = int(seed_frame_idx)
        self.task_id = task_id

    def finish_chunk(self) -> None:
        """Clear per-chunk markers while preserving broader run state."""
        self.busy = False
        self.active_chunk_idx = None
        self.active_seed_frame_idx = None
        self.task_id = None

    def reset(self) -> None:
        """Reset both chunk-local and run-wide propagation state."""
        self.finish_chunk()
        self.enabled_obj_ids = None
        self.view_frame_idx = None
        self.continue_after_chunk = False
        self.stop_requested = False
