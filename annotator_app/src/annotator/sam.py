"""SAM3 runtime bridge used by the annotator UI.

This module isolates the vendored SAM3 predictor import and wraps it in a
Qt-friendly worker that executes segment, propagation, and prefetch tasks off
the main thread.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Dict, List, Optional

import numpy as np
from PIL import Image as PilImage
from PySide6.QtCore import QObject, QMetaObject, Qt, Signal, Slot

from annotator.models import SamFrameOutput
from annotator.propagation.frame_outputs import merge_frame_outputs
from annotator.vendor.sam3_runtime import ensure_vendor_sam3_on_path

ensure_vendor_sam3_on_path()

from sam3.model.sam3_video_predictor import Sam3VideoPredictor


class Sam3Adapter:
    """Thin adapter around the vendored SAM3 predictor request protocol."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        bpe_path: Optional[str] = None,
    ) -> None:
        """Construct the vendored predictor with default tracker settings."""
        self.predictor = Sam3VideoPredictor(
            checkpoint_path=checkpoint_path,
            bpe_path=bpe_path,
        )
        self.session_id: Optional[str] = None
        self.session_abs_start_frame_idx: Optional[int] = None
        self.last_memory_transfer_report: Optional[dict] = None

    def start_session(
        self,
        resource_path,
        *,
        abs_start_frame_idx: Optional[int] = None,
        transfer_memory_from_previous: bool = False,
    ) -> None:
        """Start a SAM3 session, optionally copying tracker memory from the previous one."""
        previous_session_id = self.session_id
        previous_abs_start_frame_idx = self.session_abs_start_frame_idx
        self.last_memory_transfer_report = None
        if previous_session_id and not transfer_memory_from_previous:
            self.close_session()
            previous_session_id = None
            previous_abs_start_frame_idx = None

        request = {
            "type": "start_session",
            "resource_path": resource_path,
        }
        if (
            transfer_memory_from_previous
            and previous_session_id
            and previous_abs_start_frame_idx is not None
            and abs_start_frame_idx is not None
        ):
            request["experimental_transfer_memory_from_session_id"] = previous_session_id
            request["experimental_transfer_frame_offset"] = (
                int(previous_abs_start_frame_idx) - int(abs_start_frame_idx)
            )
        response = self.predictor.handle_request(
            request=request
        )
        self.session_id = response["session_id"]
        self.session_abs_start_frame_idx = abs_start_frame_idx
        self.last_memory_transfer_report = response.get("experimental_memory_transfer")
        if previous_session_id and previous_session_id != self.session_id:
            self._close_session_id(previous_session_id)

    def close_session(self) -> None:
        """Close the current SAM3 session if one is active."""
        if self.session_id:
            self._close_session_id(self.session_id)
            self.session_id = None
            self.session_abs_start_frame_idx = None

    def _close_session_id(self, session_id: str) -> None:
        """Close a specific SAM3 session id without changing adapter bookkeeping."""
        self.predictor.handle_request(
            request={
                "type": "close_session",
                "session_id": session_id,
            }
        )

    def reset_session(self) -> None:
        """Reset SAM3 state without tearing down the session handle."""
        if not self.session_id:
            return
        self.predictor.handle_request(
            request={
                "type": "reset_session",
                "session_id": self.session_id,
            }
        )

    def memory_summary(self) -> dict:
        """Return a compact debug summary of the active SAM3 session memory."""
        if not self.session_id:
            return {
                "session_id": None,
                "num_frames": 0,
                "tracker_states": 0,
                "object_ids": [],
                "cond_frames": [],
                "non_cond_frames": [],
                "tracked_frames": [],
                "cached_output_frames": [],
                "actions": 0,
                "transfer": self.last_memory_transfer_report,
            }
        response = self.predictor.handle_request(
            request={
                "type": "memory_summary",
                "session_id": self.session_id,
            }
        )
        response["abs_start_frame_idx"] = self.session_abs_start_frame_idx
        response["transfer"] = self.last_memory_transfer_report
        return response

    def add_object_points(
        self,
        frame_idx: int,
        obj_id: int,
        points_rel: List[List[float]],
        labels: List[int],
        mask_input: Optional[np.ndarray] = None,
    ) -> SamFrameOutput:
        """Submit point and optional mask prompts for one object on one frame."""
        if not self.session_id:
            raise RuntimeError("No active SAM3 session")

        request = {
            "type": "add_prompt",
            "session_id": self.session_id,
            "frame_index": frame_idx,
            "points": points_rel,
            "point_labels": labels,
            "obj_id": obj_id,
        }
        if mask_input is not None:
            request["mask_inputs"] = np.asarray(mask_input).astype(np.float32)

        response = self.predictor.handle_request(request=request)
        return self._parse_output(response["outputs"])

    def propagate_n_frames(self, start_frame_idx: int, max_frames: int) -> Dict[int, SamFrameOutput]:
        """Collect propagation outputs eagerly into a frame-indexed mapping."""
        if not self.session_id:
            raise RuntimeError("No active SAM3 session")
        outputs: Dict[int, SamFrameOutput] = {}
        for response in self.predictor.handle_stream_request(
            request={
                "type": "propagate_in_video",
                "session_id": self.session_id,
                "propagation_direction": "forward",
                "start_frame_index": start_frame_idx,
                "max_frame_num_to_track": max_frames,
            }
        ):
            outputs[int(response["frame_index"])] = self._parse_output(response["outputs"])
        return outputs

    def propagate_n_frames_stream(self, start_frame_idx: int, max_frames: int):
        """Yield propagation results incrementally for progress-driven UI updates."""
        if not self.session_id:
            raise RuntimeError("No active SAM3 session")
        for response in self.predictor.handle_stream_request(
            request={
                "type": "propagate_in_video",
                "session_id": self.session_id,
                "propagation_direction": "forward",
                "start_frame_index": start_frame_idx,
                "max_frame_num_to_track": max_frames,
            }
        ):
            yield int(response["frame_index"]), self._parse_output(response["outputs"])

    def _parse_output(self, outputs: dict) -> SamFrameOutput:
        """Normalize SAM3 response payloads into the annotator output model."""
        obj_ids = [int(x) for x in outputs.get("out_obj_ids", [])]
        masks = [np.asarray(m).astype(bool) for m in outputs.get("out_binary_masks", [])]
        boxes = [tuple(map(float, b)) for b in outputs.get("out_boxes_xywh", [])]
        scores = [float(x) for x in outputs.get("out_probs", [])]
        tracker_scores = [float(x) for x in outputs.get("out_tracker_probs", [])]
        if len(tracker_scores) < len(obj_ids):
            tracker_scores.extend([0.0] * (len(obj_ids) - len(tracker_scores)))
        return SamFrameOutput(
            obj_ids=obj_ids,
            masks=masks,
            boxes_xywh_norm=boxes,
            scores=scores,
            tracker_scores=tracker_scores,
        )


class SamWorker(QObject):
    """Background worker that serializes all SAM3 operations through one queue."""

    initialized = Signal(bool, str)
    memory_summary_done = Signal(str, object)
    segment_done = Signal(str, int, object)
    propagate_frame = Signal(str, int, object, int, int)
    propagate_done = Signal(str, int, int)
    propagate_stopped = Signal(str, int)
    task_failed = Signal(str, str, bool)
    task_finished = Signal(str)

    def __init__(
        self,
        checkpoint_path: Optional[str],
        bpe_path: Optional[str],
    ) -> None:
        """Capture worker configuration and initialize queue/cancellation state."""
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.bpe_path = bpe_path
        self.sam_adapter: Optional[Sam3Adapter] = None
        self._queue = deque()
        self._busy = False
        self._cancel_prefetch = False
        self._cancel_propagation_event = threading.Event()

    @Slot()
    def initialize(self) -> None:
        """Create the SAM3 adapter on the worker thread and report readiness."""
        try:
            self.sam_adapter = Sam3Adapter(
                checkpoint_path=self.checkpoint_path,
                bpe_path=self.bpe_path,
            )
        except Exception as exc:
            self.sam_adapter = None
            self.initialized.emit(False, str(exc))
            return
        self.initialized.emit(True, "")

    @Slot(str, str, object, bool)
    def enqueue_task(self, task_id: str, task_type: str, payload: object, priority: bool = False) -> None:
        """Queue a SAM task, optionally ahead of lower-priority work."""
        if priority:
            self._queue.appendleft((task_id, task_type, payload))
        else:
            self._queue.append((task_id, task_type, payload))
        if not self._busy:
            self._process_next()

    @Slot()
    def cancel_prefetch(self) -> None:
        """Mark the active prefetch stream as stale so it can be ignored."""
        self._cancel_prefetch = True

    @Slot()
    def cancel_propagation(self) -> None:
        """Request cancellation of the active propagation stream."""
        self._cancel_propagation_event.set()

    def _process_next(self) -> None:
        """Run the next queued task and schedule queue advancement back onto Qt."""
        if not self._queue:
            self._busy = False
            return
        self._busy = True
        task_id, task_type, payload = self._queue.popleft()
        try:
            if self.sam_adapter is None:
                self.task_failed.emit(task_id, "SAM3 not initialized.", False)
            elif task_type == "close_session":
                self.sam_adapter.close_session()
                self.task_finished.emit(task_id)
            elif task_type == "memory_summary":
                self.memory_summary_done.emit(task_id, self.sam_adapter.memory_summary())
            elif task_type == "segment":
                self._run_segment(task_id, payload)
            elif task_type in {"propagate", "prefetch"}:
                self._run_propagate(task_id, payload, task_type == "prefetch")
            else:
                self.task_failed.emit(task_id, f"Unknown task type: {task_type}", False)
        except Exception as exc:
            self.task_failed.emit(task_id, str(exc), False)
        finally:
            QMetaObject.invokeMethod(self, "_finish_task", Qt.QueuedConnection)

    @Slot()
    def _finish_task(self) -> None:
        """Release the worker for the next queued task."""
        self._busy = False
        self._process_next()

    def _run_segment(self, task_id: str, payload: object) -> None:
        """Start a single-frame session and merge per-object segmentation outputs."""
        data = payload or {}
        frame_idx = int(data.get("frame_idx", -1))
        obj_payload = data.get("payload", {})
        if frame_idx < 0:
            self.task_failed.emit(task_id, "Invalid frame index.", False)
            return
        if not obj_payload:
            self.task_failed.emit(task_id, "No prompts to segment.", True)
            return
        img_path = data.get("frame_path")
        if not img_path:
            self.task_failed.emit(task_id, "Missing frame path.", False)
            return
        img_pil = PilImage.open(str(img_path))
        self.sam_adapter.start_session([img_pil])
        composite = SamFrameOutput(obj_ids=[], masks=[], boxes_xywh_norm=[], scores=[], tracker_scores=[])
        for obj_id, prompt in obj_payload.items():
            points_rel = prompt.get("points_rel", [])
            labels = prompt.get("labels", [])
            if not points_rel:
                continue
            result = self.sam_adapter.add_object_points(
                frame_idx=0,
                obj_id=int(obj_id),
                points_rel=points_rel,
                labels=labels,
            )
            if int(obj_id) not in result.obj_ids:
                continue
            result_idx = result.obj_ids.index(int(obj_id))
            obj_output = SamFrameOutput(
                obj_ids=[int(obj_id)],
                masks=[result.masks[result_idx]],
                boxes_xywh_norm=[result.boxes_xywh_norm[result_idx]],
                scores=[result.scores[result_idx]],
                tracker_scores=[
                    result.tracker_scores[result_idx]
                    if result_idx < len(result.tracker_scores)
                    else 0.0
                ],
            )
            composite = merge_frame_outputs(composite, obj_output)
        self.segment_done.emit(task_id, frame_idx, composite)

    def _run_propagate(self, task_id: str, payload: object, is_prefetch: bool) -> None:
        """Execute tracker propagation for either visible playback or next-frame cache fill."""
        data = payload or {}
        seed_frame_idx = int(data.get("seed_frame_idx", -1))
        n_frames = int(data.get("n_frames", 0))
        frame_paths = data.get("frame_paths", [])
        prompt_payload = data.get("prompt_payload", {})
        allow_empty_result = bool(data.get("allow_empty_result", False))
        if seed_frame_idx < 0 or n_frames <= 1:
            self.task_failed.emit(task_id, "No forward frames available from current seed frame.", True)
            return
        abs_start = seed_frame_idx
        abs_end = min(abs_start + n_frames, len(frame_paths))
        actual_n = abs_end - abs_start
        if actual_n <= 1:
            self.task_failed.emit(task_id, "No forward frames available from current seed frame.", True)
            return
        if not is_prefetch:
            self._cancel_propagation_event.clear()

        imgs_pil = [PilImage.open(str(frame_paths[i])) for i in range(abs_start, abs_end)]
        self.sam_adapter.start_session(
            imgs_pil,
            abs_start_frame_idx=abs_start,
            transfer_memory_from_previous=not is_prefetch,
        )

        for obj_id, prompt in prompt_payload.items():
            points_rel = prompt.get("points_rel", [])
            labels = prompt.get("labels", [])
            mask_input = prompt.get("mask_input")
            if not points_rel and mask_input is None:
                continue
            self.sam_adapter.add_object_points(
                frame_idx=0,
                obj_id=int(obj_id),
                points_rel=points_rel,
                labels=labels,
                mask_input=mask_input,
            )

        last_masked_frame_idx: Optional[int] = None
        last_emitted_frame_idx = abs_start - 1
        max_frames = actual_n
        for session_idx, output in self.sam_adapter.propagate_n_frames_stream(
            start_frame_idx=0,
            max_frames=max_frames,
        ):
            if not is_prefetch and self._cancel_propagation_event.is_set():
                self._cancel_propagation_event.clear()
                self.propagate_stopped.emit(task_id, last_emitted_frame_idx)
                return
            if is_prefetch and self._cancel_prefetch:
                continue
            abs_frame_idx = abs_start + int(session_idx)
            last_emitted_frame_idx = abs_frame_idx
            chunk_session_idx = abs_frame_idx - abs_start
            self.propagate_frame.emit(task_id, abs_frame_idx, output, int(chunk_session_idx), actual_n)
            for mask in output.masks:
                if np.asarray(mask).any():
                    last_masked_frame_idx = abs_frame_idx
                    break

        if is_prefetch:
            self._cancel_prefetch = False
        else:
            self._cancel_propagation_event.clear()

        if last_masked_frame_idx is None and not allow_empty_result:
            self.task_failed.emit(
                task_id,
                "Chunk produced no valid masks. Please refine prompts and run again.",
                True,
            )
            return
        self.propagate_done.emit(
            task_id,
            last_masked_frame_idx if last_masked_frame_idx is not None else -1,
            abs_end - 1,
        )
