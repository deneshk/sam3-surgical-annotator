#!/usr/bin/env python3
"""PySide6 surgical video annotation app using SAM3 as assistive annotator."""

from __future__ import annotations

import math
import sys
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image as PilImage
from PySide6.QtCore import QObject, QPoint, QRect, QSize, Qt, QThread, Signal, Slot, QEvent, QEventLoop, QMetaObject, Q_ARG, QTimer
from PySide6.QtGui import QAction, QBrush, QColor, QGuiApplication, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QRubberBand,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sam3.model.sam3_video_predictor import Sam3VideoPredictor
from tools.exporters.coco_export import CocoExporter, ObjectInfo as ExportObjectInfo
from tools.exporters.coco_export import PointPrompt as ExportPointPrompt
from tools.exporters.coco_export import SamFrameOutput as ExportSamFrameOutput
from tools.exporters.perk_export import BoxPrompt as ExportPerkBoxPrompt
from tools.exporters.perk_export import ObjectInfo as ExportPerkObjectInfo
from tools.exporters.perk_export import PerkExporter
from tools.session_io import read_mask_png, read_session_json, write_mask_png, write_session_json
from tools.text_prompt_grounding import TextPromptProposal, build_text_prompt_proposals, next_prompt_object_name

MAX_POINTS_PER_OBJECT = 6
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_CANVAS_SIZE = (960, 540)
WINDOW_SCREEN_FRACTION = 0.85
CANVAS_SCREEN_FRACTION = (0.7, 0.6)
CANVAS_SCREEN_MARGIN_PX = 40
DEFAULT_RECONDITION_EVERY_NTH_FRAME = 16
DEFAULT_RECONDITION_HIGH_CONF_THRESH = 0.8
DEFAULT_RECONDITION_HIGH_IOU_THRESH = 0.8


@dataclass
class PointPrompt:
    x_px: int
    y_px: int
    is_positive: bool


@dataclass
class BoxPrompt:
    x1_px: int
    y1_px: int
    x2_px: int
    y2_px: int


@dataclass
class ObjectInfo:
    obj_id: int
    name: str
    color_bgr: Tuple[int, int, int]


@dataclass
class SamFrameOutput:
    obj_ids: List[int]
    masks: List[np.ndarray]
    boxes_xywh_norm: List[Tuple[float, float, float, float]]
    scores: List[float]
    tracker_scores: List[float]


@dataclass
class PendingPropagationState:
    remaining_chunks: int
    next_seed_frame_idx: int
    n_frames: int
    total_chunks: int
    target_frame_idx: Optional[int] = None
    completed_chunks: int = 0


class PropagationWorker(QObject):
    frame_ready = Signal(int, object, int, int)
    chunk_done = Signal(int, int, int)
    failed = Signal(str, bool)
    finished = Signal()

    def __init__(
        self,
        sam_adapter: "Sam3Adapter",
        frame_paths: List[Path],
        seed_frame_idx: int,
        n_frames: int,
        chunk_idx: int,
        prompt_payload: Dict[int, Dict[str, object]],
        sam_lock: Optional[threading.Lock] = None,
    ) -> None:
        super().__init__()
        self.sam_adapter = sam_adapter
        self.frame_paths = frame_paths
        self.seed_frame_idx = seed_frame_idx
        self.n_frames = n_frames
        self.chunk_idx = chunk_idx
        self.prompt_payload = prompt_payload
        self.sam_lock = sam_lock

    @Slot()
    def run(self) -> None:
        try:
            abs_start = self.seed_frame_idx
            abs_end = min(abs_start + self.n_frames, len(self.frame_paths))
            actual_n = abs_end - abs_start
            if actual_n <= 1:
                self.failed.emit("No forward frames available from current seed frame.", True)
                return

            lock = self.sam_lock
            if lock is None:
                lock_ctx = None
            else:
                lock_ctx = lock
            if lock_ctx:
                lock_ctx.acquire()
            try:
                imgs_pil = [PilImage.open(str(self.frame_paths[i])) for i in range(abs_start, abs_end)]
                self.sam_adapter.start_session(imgs_pil)

                for obj_id, payload in self.prompt_payload.items():
                    points_rel = payload.get("points_rel", [])
                    labels = payload.get("labels", [])
                    mask_input = payload.get("mask_input")
                    if not points_rel:
                        continue
                    self.sam_adapter.add_object_points(
                        frame_idx=0,
                        obj_id=obj_id,
                        points_rel=points_rel,
                        labels=labels,
                        mask_input=mask_input,
                    )

                session_outputs = self.sam_adapter.propagate_n_frames(
                    start_frame_idx=0,
                    max_frames=actual_n,
                )
            finally:
                if lock_ctx:
                    lock_ctx.release()

            last_masked_frame_idx: Optional[int] = None
            for session_idx, output in session_outputs.items():
                abs_frame_idx = abs_start + session_idx
                self.frame_ready.emit(abs_frame_idx, output, int(session_idx), actual_n)
                for mask in output.masks:
                    if np.asarray(mask).any():
                        last_masked_frame_idx = abs_frame_idx
                        break

            if last_masked_frame_idx is None:
                self.failed.emit(
                    "Chunk produced no valid masks. Please refine prompts and run again.",
                    True,
                )
                return

            self.chunk_done.emit(self.chunk_idx, last_masked_frame_idx, abs_end - 1)
        except Exception as exc:
            self.failed.emit(str(exc), False)
        finally:
            self.finished.emit()


class SamWorker(QObject):
    initialized = Signal(bool, str)
    segment_done = Signal(str, int, object)
    text_prompt_done = Signal(str, int, object, str)
    propagate_frame = Signal(str, int, object, int, int)
    propagate_done = Signal(str, int, int)
    propagate_stopped = Signal(str, int)
    task_failed = Signal(str, str, bool)
    task_finished = Signal(str)

    def __init__(
        self,
        checkpoint_path: Optional[str],
        bpe_path: Optional[str],
        recondition_every_nth_frame: int,
        recondition_high_conf_thresh: float,
        recondition_high_iou_thresh: float,
    ) -> None:
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.bpe_path = bpe_path
        self.recondition_every_nth_frame = recondition_every_nth_frame
        self.recondition_high_conf_thresh = recondition_high_conf_thresh
        self.recondition_high_iou_thresh = recondition_high_iou_thresh
        self.sam_adapter: Optional[Sam3Adapter] = None
        self._queue = deque()
        self._busy = False
        self._cancel_prefetch = False
        self._cancel_propagation_event = threading.Event()
        self._one_session_frame_signature: Optional[Tuple[str, ...]] = None

    @Slot()
    def initialize(self) -> None:
        try:
            self.sam_adapter = Sam3Adapter(
                checkpoint_path=self.checkpoint_path,
                bpe_path=self.bpe_path,
                recondition_every_nth_frame=self.recondition_every_nth_frame,
                recondition_high_conf_thresh=self.recondition_high_conf_thresh,
                recondition_high_iou_thresh=self.recondition_high_iou_thresh,
            )
        except Exception as exc:
            self.sam_adapter = None
            self.initialized.emit(False, str(exc))
            return
        self.initialized.emit(True, "")

    @Slot(str, str, object, bool)
    def enqueue_task(self, task_id: str, task_type: str, payload: object, priority: bool = False) -> None:
        if priority:
            self._queue.appendleft((task_id, task_type, payload))
        else:
            self._queue.append((task_id, task_type, payload))
        if not self._busy:
            self._process_next()

    @Slot()
    def cancel_prefetch(self) -> None:
        self._cancel_prefetch = True

    @Slot()
    def cancel_propagation(self) -> None:
        self._cancel_propagation_event.set()

    def _process_next(self) -> None:
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
                self._one_session_frame_signature = None
                self.task_finished.emit(task_id)
            elif task_type == "segment":
                self._run_segment(task_id, payload)
            elif task_type == "text_prompt":
                self._run_text_prompt(task_id, payload)
            elif task_type == "update_experimental_settings":
                self._run_update_experimental_settings(task_id, payload)
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
        self._busy = False
        self._process_next()

    def _run_segment(self, task_id: str, payload: object) -> None:
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
        self._one_session_frame_signature = None
        composite = SamFrameOutput(obj_ids=[], masks=[], boxes_xywh_norm=[], scores=[], tracker_scores=[])
        for obj_id, p in obj_payload.items():
            points_rel = p.get("points_rel", [])
            labels = p.get("labels", [])
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
            ridx = result.obj_ids.index(int(obj_id))
            obj_output = SamFrameOutput(
                obj_ids=[int(obj_id)],
                masks=[result.masks[ridx]],
                boxes_xywh_norm=[result.boxes_xywh_norm[ridx]],
                scores=[result.scores[ridx]],
                tracker_scores=[result.tracker_scores[ridx] if ridx < len(result.tracker_scores) else 0.0],
            )
            composite = _merge_outputs_for_worker(composite, obj_output)
        self.segment_done.emit(task_id, frame_idx, composite)

    def _run_text_prompt(self, task_id: str, payload: object) -> None:
        data = payload or {}
        frame_idx = int(data.get("frame_idx", -1))
        text_prompt = str(data.get("text_prompt", "")).strip()
        if frame_idx < 0:
            self.task_failed.emit(task_id, "Invalid frame index.", False)
            return
        if not text_prompt:
            self.task_failed.emit(task_id, "Enter a text prompt first.", True)
            return
        img_path = data.get("frame_path")
        if not img_path:
            self.task_failed.emit(task_id, "Missing frame path.", False)
            return
        img_pil = PilImage.open(str(img_path))
        self.sam_adapter.start_session([img_pil])
        self._one_session_frame_signature = None
        result = self.sam_adapter.add_text_prompt(frame_idx=0, text=text_prompt)
        self.text_prompt_done.emit(task_id, frame_idx, result, text_prompt)

    def _run_update_experimental_settings(self, task_id: str, payload: object) -> None:
        data = payload or {}
        self.recondition_every_nth_frame = int(
            data.get("recondition_every_nth_frame", self.recondition_every_nth_frame)
        )
        self.recondition_high_conf_thresh = float(
            data.get("recondition_high_conf_thresh", self.recondition_high_conf_thresh)
        )
        self.recondition_high_iou_thresh = float(
            data.get("recondition_high_iou_thresh", self.recondition_high_iou_thresh)
        )
        self.sam_adapter.update_experimental_settings(
            self.recondition_every_nth_frame,
            self.recondition_high_conf_thresh,
            self.recondition_high_iou_thresh,
        )
        self.task_finished.emit(task_id)

    def _run_propagate(self, task_id: str, payload: object, is_prefetch: bool) -> None:
        data = payload or {}
        seed_frame_idx = int(data.get("seed_frame_idx", -1))
        n_frames = int(data.get("n_frames", 0))
        frame_paths = data.get("frame_paths", [])
        prompt_payload = data.get("prompt_payload", {})
        use_one_session = bool(data.get("use_one_session", False)) and not is_prefetch
        rebuild_one_session = bool(data.get("rebuild_one_session", False))
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

        frame_signature = tuple(str(Path(p)) for p in frame_paths)
        if use_one_session:
            if rebuild_one_session or self.sam_adapter.session_id is None or self._one_session_frame_signature != frame_signature:
                imgs_pil = [PilImage.open(str(frame_path)) for frame_path in frame_paths]
                self.sam_adapter.start_session(imgs_pil)
                self._one_session_frame_signature = frame_signature
        else:
            imgs_pil = [PilImage.open(str(frame_paths[i])) for i in range(abs_start, abs_end)]
            self.sam_adapter.start_session(imgs_pil)
            self._one_session_frame_signature = None
        for obj_id, p in prompt_payload.items():
            points_rel = p.get("points_rel", [])
            labels = p.get("labels", [])
            mask_input = p.get("mask_input")
            if not points_rel:
                continue
            self.sam_adapter.add_object_points(
                frame_idx=abs_start if use_one_session else 0,
                obj_id=int(obj_id),
                points_rel=points_rel,
                labels=labels,
                mask_input=mask_input,
            )

        last_masked_frame_idx: Optional[int] = None
        last_emitted_frame_idx = abs_start - 1
        max_frames = (actual_n - 1) if use_one_session else actual_n
        for session_idx, output in self.sam_adapter.propagate_n_frames_stream(
            start_frame_idx=abs_start if use_one_session else 0,
            max_frames=max_frames,
        ):
            if not is_prefetch and self._cancel_propagation_event.is_set():
                self._cancel_propagation_event.clear()
                self.propagate_stopped.emit(task_id, last_emitted_frame_idx)
                return
            if is_prefetch and self._cancel_prefetch:
                continue
            abs_frame_idx = int(session_idx) if use_one_session else abs_start + int(session_idx)
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

        if last_masked_frame_idx is None:
            self.task_failed.emit(
                task_id,
                "Chunk produced no valid masks. Please refine prompts and run again.",
                True,
            )
            return
        self.propagate_done.emit(task_id, last_masked_frame_idx, abs_end - 1)


def _merge_outputs_for_worker(base_output: SamFrameOutput, new_output: SamFrameOutput) -> SamFrameOutput:
    merged = SamFrameOutput(
        obj_ids=list(base_output.obj_ids),
        masks=list(base_output.masks),
        boxes_xywh_norm=list(base_output.boxes_xywh_norm),
        scores=list(base_output.scores),
        tracker_scores=list(base_output.tracker_scores),
    )
    obj_to_idx = {obj_id: i for i, obj_id in enumerate(merged.obj_ids)}
    for i, obj_id in enumerate(new_output.obj_ids):
        if i >= len(new_output.masks) or i >= len(new_output.boxes_xywh_norm):
            continue
        score = new_output.scores[i] if i < len(new_output.scores) else 0.0
        tracker_score = new_output.tracker_scores[i] if i < len(new_output.tracker_scores) else 0.0
        if obj_id in obj_to_idx:
            idx = obj_to_idx[obj_id]
            merged.masks[idx] = new_output.masks[i]
            merged.boxes_xywh_norm[idx] = new_output.boxes_xywh_norm[i]
            merged.scores[idx] = score
            merged.tracker_scores[idx] = tracker_score
        else:
            merged.obj_ids.append(obj_id)
            merged.masks.append(new_output.masks[i])
            merged.boxes_xywh_norm.append(new_output.boxes_xywh_norm[i])
            merged.scores.append(score)
            merged.tracker_scores.append(tracker_score)
            obj_to_idx[obj_id] = len(merged.obj_ids) - 1
    return merged


class ClickableImageLabel(QLabel):
    def __init__(self, parent: "AnnotatorMainWindow"):
        super().__init__()
        self.parent_window = parent
        self._target_size = DEFAULT_CANVAS_SIZE
        self._pending_left_click_pos: Optional[Tuple[float, float]] = None
        self._pending_left_click_timer = QTimer(self)
        self._pending_left_click_timer.setSingleShot(True)
        self._pending_left_click_timer.timeout.connect(self._commit_pending_left_click)
        self.setAlignment(Qt.AlignCenter)
        self.setFocusPolicy(Qt.StrongFocus)
        # Keep layout stable while allowing full fit-to-panel rendering.
        self.setMinimumSize(*self._target_size)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setScaledContents(False)
        self.setStyleSheet("background-color: #111; border: 1px solid #444; color: #ddd;")
        self.setMouseTracking(True)

    def set_target_size(self, width: int, height: int) -> None:
        width = max(1, int(width))
        height = max(1, int(height))
        self._target_size = (width, height)
        self.setMinimumSize(width, height)
        self.updateGeometry()

    def sizeHint(self):
        return QSize(*self._target_size)

    def _cancel_pending_left_click(self) -> None:
        self._pending_left_click_timer.stop()
        self._pending_left_click_pos = None

    def _commit_pending_left_click(self) -> None:
        if self._pending_left_click_pos is None:
            return
        ui_x, ui_y = self._pending_left_click_pos
        self._pending_left_click_pos = None
        self.parent_window.on_image_press(ui_x, ui_y)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self._cancel_pending_left_click()
            self.setFocus(Qt.MouseFocusReason)
            self.parent_window.on_pan_press(event.position().x(), event.position().y())
            event.accept()
            return
        if event.button() != Qt.LeftButton:
            event.ignore()
            return
        self.setFocus(Qt.MouseFocusReason)
        if self.parent_window.is_box_mode_active():
            self._cancel_pending_left_click()
            self.parent_window.on_image_press(event.position().x(), event.position().y())
        else:
            self._pending_left_click_pos = (event.position().x(), event.position().y())
            self._pending_left_click_timer.start(QApplication.doubleClickInterval())
        event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.RightButton:
            self.parent_window.on_pan_drag(event.position().x(), event.position().y())
            return
        self.parent_window.on_image_hover(event.position().x(), event.position().y())
        self.parent_window.on_image_drag(event.position().x(), event.position().y())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.parent_window.on_pan_release()
            event.accept()
            return
        if event.button() != Qt.LeftButton:
            event.ignore()
            return
        if self.parent_window.is_box_mode_active():
            self.parent_window.on_image_release(event.position().x(), event.position().y())
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() != Qt.LeftButton:
            event.ignore()
            return
        self._cancel_pending_left_click()
        self.setFocus(Qt.MouseFocusReason)
        self.parent_window.on_image_double_click(event.position().x(), event.position().y())
        event.accept()

    def leaveEvent(self, _event):
        self._cancel_pending_left_click()
        self.parent_window.on_image_hover(None, None)

    def wheelEvent(self, event):
        self.parent_window.on_image_wheel(
            event.position().x(),
            event.position().y(),
            event.angleDelta().y(),
        )
        event.accept()


class ArrowSpinBox(QWidget):
    valueChanged = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._prev_btn = QToolButton(self)
        self._prev_btn.setText("<")
        self._prev_btn.setAutoRepeat(True)
        self._prev_btn.setMinimumSize(34, 34)

        self._spin = QSpinBox(self)
        self._spin.setButtonSymbols(QSpinBox.NoButtons)
        self._spin.setMinimumHeight(34)
        self._spin.setMinimumWidth(96)
        self._spin.valueChanged.connect(self.valueChanged.emit)

        self._next_btn = QToolButton(self)
        self._next_btn.setText(">")
        self._next_btn.setAutoRepeat(True)
        self._next_btn.setMinimumSize(34, 34)

        layout.addWidget(self._prev_btn)
        layout.addWidget(self._spin, 1)
        layout.addWidget(self._next_btn)

        self._prev_btn.clicked.connect(self._spin.stepDown)
        self._next_btn.clicked.connect(self._spin.stepUp)
        self.setMinimumHeight(38)

    def setMinimum(self, value: int) -> None:
        self._spin.setMinimum(value)

    def setMaximum(self, value: int) -> None:
        self._spin.setMaximum(value)

    def setRange(self, minimum: int, maximum: int) -> None:
        self._spin.setRange(minimum, maximum)

    def setValue(self, value: int) -> None:
        self._spin.setValue(value)

    def value(self) -> int:
        return self._spin.value()

    def blockSignals(self, block: bool) -> bool:
        self._spin.blockSignals(block)
        return super().blockSignals(block)

    def spinBox(self) -> QSpinBox:
        return self._spin


class Sam3Adapter:
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        bpe_path: Optional[str] = None,
        recondition_every_nth_frame: int = DEFAULT_RECONDITION_EVERY_NTH_FRAME,
        recondition_high_conf_thresh: float = DEFAULT_RECONDITION_HIGH_CONF_THRESH,
        recondition_high_iou_thresh: float = DEFAULT_RECONDITION_HIGH_IOU_THRESH,
    ) -> None:
        self.predictor = Sam3VideoPredictor(
            checkpoint_path=checkpoint_path,
            bpe_path=bpe_path,
            recondition_every_nth_frame=recondition_every_nth_frame,
            recondition_high_conf_thresh=recondition_high_conf_thresh,
            recondition_high_iou_thresh=recondition_high_iou_thresh,
        )
        self.session_id: Optional[str] = None

    def start_session(self, resource_path) -> None:
        """Start a new session. resource_path may be a directory/file path string
        or a list of PIL images (used for loading a subset of frames)."""
        if self.session_id:
            self.close_session()
        response = self.predictor.handle_request(
            request={
                "type": "start_session",
                "resource_path": resource_path,
            }
        )
        self.session_id = response["session_id"]

    def close_session(self) -> None:
        if self.session_id:
            self.predictor.handle_request(
                request={
                    "type": "close_session",
                    "session_id": self.session_id,
                }
            )
            self.session_id = None

    def reset_session(self) -> None:
        if not self.session_id:
            return
        self.predictor.handle_request(
            request={
                "type": "reset_session",
                "session_id": self.session_id,
            }
        )

    def add_object_points(
        self,
        frame_idx: int,
        obj_id: int,
        points_rel: List[List[float]],
        labels: List[int],
        mask_input: Optional[np.ndarray] = None,
    ) -> SamFrameOutput:
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

        response = self.predictor.handle_request(
            request=request
        )
        return self._parse_output(response["outputs"])

    def add_text_prompt(
        self,
        frame_idx: int,
        text: str,
    ) -> SamFrameOutput:
        if not self.session_id:
            raise RuntimeError("No active SAM3 session")
        response = self.predictor.handle_request(
            request={
                "type": "add_prompt",
                "session_id": self.session_id,
                "frame_index": frame_idx,
                "text": text,
            }
        )
        return self._parse_output(response["outputs"])

    def update_experimental_settings(
        self,
        recondition_every_nth_frame: int,
        recondition_high_conf_thresh: float,
        recondition_high_iou_thresh: float,
    ) -> None:
        model = self.predictor.model
        model.recondition_every_nth_frame = int(recondition_every_nth_frame)
        model.recondition_high_conf_thresh = float(recondition_high_conf_thresh)
        model.recondition_high_iou_thresh = float(recondition_high_iou_thresh)

    def propagate_n_frames(
        self, start_frame_idx: int, max_frames: int
    ) -> Dict[int, SamFrameOutput]:
        """Propagate forward from start_frame_idx (session-relative) for up to max_frames frames."""
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

    def propagate_n_frames_stream(
        self, start_frame_idx: int, max_frames: int
    ):
        """Yield (frame_idx, SamFrameOutput) as propagation progresses."""
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


class AnnotatorMainWindow(QMainWindow):
    task_requested = Signal(str, str, object, bool)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM3 Surgical Video Annotator")
        self._apply_window_sizing()

        self.sam_adapter: Optional[Sam3Adapter] = None
        self._checkpoint_path: Optional[str] = None
        self._bpe_path: Optional[str] = None
        self.frame_paths: List[Path] = []
        self.current_frame_idx: int = 0
        self.image_dir: Optional[Path] = None

        self.objects: List[ObjectInfo] = []
        self.active_object_id: Optional[int] = None
        self.next_obj_id: int = 1
        self.hidden_obj_ids: set[int] = set()
        self.solo_object_id: Optional[int] = None
        self._object_row_widgets: Dict[int, Dict[str, QWidget]] = {}
        self.flagged_frame_indices: set[int] = set()

        self.prompts_by_frame_obj: Dict[int, Dict[int, List[PointPrompt]]] = {}
        self.box_prompts_by_frame_obj: Dict[int, Dict[int, BoxPrompt]] = {}
        self.box_locked_by_frame_obj: Dict[int, Dict[int, bool]] = {}
        self.outputs_by_frame: Dict[int, SamFrameOutput] = {}
        self.manual_propagation_overrides_by_frame_obj: Dict[int, Dict[int, bool]] = {}
        self._text_prompt_proposals_frame_idx: Optional[int] = None
        self._text_prompt_last_prompt: str = ""
        self._text_prompt_proposals: List[TextPromptProposal] = []
        self._active_prompt_rows: List[Tuple[str, int]] = []

        self.segment_mode: bool = False
        self.current_prompt_positive: bool = True
        self._display_pixmap: Optional[QPixmap] = None
        self._display_scale: float = 1.0
        self._display_offset: Tuple[int, int] = (0, 0)
        self._display_image_size: Tuple[int, int] = (0, 0)
        self._zoom_multiplier: float = 1.0
        self._pan_offset_ui: Tuple[float, float] = (0.0, 0.0)
        self._pan_drag_last_ui_xy: Optional[Tuple[float, float]] = None
        self._pending_propagation: Optional[PendingPropagationState] = None
        self._propagation_thread: Optional[QThread] = None
        self._propagation_worker: Optional[PropagationWorker] = None
        self._propagation_busy: bool = False
        self._propagation_enabled_obj_ids: Optional[set[int]] = None
        self._propagation_view_frame_idx: Optional[int] = None
        self._propagation_continue_after_chunk: bool = False
        self._propagation_active_chunk_idx: Optional[int] = None
        self._propagation_active_seed_frame_idx: Optional[int] = None
        self._propagation_task_id: Optional[str] = None
        self._propagation_stop_requested: bool = False
        self._propagation_seen_obj_ids: set[int] = set()
        self._propagation_lost_obj_ids: set[int] = set()
        self._propagation_loss_notified: bool = False
        self._sam_thread: Optional[QThread] = None
        self._sam_worker: Optional[SamWorker] = None
        self._sam_ready: bool = False
        self._sam_task_counter: int = 0
        self._sam_task_contexts: Dict[str, Dict[str, object]] = {}
        self._sam_waiting: Dict[str, QEventLoop] = {}
        self._sam_task_results: Dict[str, object] = {}
        self._prefetch_thread: Optional[QThread] = None
        self._prefetch_worker: Optional[PropagationWorker] = None
        self._prefetch_busy: bool = False
        self._prefetch_cancel_requested: bool = False
        self._prefetch_pending_restart: bool = False
        self._prefetch_target_frame_idx: Optional[int] = None
        self._prefetch_seed_frame_idx: Optional[int] = None
        self._prefetch_prompt_version: int = 0
        self._prefetch_active_version: Optional[int] = None
        self._prefetch_cached_frame_idx: Optional[int] = None
        self._prefetch_cached_seed_idx: Optional[int] = None
        self._prefetch_cached_version: Optional[int] = None
        self._prefetch_provenance_by_frame: Dict[int, Dict[str, object]] = {}
        self._output_version_by_frame: Dict[int, int] = {}
        self._undo_state: Optional[dict] = None
        self._last_prompt_edit_frame: Optional[int] = None
        self._one_session_chunk_prompt_version: Optional[int] = None
        self._session_dir: Optional[Path] = None
        self._autosave_timer = QTimer(self)
        self._autosave_timer.timeout.connect(self._handle_autosave)
        self._box_draw_start_xy: Optional[Tuple[int, int]] = None
        self._box_draw_start_ui_xy: Optional[Tuple[int, int]] = None
        self._box_edit_active: bool = False
        self._box_drag_mode: Optional[str] = None
        self._box_drag_reference_box: Optional[BoxPrompt] = None
        self._box_drag_offset_xy: Tuple[int, int] = (0, 0)
        self._selected_box_obj_id: Optional[int] = None
        self.show_prompts: bool = True
        self.show_segmentations: bool = True
        self.show_boxes: bool = True
        self.show_box_titles: bool = True
        self.auto_propagate_next: bool = False
        self.propagation_mode: str = "tracker"
        self.translate_prompts_on_propagation: bool = True
        self.use_point_prompts_for_propagation: bool = True
        self.use_target_frame: bool = False
        self.use_one_session_chunked_propagation: bool = False
        self.segmentation_opacity: float = 0.6
        self.box_line_thickness: int = 2
        self.recondition_every_nth_frame: int = DEFAULT_RECONDITION_EVERY_NTH_FRAME
        self.recondition_high_conf_thresh: float = DEFAULT_RECONDITION_HIGH_CONF_THRESH
        self.recondition_high_iou_thresh: float = DEFAULT_RECONDITION_HIGH_IOU_THRESH

        self._setup_ui()
        QTimer.singleShot(0, self._apply_canvas_sizing)

    def eventFilter(self, watched: QObject, event) -> bool:
        if event.type() == QEvent.KeyPress and event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Escape):
            handled = super().eventFilter(watched, event)
            QTimer.singleShot(0, self._focus_canvas)
            return handled
        return super().eventFilter(watched, event)

    def _focus_canvas(self) -> None:
        if hasattr(self, "image_label") and self.image_label is not None:
            self.image_label.setFocus(Qt.ShortcutFocusReason)

    def _install_focus_return_on_widget(self, widget: Optional[QWidget]) -> None:
        if widget is None:
            return
        widget.installEventFilter(self)
        if isinstance(widget, QComboBox):
            line_edit = widget.lineEdit()
            if line_edit is not None:
                line_edit.installEventFilter(self)
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            line_edit = widget.lineEdit()
            if line_edit is not None:
                line_edit.installEventFilter(self)
        elif isinstance(widget, ArrowSpinBox):
            spin = widget.spinBox()
            spin.installEventFilter(self)
            line_edit = spin.lineEdit()
            if line_edit is not None:
                line_edit.installEventFilter(self)

    def _get_available_screen_geometry(self) -> QRect:
        screen = self.windowHandle().screen() if self.windowHandle() else QGuiApplication.primaryScreen()
        if screen is None:
            return QRect(0, 0, 1600, 900)
        return screen.availableGeometry()

    def _apply_window_sizing(self) -> None:
        geom = self._get_available_screen_geometry()
        avail_w = max(1, geom.width())
        avail_h = max(1, geom.height())
        win_w = max(1, int(avail_w * WINDOW_SCREEN_FRACTION))
        win_h = max(1, int(avail_h * WINDOW_SCREEN_FRACTION))
        self.resize(win_w, win_h)

    def _apply_canvas_sizing(self) -> None:
        geom = self._get_available_screen_geometry()
        avail_w = max(1, geom.width())
        avail_h = max(1, geom.height())
        win_w = max(1, self.width() or avail_w)
        win_h = max(1, self.height() or avail_h)
        target_w = max(1, int(avail_w * CANVAS_SCREEN_FRACTION[0]))
        target_h = max(1, int(avail_h * CANVAS_SCREEN_FRACTION[1]))
        cap_w = max(1, avail_w - CANVAS_SCREEN_MARGIN_PX)
        cap_h = max(1, avail_h - CANVAS_SCREEN_MARGIN_PX)
        left_w = 0
        if hasattr(self, "_left_review_panel") and self._left_review_panel is not None:
            left_w = max(
                self._left_review_panel.width(),
                self._left_review_panel.minimumWidth(),
                self._left_review_panel.sizeHint().width(),
            )
        right_w = (
            self._right_panel.sizeHint().width() if hasattr(self, "_right_panel") else 0
        )
        available_canvas_w = max(1, win_w - left_w - right_w - CANVAS_SCREEN_MARGIN_PX)
        target_w = min(target_w, cap_w, available_canvas_w)
        target_h = min(target_h, cap_h, int(win_h * CANVAS_SCREEN_FRACTION[1]))
        self.image_label.set_target_size(target_w, target_h)

    def _setup_ui(self) -> None:
        load_action = QAction("Load Frame Directory", self)
        load_action.triggered.connect(self.load_frame_directory)
        self.menuBar().addAction(load_action)

        export_action = QAction("Export Annotations", self)
        export_action.triggered.connect(self.export_annotations)
        self.menuBar().addAction(export_action)

        save_action = QAction("Save Session", self)
        save_action.triggered.connect(self.save_session_dialog)
        self.menuBar().addAction(save_action)

        load_session_action = QAction("Load Session", self)
        load_session_action.triggered.connect(self.load_session_dialog)
        self.menuBar().addAction(load_session_action)

        help_menu = self.menuBar().addMenu("Help")
        hotkeys_action = QAction("Hotkeys List", self)
        hotkeys_action.triggered.connect(self.show_hotkeys_list)
        help_menu.addAction(hotkeys_action)
        walkthrough_action = QAction("Instructions / Walkthrough", self)
        walkthrough_action.triggered.connect(self.show_instructions_walkthrough)
        help_menu.addAction(walkthrough_action)

        root = QWidget()
        root_layout = QHBoxLayout(root)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        left_panel = QWidget()
        left_panel.setMinimumWidth(180)
        left_panel.setMaximumWidth(260)
        self._left_review_panel = left_panel
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("Flagged Frames"))
        flag_button_row = QHBoxLayout()
        self.flag_toggle_btn = QPushButton("Flag / Unflag")
        self.flag_toggle_btn.clicked.connect(self.toggle_current_frame_flag)
        self.prev_flag_btn = QPushButton("Prev Flag")
        self.prev_flag_btn.clicked.connect(self.go_prev_flagged_frame)
        self.next_flag_btn = QPushButton("Next Flag")
        self.next_flag_btn.clicked.connect(self.go_next_flagged_frame)
        flag_button_row.addWidget(self.flag_toggle_btn)
        left_layout.addLayout(flag_button_row)
        jump_button_row = QHBoxLayout()
        jump_button_row.addWidget(self.prev_flag_btn)
        jump_button_row.addWidget(self.next_flag_btn)
        left_layout.addLayout(jump_button_row)
        self.flagged_frames_list = QListWidget()
        self.flagged_frames_list.currentItemChanged.connect(self._on_flagged_frame_selection_changed)
        left_layout.addWidget(self.flagged_frames_list)

        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)

        nav_row = QHBoxLayout()
        self.fit_view_btn = QPushButton("Fit to Screen")
        self.fit_view_btn.clicked.connect(self.fit_current_frame_to_view)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(1)
        self.frame_slider.setMaximum(1)
        self.frame_slider.setValue(1)
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        self.frame_slider.setMinimumHeight(34)
        self.frame_slider.setStyleSheet(
            "QSlider::groove:horizontal { height: 12px; border-radius: 6px; background: #4a4a4a; }"
            "QSlider::handle:horizontal { width: 22px; margin: -6px 0; border-radius: 11px; background: #d9d9d9; }"
            "QSlider::sub-page:horizontal { border-radius: 6px; background: #6aa3d9; }"
            "QSlider::add-page:horizontal { border-radius: 6px; background: #2a2a2a; }"
        )
        self.frame_label = QLabel("Frame: -/-")
        self.frame_jump_spin = ArrowSpinBox()
        self.frame_jump_spin.setMinimum(1)
        self.frame_jump_spin.setMaximum(1)
        self.frame_jump_spin.setValue(1)
        self.frame_jump_spin.valueChanged.connect(self._on_frame_jump_changed)
        nav_row.addWidget(self.fit_view_btn)
        nav_row.addWidget(self.frame_slider, 1)
        nav_row.addWidget(self.frame_jump_spin)
        nav_row.addWidget(self.frame_label)
        center_layout.addLayout(nav_row)

        self.image_label = ClickableImageLabel(self)
        center_layout.addWidget(self.image_label)
        self._box_rubber_band = QRubberBand(QRubberBand.Rectangle, self.image_label)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self._right_panel = right_panel
        right_panel.setMinimumWidth(550)
        control_tabs = QTabWidget()
        right_layout.addWidget(control_tabs)

        prompt_tab = QWidget()
        prompt_layout = QVBoxLayout(prompt_tab)
        processing_tab = QWidget()
        processing_layout = QVBoxLayout(processing_tab)
        experimental_tab = QWidget()
        experimental_layout = QVBoxLayout(experimental_tab)

        model_form = QFormLayout()
        self.checkpoint_combo = QComboBox()
        self.checkpoint_combo.addItem("Use default HF checkpoint")
        self.checkpoint_combo.setEditable(True)
        self.checkpoint_combo.lineEdit().setPlaceholderText("Optional local checkpoint path")
        model_form.addRow("Checkpoint", self.checkpoint_combo)
        processing_layout.addLayout(model_form)

        self.object_list = QListWidget()
        self.object_list.currentItemChanged.connect(self._on_object_selection_changed)
        prompt_layout.addWidget(QLabel("Objects"))
        prompt_layout.addWidget(self.object_list)
        self.active_object_label = QLabel("Active: None")
        prompt_layout.addWidget(self.active_object_label)

        obj_row = QHBoxLayout()
        self.add_obj_btn = QPushButton("Add Object")
        self.add_obj_btn.clicked.connect(self.add_object)
        self.remove_obj_btn = QPushButton("Remove Object")
        self.remove_obj_btn.clicked.connect(self.remove_active_object)
        obj_row.addWidget(self.add_obj_btn)
        obj_row.addWidget(self.remove_obj_btn)
        prompt_layout.addLayout(obj_row)

        self.text_prompt_input = QLineEdit()
        self.text_prompt_input.setPlaceholderText("Frame text prompt, e.g. dog")
        prompt_layout.addWidget(QLabel("Text Prompt Objects"))
        prompt_layout.addWidget(self.text_prompt_input)

        text_prompt_row = QHBoxLayout()
        self.generate_text_prompt_btn = QPushButton("Generate Objects")
        self.generate_text_prompt_btn.clicked.connect(self.generate_objects_from_text_prompt)
        self.accept_text_prompt_btn = QPushButton("Accept Selected")
        self.accept_text_prompt_btn.clicked.connect(self.accept_selected_text_prompt_proposals)
        self.clear_text_prompt_btn = QPushButton("Clear Proposals")
        self.clear_text_prompt_btn.clicked.connect(self.clear_text_prompt_proposals)
        text_prompt_row.addWidget(self.generate_text_prompt_btn)
        text_prompt_row.addWidget(self.accept_text_prompt_btn)
        text_prompt_row.addWidget(self.clear_text_prompt_btn)
        prompt_layout.addLayout(text_prompt_row)

        self.text_prompt_list = QListWidget()
        prompt_layout.addWidget(self.text_prompt_list)

        self.auto_propagate_next_check = QCheckBox("Auto Propagate Next Frame")
        self.auto_propagate_next_check.toggled.connect(self._on_auto_propagate_next_toggled)
        processing_layout.addWidget(self.auto_propagate_next_check)

        propagation_mode_form = QFormLayout()
        self.propagation_mode_combo = QComboBox()
        self.propagation_mode_combo.addItem("Tracker", "tracker")
        self.propagation_mode_combo.addItem("Copy Boxes", "copy_boxes")
        self.propagation_mode_combo.currentIndexChanged.connect(self._on_propagation_mode_changed)
        propagation_mode_form.addRow("Propagation Mode", self.propagation_mode_combo)
        processing_layout.addLayout(propagation_mode_form)

        self.translate_prompts_check = QCheckBox("Translate Point Prompts on Propagation")
        self.translate_prompts_check.setChecked(self.translate_prompts_on_propagation)
        self.translate_prompts_check.toggled.connect(self._on_translate_prompts_toggled)
        processing_layout.addWidget(self.translate_prompts_check)

        self.use_point_prompts_for_propagation_check = QCheckBox("Use Point Prompts for Propagation")
        self.use_point_prompts_for_propagation_check.setChecked(self.use_point_prompts_for_propagation)
        self.use_point_prompts_for_propagation_check.setToolTip(
            "When disabled, current-frame point prompts are ignored as tracker seed inputs."
        )
        self.use_point_prompts_for_propagation_check.toggled.connect(
            self._on_use_point_prompts_for_propagation_toggled
        )
        processing_layout.addWidget(self.use_point_prompts_for_propagation_check)

        autosave_row = QHBoxLayout()
        self.autosave_check = QCheckBox("Auto-save")
        self.autosave_check.setChecked(True)
        self.autosave_check.toggled.connect(self._on_autosave_toggled)
        self.autosave_minutes_spin = QSpinBox()
        self.autosave_minutes_spin.setMinimum(1)
        self.autosave_minutes_spin.setMaximum(60)
        self.autosave_minutes_spin.setValue(5)
        self.autosave_minutes_spin.valueChanged.connect(self._on_autosave_interval_changed)
        autosave_row.addWidget(self.autosave_check)
        autosave_row.addWidget(QLabel("Minutes:"))
        autosave_row.addWidget(self.autosave_minutes_spin)
        processing_layout.addLayout(autosave_row)

        self.lock_all_boxes_check = QCheckBox("Lock All Boxes")
        self.lock_all_boxes_check.toggled.connect(self._on_lock_all_boxes_toggled)
        prompt_layout.addWidget(self.lock_all_boxes_check)

        self.prompt_mode_label = QLabel("Prompt Mode")
        prompt_layout.addWidget(self.prompt_mode_label)
        self.prompt_mode_combo = QComboBox()
        self.prompt_mode_combo.addItems(["Positive (+)", "Negative (-)", "Box (drag)"])
        self.prompt_mode_combo.setCurrentIndex(2)
        self.prompt_mode_combo.currentIndexChanged.connect(self._on_prompt_mode_changed)
        prompt_layout.addWidget(self.prompt_mode_combo)

        self.annotation_list_label = QLabel("Current Frame Annotations")
        prompt_layout.addWidget(self.annotation_list_label)
        self.point_list = QListWidget()
        self.point_list.currentRowChanged.connect(self._on_point_list_selection_changed)
        prompt_layout.addWidget(self.point_list)

        point_row = QHBoxLayout()
        self.remove_point_btn = QPushButton("Remove Selected Prompt")
        self.remove_point_btn.clicked.connect(self.remove_selected_prompt)
        self.clear_points_btn = QPushButton("Clear Object Prompts")
        self.clear_points_btn.clicked.connect(self.clear_active_object_prompts)
        point_row.addWidget(self.remove_point_btn)
        point_row.addWidget(self.clear_points_btn)
        prompt_layout.addLayout(point_row)

        self.segment_btn = QPushButton("Segment")
        self.segment_btn.clicked.connect(self.segment_current_frame)
        processing_layout.addWidget(self.segment_btn)

        prompt_layout.addWidget(QLabel("View"))
        self.show_prompts_check = QCheckBox("Show Prompts")
        self.show_prompts_check.setChecked(self.show_prompts)
        self.show_prompts_check.toggled.connect(self._on_view_settings_changed)
        prompt_layout.addWidget(self.show_prompts_check)

        self.show_segmentations_check = QCheckBox("Show Segmentations")
        self.show_segmentations_check.setChecked(self.show_segmentations)
        self.show_segmentations_check.toggled.connect(self._on_view_settings_changed)
        prompt_layout.addWidget(self.show_segmentations_check)

        self.show_boxes_check = QCheckBox("Show Boxes")
        self.show_boxes_check.setChecked(self.show_boxes)
        self.show_boxes_check.toggled.connect(self._on_view_settings_changed)
        prompt_layout.addWidget(self.show_boxes_check)

        self.show_box_titles_check = QCheckBox("Show Box Titles")
        self.show_box_titles_check.setChecked(self.show_box_titles)
        self.show_box_titles_check.toggled.connect(self._on_view_settings_changed)
        prompt_layout.addWidget(self.show_box_titles_check)

        view_form = QFormLayout()
        self.segmentation_opacity_spin = QDoubleSpinBox()
        self.segmentation_opacity_spin.setMinimum(0.0)
        self.segmentation_opacity_spin.setMaximum(1.0)
        self.segmentation_opacity_spin.setSingleStep(0.05)
        self.segmentation_opacity_spin.setDecimals(2)
        self.segmentation_opacity_spin.setValue(self.segmentation_opacity)
        self.segmentation_opacity_spin.valueChanged.connect(self._on_view_settings_changed)
        view_form.addRow("Mask opacity", self.segmentation_opacity_spin)

        self.box_line_thickness_spin = QSpinBox()
        self.box_line_thickness_spin.setMinimum(1)
        self.box_line_thickness_spin.setMaximum(10)
        self.box_line_thickness_spin.setValue(self.box_line_thickness)
        self.box_line_thickness_spin.valueChanged.connect(self._on_view_settings_changed)
        view_form.addRow("Box thickness", self.box_line_thickness_spin)
        prompt_layout.addLayout(view_form)

        propagate_row = QHBoxLayout()
        self.propagate_btn = QPushButton("Propagate")
        self.propagate_btn.clicked.connect(self.propagate_next_frame)
        self.n_propagate_spin = QSpinBox()
        self.n_propagate_spin.setMinimum(2)
        self.n_propagate_spin.setMaximum(9999)
        self.n_propagate_spin.setValue(2)
        self.n_propagate_spin.setToolTip("Number of frames to propagate (including current frame)")
        self.n_propagate_spin.valueChanged.connect(self._on_propagation_target_changed)
        self.chunks_spin = QSpinBox()
        self.chunks_spin.setMinimum(1)
        self.chunks_spin.setMaximum(9999)
        self.chunks_spin.setValue(1)
        self.chunks_spin.setToolTip("Number of propagation chunks to run")
        propagate_row.addWidget(self.propagate_btn)
        propagate_row.addWidget(QLabel("Chunk Size:"))
        propagate_row.addWidget(self.n_propagate_spin)
        propagate_row.addWidget(QLabel("# of Chunks"))
        propagate_row.addWidget(self.chunks_spin)
        processing_layout.addLayout(propagate_row)

        propagate_control_row = QHBoxLayout()
        self.stop_propagate_btn = QPushButton("Stop")
        self.stop_propagate_btn.setEnabled(False)
        self.stop_propagate_btn.clicked.connect(self.request_stop_propagation)
        propagate_control_row.addWidget(self.stop_propagate_btn)
        processing_layout.addLayout(propagate_control_row)

        target_row = QHBoxLayout()
        self.use_target_frame_check = QCheckBox("Use Target Frame")
        self.use_target_frame_check.toggled.connect(self._on_propagation_target_toggled)
        self.target_frame_spin = QSpinBox()
        self.target_frame_spin.setMinimum(1)
        self.target_frame_spin.setMaximum(1)
        self.target_frame_spin.setValue(1)
        self.target_frame_spin.setEnabled(False)
        self.target_frame_spin.setToolTip("Last frame to include in manual propagation")
        self.target_frame_spin.valueChanged.connect(self._on_propagation_target_changed)
        self.computed_chunks_label = QLabel("Chunks: -")
        target_row.addWidget(self.use_target_frame_check)
        target_row.addWidget(QLabel("To frame:"))
        target_row.addWidget(self.target_frame_spin)
        target_row.addWidget(self.computed_chunks_label)
        processing_layout.addLayout(target_row)

        self.mode_label = QLabel("Mode: Prompt")
        processing_layout.addWidget(self.mode_label)
        experimental_layout.addWidget(QLabel("Experimental tracker heuristics"))
        experimental_layout.addWidget(
            QLabel("These settings map to SAM3 periodic re-prompting and apply to future tracker operations.")
        )
        experimental_form = QFormLayout()
        self.recondition_every_nth_frame_spin = QSpinBox()
        self.recondition_every_nth_frame_spin.setMinimum(0)
        self.recondition_every_nth_frame_spin.setMaximum(9999)
        self.recondition_every_nth_frame_spin.setValue(self.recondition_every_nth_frame)
        self.recondition_every_nth_frame_spin.setToolTip("0 disables periodic re-prompting.")
        self.recondition_every_nth_frame_spin.valueChanged.connect(self._on_experimental_settings_changed)
        experimental_form.addRow("Re-prompt Every N Frames", self.recondition_every_nth_frame_spin)

        self.recondition_high_conf_thresh_spin = QDoubleSpinBox()
        self.recondition_high_conf_thresh_spin.setMinimum(0.0)
        self.recondition_high_conf_thresh_spin.setMaximum(1.0)
        self.recondition_high_conf_thresh_spin.setSingleStep(0.05)
        self.recondition_high_conf_thresh_spin.setDecimals(2)
        self.recondition_high_conf_thresh_spin.setValue(self.recondition_high_conf_thresh)
        self.recondition_high_conf_thresh_spin.valueChanged.connect(self._on_experimental_settings_changed)
        experimental_form.addRow("High Confidence Threshold", self.recondition_high_conf_thresh_spin)

        self.recondition_high_iou_thresh_spin = QDoubleSpinBox()
        self.recondition_high_iou_thresh_spin.setMinimum(0.0)
        self.recondition_high_iou_thresh_spin.setMaximum(1.0)
        self.recondition_high_iou_thresh_spin.setSingleStep(0.05)
        self.recondition_high_iou_thresh_spin.setDecimals(2)
        self.recondition_high_iou_thresh_spin.setValue(self.recondition_high_iou_thresh)
        self.recondition_high_iou_thresh_spin.valueChanged.connect(self._on_experimental_settings_changed)
        experimental_form.addRow("High IoU Threshold", self.recondition_high_iou_thresh_spin)
        self.use_one_session_chunked_propagation_check = QCheckBox("Use One Session for Chunked Propagation")
        self.use_one_session_chunked_propagation_check.setChecked(self.use_one_session_chunked_propagation)
        self.use_one_session_chunked_propagation_check.setToolTip(
            "Keep tracker propagation chunked in the UI, but reuse one full-video SAM3 session across chunks."
        )
        self.use_one_session_chunked_propagation_check.toggled.connect(
            self._on_use_one_session_chunked_propagation_toggled
        )
        experimental_layout.addWidget(self.use_one_session_chunked_propagation_check)
        experimental_layout.addLayout(experimental_form)
        prompt_layout.addStretch(1)
        processing_layout.addStretch(1)
        experimental_layout.addStretch(1)

        control_tabs.addTab(prompt_tab, "Prompting")
        control_tabs.addTab(processing_tab, "Segment / Propagate")
        control_tabs.addTab(experimental_tab, "Experimental Features")

        self._advanced_widgets = [
            self.prompt_mode_label,
            self.prompt_mode_combo,
            self.segment_btn,
            self.show_prompts_check,
        ]
        self._set_advanced_controls_visible(True)
        self.mode_label.setText("Mode: Box Annotation")

        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        self._root_splitter = splitter
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 11)
        splitter.setStretchFactor(2, 5)
        geom = self._get_available_screen_geometry()
        avail_w = max(1, geom.width())
        left_w = 200
        right_w = int(avail_w * 0.40)
        right_w = max(360, min(520, right_w))
        center_w = max(1, avail_w - left_w - right_w)
        splitter.setSizes([left_w, center_w, right_w])

        self.setCentralWidget(root)
        self._setup_shortcuts()
        self._install_focus_return_on_widget(self.checkpoint_combo)
        self._install_focus_return_on_widget(self.frame_jump_spin)
        self._install_focus_return_on_widget(self.text_prompt_input)
        self._install_focus_return_on_widget(self.autosave_minutes_spin)
        self._install_focus_return_on_widget(self.segmentation_opacity_spin)
        self._install_focus_return_on_widget(self.box_line_thickness_spin)
        self._install_focus_return_on_widget(self.n_propagate_spin)
        self._install_focus_return_on_widget(self.chunks_spin)
        self._install_focus_return_on_widget(self.target_frame_spin)
        self._install_focus_return_on_widget(self.propagation_mode_combo)
        self._install_focus_return_on_widget(self.recondition_every_nth_frame_spin)
        self._install_focus_return_on_widget(self.recondition_high_conf_thresh_spin)
        self._install_focus_return_on_widget(self.recondition_high_iou_thresh_spin)
        self._install_focus_return_on_widget(self.use_one_session_chunked_propagation_check)
        self._sync_propagation_mode_controls()
        self._status_label = QLabel("Load a frame directory to start")
        self._status_progress = QProgressBar()
        self._status_progress.setMinimumWidth(200)
        self._status_progress.setRange(0, 1)
        self._status_progress.setValue(0)
        self._coords_label = QLabel("")
        self._coords_label.setMinimumWidth(160)
        self.statusBar().addWidget(self._status_label, 1)
        self.statusBar().addWidget(self._coords_label)
        self.statusBar().addPermanentWidget(self._status_progress)

    def _on_prompt_mode_changed(self, idx: int) -> None:
        self.current_prompt_positive = idx == 0

    def _on_auto_propagate_next_toggled(self, checked: bool) -> None:
        self.auto_propagate_next = checked
        if not checked:
            self._cancel_prefetch(restart=False)

    def _on_propagation_mode_changed(self, _idx: int) -> None:
        if not hasattr(self, "propagation_mode_combo"):
            return
        data = self.propagation_mode_combo.currentData()
        self.propagation_mode = str(data) if data is not None else "tracker"
        self._cancel_prefetch(restart=False)
        self._prefetch_cached_frame_idx = None
        self._prefetch_cached_seed_idx = None
        self._prefetch_cached_version = None
        self._prefetch_provenance_by_frame.clear()
        self._sync_propagation_mode_controls()

    def _on_translate_prompts_toggled(self, checked: bool) -> None:
        self.translate_prompts_on_propagation = checked

    def _on_use_point_prompts_for_propagation_toggled(self, checked: bool) -> None:
        self.use_point_prompts_for_propagation = checked

    def _on_use_one_session_chunked_propagation_toggled(self, checked: bool) -> None:
        self.use_one_session_chunked_propagation = checked
        if not checked:
            self._one_session_chunk_prompt_version = None

    def _on_experimental_settings_changed(self, _value) -> None:
        if hasattr(self, "recondition_every_nth_frame_spin"):
            self.recondition_every_nth_frame = int(self.recondition_every_nth_frame_spin.value())
        if hasattr(self, "recondition_high_conf_thresh_spin"):
            self.recondition_high_conf_thresh = float(self.recondition_high_conf_thresh_spin.value())
        if hasattr(self, "recondition_high_iou_thresh_spin"):
            self.recondition_high_iou_thresh = float(self.recondition_high_iou_thresh_spin.value())
        self._apply_experimental_settings_live()

    def _apply_experimental_settings_live(self) -> None:
        if self._sam_worker is None or not self._sam_ready:
            return
        task_id = self._enqueue_sam_task(
            "update_experimental_settings",
            {
                "recondition_every_nth_frame": self.recondition_every_nth_frame,
                "recondition_high_conf_thresh": self.recondition_high_conf_thresh,
                "recondition_high_iou_thresh": self.recondition_high_iou_thresh,
            },
            priority=True,
        )
        self._sam_task_contexts[task_id] = {"kind": "update_experimental_settings"}
        self._wait_for_sam_task(task_id)

    def _sync_experimental_controls(self) -> None:
        if hasattr(self, "recondition_every_nth_frame_spin"):
            self.recondition_every_nth_frame_spin.blockSignals(True)
            self.recondition_every_nth_frame_spin.setValue(int(self.recondition_every_nth_frame))
            self.recondition_every_nth_frame_spin.blockSignals(False)
        if hasattr(self, "recondition_high_conf_thresh_spin"):
            self.recondition_high_conf_thresh_spin.blockSignals(True)
            self.recondition_high_conf_thresh_spin.setValue(float(self.recondition_high_conf_thresh))
            self.recondition_high_conf_thresh_spin.blockSignals(False)
        if hasattr(self, "recondition_high_iou_thresh_spin"):
            self.recondition_high_iou_thresh_spin.blockSignals(True)
            self.recondition_high_iou_thresh_spin.setValue(float(self.recondition_high_iou_thresh))
            self.recondition_high_iou_thresh_spin.blockSignals(False)
        if hasattr(self, "use_one_session_chunked_propagation_check"):
            self.use_one_session_chunked_propagation_check.blockSignals(True)
            self.use_one_session_chunked_propagation_check.setChecked(
                bool(self.use_one_session_chunked_propagation)
            )
            self.use_one_session_chunked_propagation_check.blockSignals(False)

    def _on_propagation_target_toggled(self, checked: bool) -> None:
        self.use_target_frame = checked
        self.target_frame_spin.setEnabled(checked)
        self.chunks_spin.setEnabled(not checked)
        if checked and self.frame_paths:
            current_value = self.current_frame_idx + 1
            if self.target_frame_spin.value() <= current_value:
                self.target_frame_spin.setValue(min(len(self.frame_paths), current_value + 1))
        self._update_target_chunk_label()

    def _on_propagation_target_changed(self, _value: int) -> None:
        self._update_target_chunk_label()

    def _compute_target_chunk_count(
        self,
        current_frame_idx: int,
        target_frame_idx: int,
        n_frames: int,
    ) -> Optional[int]:
        if n_frames <= 1 or target_frame_idx <= current_frame_idx:
            return None
        stride = n_frames - 1
        distance = target_frame_idx - current_frame_idx
        return int(math.ceil(distance / stride))

    def _update_target_chunk_label(self) -> None:
        if not hasattr(self, "computed_chunks_label"):
            return
        if not self.use_target_frame or not self.frame_paths:
            self.computed_chunks_label.setText("Chunks: -")
            return
        target_idx = self.target_frame_spin.value() - 1
        chunks = self._compute_target_chunk_count(
            self.current_frame_idx,
            target_idx,
            int(self.n_propagate_spin.value()),
        )
        if chunks is None:
            self.computed_chunks_label.setText("Chunks: -")
        else:
            self.computed_chunks_label.setText(f"Chunks: {chunks}")

    def request_stop_propagation(self) -> None:
        if not self._propagation_busy:
            if self._pending_propagation is not None:
                self._clear_pending_propagation_state()
                self._set_status("Propagation stopped.", progress=0, total=1)
            return
        self._propagation_stop_requested = True
        if self._sam_worker is not None:
            self._sam_worker.cancel_propagation()

    def _on_autosave_toggled(self, checked: bool) -> None:
        if checked:
            if self._session_dir is None:
                QMessageBox.information(self, "Auto-save", "Save a session once to choose a folder before auto-save.")
                self.autosave_check.blockSignals(True)
                self.autosave_check.setChecked(False)
                self.autosave_check.blockSignals(False)
                return
            self._start_autosave_timer()
        else:
            self._autosave_timer.stop()

    def _on_autosave_interval_changed(self, _value: int) -> None:
        if self.autosave_check.isChecked():
            self._start_autosave_timer()

    def _start_autosave_timer(self) -> None:
        interval_ms = int(self.autosave_minutes_spin.value() * 60 * 1000)
        self._autosave_timer.stop()
        self._autosave_timer.start(interval_ms)

    def _ensure_autosave_running_if_enabled(self) -> None:
        if self.autosave_check.isChecked() and self._session_dir is not None:
            self._start_autosave_timer()

    def _handle_autosave(self) -> None:
        if self._session_dir is None or not self.frame_paths:
            return
        self._save_session(self._session_dir)

    def _on_object_lock_toggled(self, obj_id: int, checked: bool) -> None:
        self._set_active_object_by_id(obj_id)
        frame_boxes = self.box_prompts_by_frame_obj.get(self.current_frame_idx, {})
        if obj_id not in frame_boxes:
            self._sync_current_box_lock_check()
            return
        self._set_current_box_locked(self.current_frame_idx, obj_id, checked)
        self._sync_current_box_lock_check()
        self.refresh_point_list()
        self._render_current_frame()

    def _on_lock_all_boxes_toggled(self, checked: bool) -> None:
        frame_boxes = self.box_prompts_by_frame_obj.get(self.current_frame_idx, {})
        target_obj_ids = [obj_id for obj_id in frame_boxes.keys() if self._is_object_visible(obj_id)]
        if not target_obj_ids:
            self._sync_current_box_lock_check()
            return
        for obj_id in target_obj_ids:
            self._set_current_box_locked(self.current_frame_idx, obj_id, checked)
        self._sync_current_box_lock_check()
        self.refresh_point_list()
        self._render_current_frame()

    def _set_advanced_controls_visible(self, visible: bool) -> None:
        for widget in self._advanced_widgets:
            widget.setVisible(visible)

    def _on_object_selection_changed(self, current: Optional[QListWidgetItem], _previous: Optional[QListWidgetItem]) -> None:
        if current is None:
            self.active_object_id = None
        else:
            self.active_object_id = int(current.data(Qt.UserRole))
        self._sync_selected_box_state()
        self._refresh_object_list_visuals()
        self.refresh_point_list()
        self._sync_current_box_lock_check()
        self._render_current_frame()

    def _setup_shortcuts(self) -> None:
        prev_action = QAction(self)
        prev_action.setShortcut(QKeySequence(Qt.Key_Left))
        prev_action.triggered.connect(self.go_prev_frame)
        self.addAction(prev_action)

        prev_action_alt = QAction(self)
        prev_action_alt.setShortcut(QKeySequence("A"))
        prev_action_alt.triggered.connect(self.go_prev_frame)
        self.addAction(prev_action_alt)

        next_action = QAction(self)
        next_action.setShortcut(QKeySequence(Qt.Key_Right))
        next_action.triggered.connect(self.go_next_frame_shortcut)
        self.addAction(next_action)

        next_action_alt = QAction(self)
        next_action_alt.setShortcut(QKeySequence("D"))
        next_action_alt.triggered.connect(self.go_next_frame_shortcut)
        self.addAction(next_action_alt)

        propagate_action = QAction(self)
        propagate_action.setShortcut(QKeySequence("P"))
        propagate_action.triggered.connect(self.propagate_next_frame)
        self.addAction(propagate_action)

        toggle_flag_action = QAction(self)
        toggle_flag_action.setShortcut(QKeySequence("F"))
        toggle_flag_action.triggered.connect(self.toggle_current_frame_flag)
        self.addAction(toggle_flag_action)

        prev_flag_action = QAction(self)
        prev_flag_action.setShortcut(QKeySequence("Shift+Left"))
        prev_flag_action.triggered.connect(self.go_prev_flagged_frame)
        self.addAction(prev_flag_action)

        next_flag_action = QAction(self)
        next_flag_action.setShortcut(QKeySequence("Shift+Right"))
        next_flag_action.triggered.connect(self.go_next_flagged_frame)
        self.addAction(next_flag_action)

        delete_prompt_action = QAction(self)
        delete_prompt_action.setShortcut(QKeySequence(Qt.Key_Delete))
        delete_prompt_action.triggered.connect(self._delete_current_prompt_shortcut)
        self.addAction(delete_prompt_action)

        toggle_auto_prop_action = QAction(self)
        toggle_auto_prop_action.setShortcut(QKeySequence("N"))
        toggle_auto_prop_action.triggered.connect(self._toggle_auto_propagate_shortcut)
        self.addAction(toggle_auto_prop_action)

        cycle_prop_mode_action = QAction(self)
        cycle_prop_mode_action.setShortcut(QKeySequence("M"))
        cycle_prop_mode_action.triggered.connect(self._cycle_propagation_mode_shortcut)
        self.addAction(cycle_prop_mode_action)

        undo_action = QAction(self)
        undo_action.setShortcut(QKeySequence.Undo)
        undo_action.triggered.connect(self.undo_last_prompt_change)
        self.addAction(undo_action)

    def _focus_widget_is_text_entry(self) -> bool:
        focus_widget = QApplication.focusWidget()
        if focus_widget is None:
            return False
        if isinstance(focus_widget, QLineEdit):
            return True
        parent = focus_widget.parentWidget()
        while parent is not None:
            if isinstance(parent, (QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, ArrowSpinBox)):
                return True
            parent = parent.parentWidget()
        return False

    def _delete_current_prompt_shortcut(self) -> None:
        if self._focus_widget_is_text_entry():
            return
        self.remove_selected_prompt()

    def _toggle_auto_propagate_shortcut(self) -> None:
        if self._focus_widget_is_text_entry():
            return
        self.auto_propagate_next_check.toggle()

    def _cycle_propagation_mode_shortcut(self) -> None:
        if self._focus_widget_is_text_entry():
            return
        count = self.propagation_mode_combo.count()
        if count <= 1:
            return
        next_idx = (self.propagation_mode_combo.currentIndex() + 1) % count
        self.propagation_mode_combo.setCurrentIndex(next_idx)

    def show_hotkeys_list(self) -> None:
        QMessageBox.information(
            self,
            "Hotkeys List",
            "\n".join(
                [
                    "Left / A: Previous frame",
                    "Right / D: Next frame",
                    "P: Propagate",
                    "F: Flag / Unflag current frame",
                    "Shift+Left: Previous flagged frame",
                    "Shift+Right: Next flagged frame",
                    "Delete: Delete selected box / point prompt",
                    "N: Toggle Auto Propagate Next Frame",
                    "M: Cycle Propagation Mode",
                    "Ctrl+Z: Undo last prompt change",
                    "Enter / Esc in text boxes: Return focus to canvas",
                ]
            ),
        )

    def show_instructions_walkthrough(self) -> None:
        QMessageBox.information(
            self,
            "Instructions / Walkthrough",
            "\n".join(
                [
                    "1. Load a frame directory and add one or more objects.",
                    "2. Select an object, then add point prompts or draw a box on the current frame.",
                    "3. Use Segment to refine the current frame.",
                    "4. Enable Prop for objects you want to carry forward.",
                    "5. Use Propagate to track forward across frames.",
                    "6. Double-click a visible box to select that object and enter box mode.",
                    "7. Use the flagged-frames panel to mark frames that need review.",
                    "8. Auto-save defaults to every 5 minutes once a session folder exists.",
                ]
            ),
        )

    def _flagged_frame_display_text(self, frame_idx: int) -> str:
        label = f"Frame {frame_idx + 1}"
        if 0 <= frame_idx < len(self.frame_paths):
            label += f" - {self.frame_paths[frame_idx].name}"
        return label

    def _sorted_flagged_frames(self) -> List[int]:
        return sorted(
            frame_idx
            for frame_idx in self.flagged_frame_indices
            if 0 <= frame_idx < len(self.frame_paths)
        )

    def _refresh_flagged_frame_list(self) -> None:
        if not hasattr(self, "flagged_frames_list"):
            return
        flagged_frames = self._sorted_flagged_frames()
        self.flagged_frames_list.blockSignals(True)
        self.flagged_frames_list.clear()
        for frame_idx in flagged_frames:
            item = QListWidgetItem(self._flagged_frame_display_text(frame_idx))
            item.setData(Qt.UserRole, frame_idx)
            self.flagged_frames_list.addItem(item)
        self.flagged_frames_list.blockSignals(False)
        self._sync_flagged_frame_selection()

    def _sync_flagged_frame_selection(self) -> None:
        if not hasattr(self, "flagged_frames_list"):
            return
        self.flagged_frames_list.blockSignals(True)
        target_row = -1
        for row in range(self.flagged_frames_list.count()):
            item = self.flagged_frames_list.item(row)
            if item is not None and int(item.data(Qt.UserRole)) == self.current_frame_idx:
                target_row = row
                break
        self.flagged_frames_list.setCurrentRow(target_row)
        self.flagged_frames_list.blockSignals(False)

    def _jump_to_flagged_frame(self, direction: int) -> None:
        if not self.frame_paths:
            return
        flagged_frames = self._sorted_flagged_frames()
        if not flagged_frames:
            self._set_status("No flagged frames.", progress=0, total=1)
            return
        if direction > 0:
            for frame_idx in flagged_frames:
                if frame_idx > self.current_frame_idx:
                    self._set_current_frame_idx(frame_idx)
                    return
            self._set_current_frame_idx(flagged_frames[0])
            return
        for frame_idx in reversed(flagged_frames):
            if frame_idx < self.current_frame_idx:
                self._set_current_frame_idx(frame_idx)
                return
        self._set_current_frame_idx(flagged_frames[-1])

    def toggle_current_frame_flag(self) -> None:
        if not self.frame_paths:
            return
        if self.current_frame_idx in self.flagged_frame_indices:
            self.flagged_frame_indices.remove(self.current_frame_idx)
            status = f"Removed flag from frame {self.current_frame_idx + 1}."
        else:
            self.flagged_frame_indices.add(self.current_frame_idx)
            status = f"Flagged frame {self.current_frame_idx + 1}."
        self._refresh_flagged_frame_list()
        self._render_current_frame()
        self._set_status(status, progress=1, total=1)

    def go_prev_flagged_frame(self) -> None:
        self._jump_to_flagged_frame(-1)

    def go_next_flagged_frame(self) -> None:
        self._jump_to_flagged_frame(1)

    def _on_flagged_frame_selection_changed(
        self,
        current: Optional[QListWidgetItem],
        _previous: Optional[QListWidgetItem],
    ) -> None:
        if current is None:
            return
        frame_idx = current.data(Qt.UserRole)
        if frame_idx is None:
            return
        self._set_current_frame_idx(int(frame_idx))

    def _refresh_object_list_visuals(self) -> None:
        active_label = "Active: None"
        for i in range(self.object_list.count()):
            item = self.object_list.item(i)
            obj_id = int(item.data(Qt.UserRole))
            obj = self._find_object(obj_id)
            row = self._object_row_widgets.get(obj_id, {})
            label = row.get("label")
            if isinstance(label, QLabel):
                label.setText(self._object_display_text(obj_id))
            propagate_check = row.get("propagate_check")
            if isinstance(propagate_check, QCheckBox):
                has_current_prompt = self._object_has_current_frame_seed(obj_id)
                checked = self._is_object_enabled_for_current_frame_propagation(obj_id)
                propagate_check.blockSignals(True)
                propagate_check.setChecked(checked)
                propagate_check.setEnabled(self.object_list.isEnabled() and has_current_prompt)
                propagate_check.blockSignals(False)
            rename_btn = row.get("rename_btn")
            if isinstance(rename_btn, QToolButton):
                rename_btn.setEnabled(self.object_list.isEnabled())
            lock_check = row.get("lock_check")
            if isinstance(lock_check, QCheckBox):
                frame_boxes = self.box_prompts_by_frame_obj.get(self.current_frame_idx, {})
                has_box = obj_id in frame_boxes
                checked = has_box and self._is_current_box_locked(self.current_frame_idx, obj_id)
                lock_check.blockSignals(True)
                lock_check.setEnabled(self.object_list.isEnabled() and has_box)
                lock_check.setChecked(checked)
                lock_check.blockSignals(False)
            solo_btn = row.get("solo_btn")
            if isinstance(solo_btn, QToolButton):
                solo_btn.setText("Unsolo" if self.solo_object_id == obj_id else "Solo")
                solo_btn.setEnabled(self.object_list.isEnabled())
            hide_btn = row.get("hide_btn")
            if isinstance(hide_btn, QToolButton):
                hide_btn.setText("Show" if obj_id in self.hidden_obj_ids else "Hide")
                hide_btn.setEnabled(self.object_list.isEnabled())
            row_widget = row.get("widget")
            if isinstance(row_widget, QWidget):
                if obj_id == self.active_object_id:
                    row_widget.setStyleSheet(
                        "background-color: #1f4f7a; border-radius: 4px;"
                        " color: #ffffff;"
                        " QCheckBox { color: #ffffff; }"
                        " QLabel { color: #ffffff; }"
                        " QToolButton { color: #ffffff; }"
                    )
                else:
                    row_widget.setStyleSheet("background-color: transparent;")
            if obj_id == self.active_object_id:
                item.setBackground(QBrush(QColor("#1f4f7a")))
                item.setForeground(QBrush(QColor("#ffffff")))
                active_label = f"Active: {obj.name if obj else item.text()}"
            else:
                item.setBackground(QBrush())
                item.setForeground(QBrush())
        self.active_object_label.setText(active_label)

    def _on_view_settings_changed(self, _value=None) -> None:
        self.show_prompts = self.show_prompts_check.isChecked()
        self.show_segmentations = self.show_segmentations_check.isChecked()
        self.show_boxes = self.show_boxes_check.isChecked()
        self.show_box_titles = self.show_box_titles_check.isChecked()
        self.segmentation_opacity = float(self.segmentation_opacity_spin.value())
        self.box_line_thickness = int(self.box_line_thickness_spin.value())
        self._render_current_frame()

    def _object_display_text(self, obj_id: int) -> str:
        obj = self._find_object(obj_id)
        base = f"{obj.name} (id={obj_id})" if obj else str(obj_id)
        tracker_score = self._get_current_frame_tracker_score(obj_id)
        if tracker_score is not None:
            base += f" [trk={tracker_score:.2f}]"
        if self.solo_object_id == obj_id:
            base += " [solo]"
        elif obj_id in self.hidden_obj_ids:
            base += " [hidden]"
        return base

    def _get_current_frame_tracker_score(self, obj_id: int) -> Optional[float]:
        output = self.outputs_by_frame.get(self.current_frame_idx)
        if output is None:
            return None
        if obj_id not in output.obj_ids:
            return None
        idx = output.obj_ids.index(obj_id)
        if idx >= len(output.tracker_scores):
            return None
        return float(output.tracker_scores[idx])

    def _create_object_row_widget(self, obj_id: int) -> QWidget:
        row_widget = QWidget()
        layout = QHBoxLayout(row_widget)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(4)

        propagate_check = QCheckBox("Prop")
        propagate_check.setChecked(True)
        propagate_check.toggled.connect(lambda checked, oid=obj_id: self._on_object_propagate_toggled(oid, checked))
        layout.addWidget(propagate_check)

        label = QLabel(self._object_display_text(obj_id))
        label.setMinimumWidth(110)
        layout.addWidget(label, 1)

        rename_btn = QToolButton()
        rename_btn.setText("Rename")
        rename_btn.clicked.connect(lambda _checked=False, oid=obj_id: self.rename_object(oid))
        layout.addWidget(rename_btn)

        lock_check = QCheckBox("Lock")
        lock_check.toggled.connect(lambda checked, oid=obj_id: self._on_object_lock_toggled(oid, checked))
        layout.addWidget(lock_check)

        solo_btn = QToolButton()
        solo_btn.setText("Solo")
        solo_btn.clicked.connect(lambda _checked=False, oid=obj_id: self.toggle_solo_object(oid))
        layout.addWidget(solo_btn)

        hide_btn = QToolButton()
        hide_btn.setText("Hide")
        hide_btn.clicked.connect(lambda _checked=False, oid=obj_id: self.toggle_hide_object(oid))
        layout.addWidget(hide_btn)

        self._object_row_widgets[obj_id] = {
            "widget": row_widget,
            "propagate_check": propagate_check,
            "label": label,
            "rename_btn": rename_btn,
            "lock_check": lock_check,
            "solo_btn": solo_btn,
            "hide_btn": hide_btn,
        }
        return row_widget

    def _is_object_visible(self, obj_id: int) -> bool:
        if self.solo_object_id is not None:
            return obj_id == self.solo_object_id
        return obj_id not in self.hidden_obj_ids

    def _object_has_current_frame_seed(self, obj_id: int) -> bool:
        frame_prompts = self.prompts_by_frame_obj.get(self.current_frame_idx, {})
        if frame_prompts.get(obj_id):
            return True
        frame_boxes = self.box_prompts_by_frame_obj.get(self.current_frame_idx, {})
        if obj_id in frame_boxes:
            return True
        output = self.outputs_by_frame.get(self.current_frame_idx)
        return output is not None and obj_id in output.obj_ids

    def _set_manual_propagation_override(
        self,
        frame_idx: int,
        obj_id: int,
        enabled: Optional[bool],
    ) -> None:
        if enabled is None:
            per_obj = self.manual_propagation_overrides_by_frame_obj.get(frame_idx)
            if per_obj is None:
                return
            per_obj.pop(obj_id, None)
            if not per_obj:
                self.manual_propagation_overrides_by_frame_obj.pop(frame_idx, None)
            return
        self.manual_propagation_overrides_by_frame_obj.setdefault(frame_idx, {})[obj_id] = bool(enabled)

    def _is_object_enabled_for_current_frame_propagation(self, obj_id: int) -> bool:
        if not self._object_has_current_frame_seed(obj_id):
            return False
        override = self.manual_propagation_overrides_by_frame_obj.get(self.current_frame_idx, {}).get(obj_id)
        if override is None:
            return True
        return bool(override)

    def _on_object_propagate_toggled(self, obj_id: int, checked: bool) -> None:
        self._set_active_object_by_id(obj_id)
        if not self._object_has_current_frame_seed(obj_id):
            self._set_manual_propagation_override(self.current_frame_idx, obj_id, None)
            return
        if checked:
            self._set_manual_propagation_override(self.current_frame_idx, obj_id, None)
        else:
            self._set_manual_propagation_override(self.current_frame_idx, obj_id, False)

    def _set_active_object_by_id(self, obj_id: int) -> None:
        for i in range(self.object_list.count()):
            item = self.object_list.item(i)
            if item is not None and int(item.data(Qt.UserRole)) == obj_id:
                self.object_list.setCurrentItem(item)
                break

    def rename_object(self, obj_id: int) -> None:
        self._set_active_object_by_id(obj_id)
        obj = self._find_object(obj_id)
        if obj is None:
            return
        name, ok = QInputDialog.getText(self, "Rename Object", "Object name:", text=obj.name)
        if not ok:
            return
        name = name.strip()
        if not name:
            QMessageBox.warning(self, "Invalid name", "Object name cannot be empty.")
            return
        obj.name = name
        self._refresh_object_list_visuals()
        self._render_current_frame()

    def toggle_solo_object(self, obj_id: int) -> None:
        self._set_active_object_by_id(obj_id)
        if self.active_object_id is None:
            return
        if self.solo_object_id == obj_id:
            self.solo_object_id = None
        else:
            self.solo_object_id = obj_id
            self.hidden_obj_ids.discard(obj_id)
        self._refresh_object_list_visuals()
        self._render_current_frame()

    def toggle_hide_object(self, obj_id: int) -> None:
        self._set_active_object_by_id(obj_id)
        if self.active_object_id is None:
            return
        if obj_id in self.hidden_obj_ids:
            self.hidden_obj_ids.discard(obj_id)
        else:
            self.hidden_obj_ids.add(obj_id)
            if self.solo_object_id == obj_id:
                self.solo_object_id = None
        self._refresh_object_list_visuals()
        self._render_current_frame()

    def _on_point_list_selection_changed(self, _row: int) -> None:
        if self.frame_paths:
            self._render_current_frame()

    def _set_status(self, text: str, *, progress: Optional[int] = None, total: Optional[int] = None, indeterminate: bool = False) -> None:
        self._status_label.setText(text)
        if indeterminate:
            self._status_progress.setRange(0, 0)
            return
        if total is not None:
            self._status_progress.setRange(0, max(1, total))
        if progress is not None:
            self._status_progress.setValue(progress)

    def _on_sam_worker_initialized(self, ok: bool, message: str) -> None:
        self._sam_ready = ok
        self._sam_init_error = message
        loop = self._sam_waiting.get("__init__")
        if loop is not None:
            loop.quit()

    def _enqueue_sam_task(self, task_type: str, payload: Dict[str, object], priority: bool = False) -> str:
        self._sam_task_counter += 1
        task_id = f"{task_type}:{self._sam_task_counter}"
        if self._sam_worker is None:
            return task_id
        self.task_requested.emit(task_id, task_type, payload, priority)
        return task_id

    def _wait_for_sam_task(self, task_id: str) -> object:
        loop = QEventLoop()
        self._sam_waiting[task_id] = loop
        loop.exec()
        self._sam_waiting.pop(task_id, None)
        return self._sam_task_results.pop(task_id, None)

    def _reset_prefetch_state(self) -> None:
        self._cancel_prefetch(restart=False)
        self._prefetch_prompt_version = 0
        self._prefetch_cached_frame_idx = None
        self._prefetch_cached_seed_idx = None
        self._prefetch_cached_version = None
        self._prefetch_provenance_by_frame.clear()

    def _is_tracker_propagation_mode(self) -> bool:
        return self.propagation_mode == "tracker"

    def _sync_propagation_mode_controls(self) -> None:
        if hasattr(self, "propagation_mode_combo"):
            idx = self.propagation_mode_combo.findData(self.propagation_mode)
            if idx >= 0 and self.propagation_mode_combo.currentIndex() != idx:
                self.propagation_mode_combo.blockSignals(True)
                self.propagation_mode_combo.setCurrentIndex(idx)
                self.propagation_mode_combo.blockSignals(False)
        if hasattr(self, "translate_prompts_check"):
            self.translate_prompts_check.setEnabled(self._is_tracker_propagation_mode())
        if hasattr(self, "use_point_prompts_for_propagation_check"):
            self.use_point_prompts_for_propagation_check.setEnabled(self._is_tracker_propagation_mode())

    def _note_prompt_change_and_prefetch(self) -> None:
        if self.active_object_id is not None:
            self._invalidate_prefetched_next_frame_for_objects(
                seed_frame_idx=self.current_frame_idx,
                obj_ids={self.active_object_id},
            )
        self._prefetch_prompt_version += 1
        self._prefetch_cached_frame_idx = None
        self._prefetch_cached_seed_idx = None
        self._prefetch_cached_version = None
        self._last_prompt_edit_frame = self.current_frame_idx
        next_idx = min(len(self.frame_paths) - 1, self.current_frame_idx + 1) if self.frame_paths else None
        if next_idx is not None:
            self._output_version_by_frame.pop(next_idx, None)
        self._schedule_prefetch_for_next_frame()

    def _has_valid_prefetch(self, target_frame_idx: int) -> bool:
        return (
            self._prefetch_cached_frame_idx == target_frame_idx
            and self._prefetch_cached_seed_idx == target_frame_idx - 1
            and self._prefetch_cached_version == self._prefetch_prompt_version
        )

    def _invalidate_prefetched_next_frame_for_objects(self, seed_frame_idx: int, obj_ids: set[int]) -> None:
        if not self.frame_paths or not obj_ids:
            return
        next_idx = seed_frame_idx + 1
        if next_idx >= len(self.frame_paths):
            return
        provenance = self._prefetch_provenance_by_frame.get(next_idx)
        if not provenance or int(provenance.get("seed_frame_idx", -1)) != seed_frame_idx:
            return

        prefetched_obj_ids = {int(obj_id) for obj_id in provenance.get("obj_ids", [])}
        target_obj_ids = prefetched_obj_ids & {int(obj_id) for obj_id in obj_ids}
        if not target_obj_ids:
            return

        for obj_id in target_obj_ids:
            self._remove_object_annotations_from_frame(next_idx, obj_id)

        remaining_obj_ids = sorted(prefetched_obj_ids - target_obj_ids)
        if remaining_obj_ids:
            provenance["obj_ids"] = remaining_obj_ids
        else:
            self._prefetch_provenance_by_frame.pop(next_idx, None)
        if self.current_frame_idx == next_idx:
            self.refresh_point_list()
            self._sync_current_box_lock_check()
            self._render_current_frame()

    def _cancel_prefetch(self, *, restart: bool) -> None:
        if not self._prefetch_busy:
            self._prefetch_pending_restart = False
            return
        self._prefetch_cancel_requested = True
        self._prefetch_pending_restart = restart
        if self._sam_worker is not None:
            QMetaObject.invokeMethod(self._sam_worker, "cancel_prefetch", Qt.QueuedConnection)


    def _schedule_prefetch_for_next_frame(self) -> None:
        if not self.auto_propagate_next:
            return
        if not self._is_tracker_propagation_mode():
            return
        if self._propagation_busy:
            return
        if not self.frame_paths or self.current_frame_idx >= len(self.frame_paths) - 1:
            return
        next_idx = min(len(self.frame_paths) - 1, self.current_frame_idx + 1)
        if (
            next_idx in self.outputs_by_frame
            and self._output_version_by_frame.get(next_idx, -1) == self._prefetch_prompt_version
        ):
            return
        if self._has_valid_prefetch(next_idx):
            return
        if self._prefetch_busy:
            self._cancel_prefetch(restart=True)
            return
        if not self._ensure_sam_initialized():
            return
        enabled_obj_ids = self._get_enabled_propagation_obj_ids()
        if not enabled_obj_ids:
            return
        prompt_payload = self._build_seed_prompts(
            seed_frame_idx=self.current_frame_idx,
            use_carryover_sampling=False,
            enabled_obj_ids=enabled_obj_ids,
        )
        if prompt_payload is None:
            return

        target_frame_idx = next_idx
        self._prefetch_busy = True
        self._prefetch_cancel_requested = False
        self._prefetch_pending_restart = False
        self._prefetch_target_frame_idx = target_frame_idx
        self._prefetch_seed_frame_idx = self.current_frame_idx
        self._prefetch_active_version = self._prefetch_prompt_version

        task_id = self._enqueue_sam_task(
            "prefetch",
            {
                "seed_frame_idx": self.current_frame_idx,
                "n_frames": 2,
                "frame_paths": self.frame_paths,
                "prompt_payload": prompt_payload,
            },
        )
        self._sam_task_contexts[task_id] = {
            "kind": "prefetch",
            "target_frame_idx": target_frame_idx,
            "seed_frame_idx": self.current_frame_idx,
            "version": self._prefetch_prompt_version,
        }

    def on_image_hover(self, ui_x: Optional[float], ui_y: Optional[float]) -> None:
        if ui_x is None or ui_y is None or not self.frame_paths:
            self._coords_label.setText("")
            return
        mapped = self._map_ui_to_image_xy(ui_x, ui_y)
        if mapped is None:
            self._coords_label.setText("")
            return
        x_img, y_img = mapped
        self._coords_label.setText(f"X: {x_img}  Y: {y_img}")

    def _set_propagation_ui_enabled(self, enabled: bool) -> None:
        self.propagate_btn.setEnabled(enabled)
        self.segment_btn.setEnabled(enabled)
        self.add_obj_btn.setEnabled(enabled)
        self.remove_obj_btn.setEnabled(enabled)
        self.object_list.setEnabled(enabled)
        self.text_prompt_input.setEnabled(enabled)
        self.generate_text_prompt_btn.setEnabled(enabled)
        self.accept_text_prompt_btn.setEnabled(enabled)
        self.clear_text_prompt_btn.setEnabled(enabled)
        self.text_prompt_list.setEnabled(enabled)
        for row in self._object_row_widgets.values():
            propagate_check = row.get("propagate_check")
            rename_btn = row.get("rename_btn")
            solo_btn = row.get("solo_btn")
            hide_btn = row.get("hide_btn")
            if isinstance(propagate_check, QCheckBox):
                propagate_check.setEnabled(enabled)
            if isinstance(rename_btn, QToolButton):
                rename_btn.setEnabled(enabled)
            if isinstance(solo_btn, QToolButton):
                solo_btn.setEnabled(enabled)
            if isinstance(hide_btn, QToolButton):
                hide_btn.setEnabled(enabled)
        self.point_list.setEnabled(enabled)
        self.remove_point_btn.setEnabled(enabled)
        self.clear_points_btn.setEnabled(enabled)
        self.lock_all_boxes_check.setEnabled(enabled)
        self.prompt_mode_combo.setEnabled(enabled)
        self.fit_view_btn.setEnabled(enabled)
        self.frame_slider.setEnabled(enabled)
        self.frame_jump_spin.setEnabled(enabled)
        self.autosave_check.setEnabled(enabled)
        self.autosave_minutes_spin.setEnabled(enabled)
        self.propagation_mode_combo.setEnabled(enabled)
        self.recondition_every_nth_frame_spin.setEnabled(enabled)
        self.recondition_high_conf_thresh_spin.setEnabled(enabled)
        self.recondition_high_iou_thresh_spin.setEnabled(enabled)
        self.use_one_session_chunked_propagation_check.setEnabled(enabled)
        running = self._propagation_busy
        pending = self._pending_propagation is not None
        self.stop_propagate_btn.setEnabled(running or pending)
        self.use_target_frame_check.setEnabled(enabled)
        self.target_frame_spin.setEnabled(enabled and self.use_target_frame_check.isChecked())
        self.chunks_spin.setEnabled(enabled and not self.use_target_frame_check.isChecked())
        self.translate_prompts_check.setEnabled(enabled and self._is_tracker_propagation_mode())
        self.use_point_prompts_for_propagation_check.setEnabled(enabled and self._is_tracker_propagation_mode())
        self._sync_current_box_lock_check()

    def _sync_current_box_lock_check(self) -> None:
        frame_boxes = self.box_prompts_by_frame_obj.get(self.current_frame_idx, {})
        visible_obj_ids = [obj_id for obj_id in frame_boxes.keys() if self._is_object_visible(obj_id)]
        has_boxes = bool(visible_obj_ids)
        all_locked = has_boxes and all(
            self._is_current_box_locked(self.current_frame_idx, obj_id) for obj_id in visible_obj_ids
        )
        self.lock_all_boxes_check.blockSignals(True)
        self.lock_all_boxes_check.setEnabled(has_boxes)
        self.lock_all_boxes_check.setChecked(all_locked)
        self.lock_all_boxes_check.blockSignals(False)
        self._refresh_object_list_visuals()

    def _use_point_prompt_mode(self) -> bool:
        return self.prompt_mode_combo.currentIndex() != 2

    def load_frame_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Frame Directory")
        if not directory:
            return

        self._set_status("Loading frames...", indeterminate=True)
        dir_path = Path(directory)
        frame_paths = sorted([p for p in dir_path.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        if not frame_paths:
            QMessageBox.warning(self, "No frames", "Selected directory has no supported image files.")
            self._set_status("No supported images found.", progress=0, total=1)
            return

        self._checkpoint_path = self._read_optional_combo_path(self.checkpoint_combo)
        self._bpe_path = None

        if not self._initialize_sam_worker(show_errors=True):
            return
        self._enqueue_sam_task("close_session", {}, priority=True)

        self.image_dir = dir_path
        self.frame_paths = frame_paths
        self.current_frame_idx = 0
        self.objects.clear()
        self.object_list.clear()
        self._object_row_widgets.clear()
        self.next_obj_id = 1
        self.active_object_id = None
        self.hidden_obj_ids.clear()
        self.solo_object_id = None
        self.flagged_frame_indices.clear()
        self.prompts_by_frame_obj.clear()
        self.box_prompts_by_frame_obj.clear()
        self.box_locked_by_frame_obj.clear()
        self.outputs_by_frame.clear()
        self.manual_propagation_overrides_by_frame_obj.clear()
        self._clear_text_prompt_proposals_if_needed()
        self._output_version_by_frame.clear()
        self._prefetch_provenance_by_frame.clear()
        self._undo_state = None
        self._last_prompt_edit_frame = None
        self.segment_mode = False
        self.mode_label.setText("Mode: Box Annotation")
        self._clear_pending_propagation_state()
        self._box_draw_start_xy = None
        self._box_draw_start_ui_xy = None
        self._display_image_size = (0, 0)
        self._zoom_multiplier = 1.0
        self._pan_offset_ui = (0.0, 0.0)
        self._pan_drag_last_ui_xy = None
        self._box_edit_active = False
        self._box_drag_mode = None
        self._box_drag_reference_box = None
        self._box_drag_offset_xy = (0, 0)
        self._selected_box_obj_id = None
        self._box_rubber_band.hide()
        self.auto_propagate_next = False
        self.auto_propagate_next_check.setChecked(False)
        self.propagation_mode = "tracker"
        self._sync_propagation_mode_controls()
        self.use_target_frame = False
        self.use_target_frame_check.setChecked(False)
        self.prompt_mode_combo.setCurrentIndex(2)
        self._sync_frame_navigation_controls()
        self._refresh_flagged_frame_list()
        self._refresh_object_list_visuals()
        self._sync_current_box_lock_check()

        self._render_current_frame()
        self._set_status(f"Loaded {len(self.frame_paths)} frames from {dir_path}", progress=1, total=1)
        self._reset_prefetch_state()
        self._session_dir = None

    def _read_optional_combo_path(self, combo: QComboBox) -> Optional[str]:
        text = combo.currentText().strip()
        if not text or text.lower().startswith("use default"):
            return None
        return text

    def add_object(self) -> None:
        name, ok = QInputDialog.getText(self, "Add Object", "Object name:")
        if not ok:
            return
        name = name.strip()
        if not name:
            QMessageBox.warning(self, "Invalid name", "Object name cannot be empty.")
            return
        self._create_object_entry(name)

    def _create_object_entry(self, name: str) -> ObjectInfo:
        obj_id = self.next_obj_id
        self.next_obj_id += 1
        color = self._color_for_obj(obj_id)
        obj = ObjectInfo(obj_id=obj_id, name=name, color_bgr=color)
        self.objects.append(obj)

        item = QListWidgetItem()
        item.setData(Qt.UserRole, obj.obj_id)
        self.object_list.addItem(item)
        self.object_list.setItemWidget(item, self._create_object_row_widget(obj.obj_id))
        self.object_list.setCurrentItem(item)
        self._refresh_object_list_visuals()
        return obj

    def generate_objects_from_text_prompt(self) -> None:
        if not self.frame_paths:
            QMessageBox.information(self, "No frames loaded", "Load a frame directory first.")
            return
        text_prompt = self.text_prompt_input.text().strip()
        if not text_prompt:
            QMessageBox.information(self, "No text prompt", "Enter a text prompt first.")
            return
        if self._prefetch_busy:
            self._cancel_prefetch(restart=False)
        if not self._ensure_sam_initialized():
            return
        task_id = self._enqueue_sam_task(
            "text_prompt",
            {
                "frame_idx": self.current_frame_idx,
                "frame_path": self.frame_paths[self.current_frame_idx],
                "text_prompt": text_prompt,
            },
        )
        self._sam_task_contexts[task_id] = {"kind": "text_prompt"}
        self._set_status("Generating text prompt proposals...", progress=0, total=1)
        self._wait_for_sam_task(task_id)

    def accept_selected_text_prompt_proposals(self) -> None:
        if self._text_prompt_proposals_frame_idx != self.current_frame_idx or not self._text_prompt_proposals:
            QMessageBox.information(self, "No proposals", "Generate text prompt proposals on this frame first.")
            return
        selected_indices = self._selected_text_prompt_proposal_indices()
        if not selected_indices:
            QMessageBox.information(self, "No proposals selected", "Check at least one proposal to accept.")
            return
        frame_size = self._get_current_frame_size()
        if frame_size is None:
            QMessageBox.warning(self, "Missing frame size", "Could not determine the current frame size.")
            return
        existing_output = self.outputs_by_frame.get(
            self.current_frame_idx,
            SamFrameOutput(obj_ids=[], masks=[], boxes_xywh_norm=[], scores=[], tracker_scores=[]),
        )
        accepted_obj_ids: List[int] = []
        prompt_label = self._text_prompt_last_prompt or self.text_prompt_input.text().strip() or "Object"
        for proposal_idx in selected_indices:
            if proposal_idx < 0 or proposal_idx >= len(self._text_prompt_proposals):
                continue
            proposal = self._text_prompt_proposals[proposal_idx]
            obj_name = next_prompt_object_name(prompt_label, (obj.name for obj in self.objects))
            obj = self._create_object_entry(obj_name)
            accepted_obj_ids.append(obj.obj_id)
            self._commit_text_prompt_proposal(obj.obj_id, proposal, frame_size, existing_output)
            existing_output = self.outputs_by_frame.get(self.current_frame_idx, existing_output)
        if not accepted_obj_ids:
            QMessageBox.information(self, "No proposals accepted", "No valid proposals were accepted.")
            return
        self._note_prompt_change_and_prefetch()
        self._set_status(f"Accepted {len(accepted_obj_ids)} text prompt object(s).", progress=1, total=1)
        self.clear_text_prompt_proposals()
        self._render_current_frame()

    def clear_text_prompt_proposals(self) -> None:
        self._text_prompt_proposals_frame_idx = None
        self._text_prompt_last_prompt = ""
        self._text_prompt_proposals = []
        if hasattr(self, "text_prompt_list"):
            self.text_prompt_list.clear()
        self._render_current_frame()

    def _clear_text_prompt_proposals_if_needed(self, frame_idx: Optional[int] = None) -> None:
        if not self._text_prompt_proposals:
            return
        if frame_idx is None or self._text_prompt_proposals_frame_idx != frame_idx:
            self._text_prompt_proposals_frame_idx = None
            self._text_prompt_last_prompt = ""
            self._text_prompt_proposals = []
            if hasattr(self, "text_prompt_list"):
                self.text_prompt_list.clear()

    def _selected_text_prompt_proposal_indices(self) -> List[int]:
        selected: List[int] = []
        for idx in range(self.text_prompt_list.count()):
            item = self.text_prompt_list.item(idx)
            if item is None:
                continue
            if item.checkState() == Qt.Checked:
                selected.append(idx)
        return selected

    def _refresh_text_prompt_list(self) -> None:
        if not hasattr(self, "text_prompt_list"):
            return
        self.text_prompt_list.clear()
        for proposal in self._text_prompt_proposals:
            x1, y1, x2, y2 = proposal.box_xyxy_px
            item = QListWidgetItem(
                f"{proposal.proposal_idx + 1}. score={proposal.score:.3f} box=({x1}, {y1}) -> ({x2}, {y2})"
            )
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.text_prompt_list.addItem(item)

    def _commit_text_prompt_proposal(
        self,
        obj_id: int,
        proposal: TextPromptProposal,
        frame_size: Tuple[int, int],
        existing_output: SamFrameOutput,
    ) -> None:
        width, height = frame_size
        x1, y1, x2, y2 = proposal.box_xyxy_px
        frame_boxes = self.box_prompts_by_frame_obj.setdefault(self.current_frame_idx, {})
        frame_boxes[obj_id] = BoxPrompt(x1_px=x1, y1_px=y1, x2_px=x2, y2_px=y2)
        self._set_current_box_locked(self.current_frame_idx, obj_id, False)
        self.prompts_by_frame_obj.setdefault(self.current_frame_idx, {}).setdefault(obj_id, [])
        box_xywh_norm = (
            float(x1) / width,
            float(y1) / height,
            float(max(0, x2 - x1)) / width,
            float(max(0, y2 - y1)) / height,
        )
        proposal_output = SamFrameOutput(
            obj_ids=[obj_id],
            masks=[np.asarray(proposal.mask).astype(bool)],
            boxes_xywh_norm=[box_xywh_norm],
            scores=[float(proposal.score)],
            tracker_scores=[0.0],
        )
        self.outputs_by_frame[self.current_frame_idx] = self._merge_frame_outputs(existing_output, proposal_output)
        self._output_version_by_frame[self.current_frame_idx] = self._prefetch_prompt_version

    def remove_active_object(self) -> None:
        if self.active_object_id is None:
            return

        obj_id = self.active_object_id
        self.objects = [o for o in self.objects if o.obj_id != obj_id]
        self.hidden_obj_ids.discard(obj_id)
        if self.solo_object_id == obj_id:
            self.solo_object_id = None

        for frame_map in self.prompts_by_frame_obj.values():
            frame_map.pop(obj_id, None)
        for frame_map in self.box_prompts_by_frame_obj.values():
            frame_map.pop(obj_id, None)
        for frame_map in self.box_locked_by_frame_obj.values():
            frame_map.pop(obj_id, None)
        for frame_map in self.manual_propagation_overrides_by_frame_obj.values():
            frame_map.pop(obj_id, None)
        self.manual_propagation_overrides_by_frame_obj = {
            frame_idx: frame_map
            for frame_idx, frame_map in self.manual_propagation_overrides_by_frame_obj.items()
            if frame_map
        }

        for frame_idx, out in list(self.outputs_by_frame.items()):
            keep_idx = [i for i, oid in enumerate(out.obj_ids) if oid != obj_id]
            if len(keep_idx) == len(out.obj_ids):
                continue
            self.outputs_by_frame[frame_idx] = SamFrameOutput(
                obj_ids=[out.obj_ids[i] for i in keep_idx],
                masks=[out.masks[i] for i in keep_idx],
                boxes_xywh_norm=[out.boxes_xywh_norm[i] for i in keep_idx],
                scores=[out.scores[i] for i in keep_idx],
                tracker_scores=[out.tracker_scores[i] for i in keep_idx],
            )

        for i in range(self.object_list.count()):
            item = self.object_list.item(i)
            if int(item.data(Qt.UserRole)) == obj_id:
                self.object_list.takeItem(i)
                break
        self._object_row_widgets.pop(obj_id, None)

        self.active_object_id = None
        self._refresh_object_list_visuals()
        self.refresh_point_list()
        self._render_current_frame()

    def go_prev_frame(self) -> None:
        if not self.frame_paths:
            return
        self._set_current_frame_idx(max(0, self.current_frame_idx - 1))

    def go_next_frame(self) -> None:
        if not self.frame_paths:
            return
        self._set_current_frame_idx(min(len(self.frame_paths) - 1, self.current_frame_idx + 1))

    def go_next_frame_shortcut(self) -> None:
        if not self.frame_paths:
            return
        if self.auto_propagate_next and self.current_frame_idx < len(self.frame_paths) - 1:
            next_idx = min(len(self.frame_paths) - 1, self.current_frame_idx + 1)
            if self._has_valid_prefetch(next_idx):
                self._set_current_frame_idx(next_idx)
                self._schedule_prefetch_for_next_frame()
                return
            if self._auto_propagate_next_frame():
                self._set_current_frame_idx(next_idx)
                self._schedule_prefetch_for_next_frame()
                return
        self.go_next_frame()

    def _auto_propagate_next_frame(self) -> bool:
        enabled_obj_ids = self._get_enabled_propagation_obj_ids()
        if not enabled_obj_ids:
            self._set_status("No objects enabled for auto propagation; moving to next frame.", progress=0, total=1)
            return False

        next_idx = min(len(self.frame_paths) - 1, self.current_frame_idx + 1)
        if not self._is_tracker_propagation_mode():
            if not self._ensure_sam_initialized():
                return False
            copied_obj_ids = self._copy_boxes_forward_once(
                self.current_frame_idx,
                next_idx,
                enabled_obj_ids,
            )
            if not copied_obj_ids:
                self._set_status("No enabled objects have a box to copy forward.", progress=0, total=1)
                return False
            if not self._segment_frame_objects(
                frame_idx=next_idx,
                obj_ids=set(copied_obj_ids),
                show_no_prompts=False,
            ):
                return False
            self._set_status(f"Copied boxes to frame {next_idx + 1}.", progress=1, total=1)
            return True
        if (
            next_idx in self.outputs_by_frame
            and self._output_version_by_frame.get(next_idx, -1) == self._prefetch_prompt_version
        ):
            return True

        if not self._ensure_sam_initialized():
            return False
        if self._has_valid_prefetch(next_idx):
            return True

        prompt_payload = self._build_seed_prompts(
            seed_frame_idx=self.current_frame_idx,
            use_carryover_sampling=False,
            enabled_obj_ids=enabled_obj_ids,
        )
        if prompt_payload is None:
            return False

        self._output_version_by_frame.pop(next_idx, None)
        if not self._prefetch_busy:
            task_id = self._enqueue_sam_task(
                "prefetch",
                {
                    "seed_frame_idx": self.current_frame_idx,
                    "n_frames": 2,
                    "frame_paths": self.frame_paths,
                    "prompt_payload": prompt_payload,
                },
            )
            self._sam_task_contexts[task_id] = {
                "kind": "prefetch",
                    "target_frame_idx": next_idx,
                    "seed_frame_idx": self.current_frame_idx,
                    "version": self._prefetch_prompt_version,
                }
            self._prefetch_busy = True
            self._prefetch_seed_frame_idx = self.current_frame_idx
            self._prefetch_target_frame_idx = next_idx
            self._prefetch_active_version = self._prefetch_prompt_version
            self._set_status("Prefetching next frame...", progress=0, total=1)
        if self._prefetch_busy:
            task_id = f"prefetch-wait:{self.current_frame_idx}->{next_idx}:{self._prefetch_prompt_version}"
            self._sam_task_contexts[task_id] = {"kind": "prefetch_wait"}
            self._sam_waiting[task_id] = QEventLoop()
            self._sam_waiting[task_id].exec()
            self._sam_waiting.pop(task_id, None)
            self._sam_task_contexts.pop(task_id, None)
        return self._has_valid_prefetch(next_idx)

    def _set_current_frame_idx(self, frame_idx: int) -> None:
        if not self.frame_paths:
            return
        self.current_frame_idx = max(0, min(len(self.frame_paths) - 1, frame_idx))
        self._clear_text_prompt_proposals_if_needed(self.current_frame_idx)
        self._sync_frame_navigation_controls()
        self._render_current_frame()

    def _sync_frame_navigation_controls(self) -> None:
        max_frame = max(1, len(self.frame_paths))
        current_value = min(max_frame, self.current_frame_idx + 1)

        self.frame_slider.blockSignals(True)
        self.frame_slider.setMinimum(1)
        self.frame_slider.setMaximum(max_frame)
        self.frame_slider.setValue(current_value)
        self.frame_slider.blockSignals(False)

        self.frame_jump_spin.blockSignals(True)
        self.frame_jump_spin.setMinimum(1)
        self.frame_jump_spin.setMaximum(max_frame)
        self.frame_jump_spin.setValue(current_value)
        self.frame_jump_spin.blockSignals(False)

        if hasattr(self, "target_frame_spin"):
            self.target_frame_spin.blockSignals(True)
            self.target_frame_spin.setMinimum(1)
            self.target_frame_spin.setMaximum(max_frame)
            if self.target_frame_spin.value() < 1 or self.target_frame_spin.value() > max_frame:
                self.target_frame_spin.setValue(current_value)
            self.target_frame_spin.blockSignals(False)
            self._update_target_chunk_label()

    def _on_frame_slider_changed(self, value: int) -> None:
        if not self.frame_paths:
            return
        self._set_current_frame_idx(value - 1)

    def _on_frame_jump_changed(self, value: int) -> None:
        if not self.frame_paths:
            return
        target_idx = value - 1
        if self.auto_propagate_next and target_idx == self.current_frame_idx + 1:
            self.go_next_frame_shortcut()
            return
        self._set_current_frame_idx(target_idx)

    def fit_current_frame_to_view(self) -> None:
        self._zoom_multiplier = 1.0
        self._pan_offset_ui = (0.0, 0.0)
        self._pan_drag_last_ui_xy = None
        self._render_current_frame()

    def on_pan_press(self, ui_x: float, ui_y: float) -> None:
        if not self.frame_paths:
            return
        self._pan_drag_last_ui_xy = (ui_x, ui_y)

    def on_pan_drag(self, ui_x: float, ui_y: float) -> None:
        if self._pan_drag_last_ui_xy is None or not self.frame_paths:
            return
        last_x, last_y = self._pan_drag_last_ui_xy
        dx = ui_x - last_x
        dy = ui_y - last_y
        pan_x, pan_y = self._pan_offset_ui
        self._pan_offset_ui = (pan_x + dx, pan_y + dy)
        self._clamp_pan_offset()
        self._pan_drag_last_ui_xy = (ui_x, ui_y)
        self._render_current_frame()

    def on_pan_release(self) -> None:
        self._pan_drag_last_ui_xy = None

    def on_image_wheel(self, ui_x: float, ui_y: float, delta_y: int) -> None:
        if not self.frame_paths or delta_y == 0:
            return
        anchor_xy = self._map_ui_to_image_xy_clamped(ui_x, ui_y)
        if anchor_xy is None:
            frame_size = self._get_current_frame_size()
            if frame_size is None:
                return
            frame_w, frame_h = frame_size
            anchor_xy = (frame_w // 2, frame_h // 2)
            ui_x = self.image_label.width() / 2.0
            ui_y = self.image_label.height() / 2.0

        zoom_step = 1.15 if delta_y > 0 else (1.0 / 1.15)
        new_zoom = float(min(12.0, max(1.0, self._zoom_multiplier * zoom_step)))
        if abs(new_zoom - self._zoom_multiplier) < 1e-6:
            return
        self._zoom_multiplier = new_zoom
        self._set_pan_for_anchor(anchor_xy, ui_x, ui_y)
        self._render_current_frame()

    def on_image_press(self, ui_x: float, ui_y: float) -> None:
        if not self.frame_paths:
            return
        if self.active_object_id is None:
            QMessageBox.information(self, "No active object", "Select an object before annotating.")
            return

        mapped = self._map_ui_to_image_xy(ui_x, ui_y)
        if mapped is None:
            return
        x_px, y_px = mapped

        if self._is_box_mode():
            if not self.show_boxes:
                self.statusBar().showMessage("Enable Show Boxes to edit annotation boxes.")
                return
            preview_box, is_edit = self._begin_box_interaction(x_px, y_px)
            if preview_box is None:
                self._render_current_frame()
                return
            self._box_edit_active = is_edit
            self._update_box_rubber_band_for_box(preview_box)
            self._box_rubber_band.show()
            return

        points = self._get_active_points()
        if len(points) >= MAX_POINTS_PER_OBJECT:
            QMessageBox.warning(self, "Point limit", f"Maximum {MAX_POINTS_PER_OBJECT} points per object per frame.")
            return

        self._stash_undo_state()
        points.append(PointPrompt(x_px=x_px, y_px=y_px, is_positive=self.current_prompt_positive))
        self.refresh_point_list()
        if self._active_prompt_rows:
            self.point_list.setCurrentRow(len(self._active_prompt_rows) - 1)

        if self.segment_mode:
            self.segment_current_frame()
            self._note_prompt_change_and_prefetch()
        else:
            self._note_prompt_change_and_prefetch()
            self._render_current_frame()

    def on_image_double_click(self, ui_x: float, ui_y: float) -> None:
        if not self.frame_paths or not self.show_boxes:
            return
        mapped = self._map_ui_to_image_xy(ui_x, ui_y)
        if mapped is None:
            return
        hit = self._find_box_at_point(*mapped)
        if hit is None:
            return
        obj_id, _box = hit
        self._set_active_object_by_id(obj_id)
        self._selected_box_obj_id = obj_id
        self.prompt_mode_combo.setCurrentIndex(2)
        self.refresh_point_list()
        if self._active_prompt_rows:
            for row_idx, (prompt_type, _idx) in enumerate(self._active_prompt_rows):
                if prompt_type == "box":
                    self.point_list.setCurrentRow(row_idx)
                    break
        self._render_current_frame()

    def on_image_drag(self, ui_x: float, ui_y: float) -> None:
        if self._box_draw_start_xy is None:
            return
        mapped = self._map_ui_to_image_xy_clamped(ui_x, ui_y)
        if mapped is None:
            return
        preview_box = self._build_box_from_drag(mapped)
        if preview_box is None:
            return
        self._update_box_rubber_band_for_box(preview_box)

    def on_image_release(self, ui_x: float, ui_y: float) -> None:
        if self._box_draw_start_xy is None:
            return
        self._box_rubber_band.hide()
        mapped = self._map_ui_to_image_xy_clamped(ui_x, ui_y)
        if mapped is None:
            mapped = self._box_draw_start_xy
        box = self._build_box_from_drag(mapped)
        self._box_draw_start_xy = None
        self._box_draw_start_ui_xy = None
        self._box_edit_active = False
        self._box_drag_mode = None
        self._box_drag_reference_box = None
        self._box_drag_offset_xy = (0, 0)

        if box is None:
            self._render_current_frame()
            return

        self._stash_undo_state()
        frame_boxes = self.box_prompts_by_frame_obj.setdefault(self.current_frame_idx, {})
        frame_boxes[self.active_object_id] = box
        self._selected_box_obj_id = self.active_object_id
        self.refresh_point_list()
        if self._active_prompt_rows:
            self.point_list.setCurrentRow(0)
        is_locked = self._is_current_box_locked(self.current_frame_idx, self.active_object_id)
        self._sync_current_box_lock_check()
        if not is_locked:
            self._note_prompt_change_and_prefetch()
            self._refine_current_frame_boxes({self.active_object_id}, show_no_prompts=False)
            return
        self._note_prompt_change_and_prefetch()
        self._render_current_frame()

    def _get_active_points(self) -> List[PointPrompt]:
        frame_map = self.prompts_by_frame_obj.setdefault(self.current_frame_idx, {})
        return frame_map.setdefault(self.active_object_id, [])

    def refresh_point_list(self) -> None:
        current_row = self.point_list.currentRow()
        self.point_list.blockSignals(True)
        self.point_list.clear()
        self._active_prompt_rows = []
        if self.active_object_id is None:
            self.point_list.blockSignals(False)
            return
        frame_boxes = self.box_prompts_by_frame_obj.get(self.current_frame_idx, {})
        box = frame_boxes.get(self.active_object_id)
        if box:
            lock_suffix = " [Locked]" if self._is_current_box_locked(self.current_frame_idx, self.active_object_id) else ""
            self.point_list.addItem(f"Box{lock_suffix} ({box.x1_px}, {box.y1_px}) -> ({box.x2_px}, {box.y2_px})")
            self._active_prompt_rows.append(("box", 0))
        frame_map = self.prompts_by_frame_obj.get(self.current_frame_idx, {})
        points = frame_map.get(self.active_object_id, [])
        for i, p in enumerate(points, start=1):
            sign = "+" if p.is_positive else "-"
            self.point_list.addItem(f"{i}. {sign} ({p.x_px}, {p.y_px})")
            self._active_prompt_rows.append(("point", i - 1))
        if self._active_prompt_rows and 0 <= current_row < len(self._active_prompt_rows):
            self.point_list.setCurrentRow(current_row)
        self.point_list.blockSignals(False)

    def remove_selected_prompt(self) -> None:
        if self.active_object_id is None:
            return
        row = self.point_list.currentRow()
        if row < 0:
            return
        if row >= len(self._active_prompt_rows):
            return
        self._stash_undo_state()
        prompt_type, idx = self._active_prompt_rows[row]
        frame_map = self.prompts_by_frame_obj.get(self.current_frame_idx, {})
        if prompt_type == "point":
            points = frame_map.get(self.active_object_id, [])
            if idx >= len(points):
                return
            del points[idx]
        elif prompt_type == "box":
            frame_boxes = self.box_prompts_by_frame_obj.get(self.current_frame_idx, {})
            frame_boxes.pop(self.active_object_id, None)
            self._set_current_box_locked(self.current_frame_idx, self.active_object_id, False)
            if self._selected_box_obj_id == self.active_object_id:
                self._selected_box_obj_id = None
            self._remove_object_output_from_frame(self.current_frame_idx, self.active_object_id)
            self.refresh_point_list()
            self._render_current_frame()
            self._note_prompt_change_and_prefetch()
            return
        else:
            return
        self.refresh_point_list()
        self._refresh_after_prompt_edit()

    def clear_active_object_prompts(self) -> None:
        if self.active_object_id is None:
            return
        self._stash_undo_state()
        frame_map = self.prompts_by_frame_obj.get(self.current_frame_idx, {})
        frame_map[self.active_object_id] = []
        frame_boxes = self.box_prompts_by_frame_obj.get(self.current_frame_idx, {})
        removed_box = self.active_object_id in frame_boxes
        frame_boxes.pop(self.active_object_id, None)
        self._set_current_box_locked(self.current_frame_idx, self.active_object_id, False)
        if self._selected_box_obj_id == self.active_object_id:
            self._selected_box_obj_id = None
        self.refresh_point_list()
        if removed_box:
            self._remove_object_output_from_frame(self.current_frame_idx, self.active_object_id)
            self._render_current_frame()
            self._note_prompt_change_and_prefetch()
            return
        self._refresh_after_prompt_edit()

    def _ensure_sam_initialized(self) -> bool:
        """Ensure the SAM3 worker is available (no session started here)."""
        if self._sam_ready:
            return True
        return self._initialize_sam_worker(show_errors=True)

    def _initialize_sam_worker(self, show_errors: bool) -> bool:
        if self._sam_worker is not None and self._sam_ready:
            return True
        if self._sam_worker is not None and not self._sam_ready:
            return False
        worker = SamWorker(
            self._checkpoint_path,
            self._bpe_path,
            self.recondition_every_nth_frame,
            self.recondition_high_conf_thresh,
            self.recondition_high_iou_thresh,
        )
        thread = QThread()
        worker.moveToThread(thread)
        worker.initialized.connect(self._on_sam_worker_initialized)
        worker.segment_done.connect(self._on_sam_segment_done)
        worker.text_prompt_done.connect(self._on_sam_text_prompt_done)
        worker.propagate_frame.connect(self._on_sam_propagate_frame)
        worker.propagate_done.connect(self._on_sam_propagate_done)
        worker.propagate_stopped.connect(self._on_sam_propagate_stopped)
        worker.task_failed.connect(self._on_sam_task_failed)
        worker.task_finished.connect(self._on_sam_task_finished)
        self.task_requested.connect(worker.enqueue_task, Qt.QueuedConnection)
        thread.started.connect(worker.initialize)
        thread.start()

        self._sam_worker = worker
        self._sam_thread = thread

        loop = QEventLoop()
        self._sam_waiting["__init__"] = loop
        loop.exec()
        self._sam_waiting.pop("__init__", None)

        if not self._sam_ready:
            if show_errors and hasattr(self, "_sam_init_error") and self._sam_init_error:
                QMessageBox.critical(self, "SAM3 init failed", self._sam_init_error)
            return False
        return True

    def segment_current_frame(self) -> None:
        frame_prompts = self.prompts_by_frame_obj.get(self.current_frame_idx, {})
        frame_boxes = self.box_prompts_by_frame_obj.get(self.current_frame_idx, {})
        obj_ids = {
            obj.obj_id
            for obj in self.objects
            if frame_prompts.get(obj.obj_id) or frame_boxes.get(obj.obj_id)
        }
        if self._prefetch_busy:
            self._cancel_prefetch(restart=True)
        self._refine_current_frame_boxes(obj_ids, show_no_prompts=True)

    def _refine_current_frame_boxes(
        self,
        obj_ids: set[int],
        show_no_prompts: bool,
    ) -> bool:
        return self._segment_frame_objects(
            frame_idx=self.current_frame_idx,
            obj_ids=obj_ids,
            show_no_prompts=show_no_prompts,
        )

    def _segment_frame_objects(
        self,
        *,
        frame_idx: int,
        obj_ids: set[int],
        show_no_prompts: bool,
    ) -> bool:
        if not obj_ids:
            if show_no_prompts:
                QMessageBox.information(self, "No prompts", "Add point or box prompts for at least one object on this frame.")
            return False
        if not self._ensure_sam_initialized():
            return False
        payload = self._build_segment_payload(frame_idx, obj_ids, show_no_prompts)
        if payload is None:
            return False
        task_id = self._enqueue_sam_task(
            "segment",
            {
                "frame_idx": frame_idx,
                "frame_path": self.frame_paths[frame_idx],
                "payload": payload,
            },
        )
        result = self._wait_for_sam_task(task_id)
        return bool(result)

    def propagate_next_frame(self) -> None:
        if not self.frame_paths:
            return
        if self._propagation_busy:
            self._set_status("Propagation already in progress.", progress=0, total=1)
            return
        if self._prefetch_busy:
            self._cancel_prefetch(restart=False)
        enabled_obj_ids = self._get_enabled_propagation_obj_ids()
        if not enabled_obj_ids:
            QMessageBox.information(self, "No objects enabled", "Check at least one object to propagate.")
            return
        if self._pending_propagation is None and self._is_tracker_propagation_mode():
            if not self._ensure_sam_initialized():
                return
            unlocked_obj_ids = {
                obj_id for obj_id in enabled_obj_ids
                if self.box_prompts_by_frame_obj.get(self.current_frame_idx, {}).get(obj_id) is not None
                and not self._is_current_box_locked(self.current_frame_idx, obj_id)
            }
            if unlocked_obj_ids:
                self._refine_current_frame_boxes(unlocked_obj_ids, show_no_prompts=False)
        if self._pending_propagation is None:
            if self.current_frame_idx >= len(self.frame_paths) - 1:
                QMessageBox.information(self, "End of video", "Already at last frame.")
                return
            target_frame_idx: Optional[int] = None
            total_chunks = self.chunks_spin.value()
            if self.use_target_frame_check.isChecked():
                target_frame_idx = self.target_frame_spin.value() - 1
                if target_frame_idx <= self.current_frame_idx:
                    QMessageBox.information(self, "Invalid target", "Target frame must be after the current frame.")
                    return
                total_chunks = self._compute_target_chunk_count(
                    self.current_frame_idx,
                    target_frame_idx,
                    self.n_propagate_spin.value(),
                ) or 0
                if total_chunks <= 0:
                    QMessageBox.information(self, "Invalid target", "Target frame is too close for the current N frames setting.")
                    return
            self._pending_propagation = PendingPropagationState(
                remaining_chunks=total_chunks,
                next_seed_frame_idx=self.current_frame_idx,
                n_frames=self.n_propagate_spin.value(),
                total_chunks=total_chunks,
                target_frame_idx=target_frame_idx,
            )
            self._propagation_stop_requested = False
            target_note = ""
            if target_frame_idx is not None:
                target_note = f" (target frame {target_frame_idx + 1})"
            self._set_status(
                f"Propagation starting: 0/{self._pending_propagation.total_chunks} chunks.{target_note}",
                progress=0,
                total=self._pending_propagation.total_chunks,
            )
            self._propagation_enabled_obj_ids = enabled_obj_ids
            self._propagation_view_frame_idx = self.current_frame_idx
            self._one_session_chunk_prompt_version = self._prefetch_prompt_version if self.use_one_session_chunked_propagation else None
        state = self._pending_propagation
        if state is None:
            return
        if state.next_seed_frame_idx >= len(self.frame_paths) - 1:
            self._clear_pending_propagation_state()
            QMessageBox.information(self, "End of video", "No more frames left to propagate.")
            return

        self._start_propagation_chunk_async(
            enabled_obj_ids=self._propagation_enabled_obj_ids or enabled_obj_ids,
            state=state,
        )

    def _build_seed_prompts(
        self,
        seed_frame_idx: int,
        use_carryover_sampling: bool,
        enabled_obj_ids: set[int],
    ) -> Optional[Dict[int, Dict[str, object]]]:
        del use_carryover_sampling
        frame_boxes = self.box_prompts_by_frame_obj.get(seed_frame_idx, {})
        frame_prompts = self.prompts_by_frame_obj.get(seed_frame_idx, {})

        frame_size = self._get_frame_size(seed_frame_idx)
        if frame_size is None:
            QMessageBox.critical(self, "Propagation setup failed", "Failed to load prompt source frame.")
            return None
        w, h = frame_size

        boxes_to_apply: Dict[int, BoxPrompt] = {}
        masks_to_apply: Dict[int, np.ndarray] = {}
        points_to_apply: Dict[int, List[PointPrompt]] = {}
        missing_obj_ids: List[int] = []
        for obj_id in enabled_obj_ids:
            canonical_box = frame_boxes.get(obj_id)
            if canonical_box is not None:
                boxes_to_apply[obj_id] = canonical_box
            manual_points = list(frame_prompts.get(obj_id, [])) if self.use_point_prompts_for_propagation else []
            if manual_points:
                points_to_apply[obj_id] = manual_points
        seed_output = self.outputs_by_frame.get(seed_frame_idx)
        if seed_output is not None:
            for i, obj_id in enumerate(seed_output.obj_ids):
                if obj_id in enabled_obj_ids and i < len(seed_output.masks):
                    masks_to_apply[obj_id] = np.asarray(seed_output.masks[i]).astype(np.float32)

        sampled_boxes = self._sample_boxes_from_seed_masks(seed_frame_idx, h, w)
        for obj_id in enabled_obj_ids:
            if obj_id not in boxes_to_apply and obj_id in sampled_boxes:
                boxes_to_apply[obj_id] = sampled_boxes[obj_id]

        for obj_id in enabled_obj_ids:
            has_box = obj_id in boxes_to_apply
            has_mask = obj_id in masks_to_apply
            has_points = bool(points_to_apply.get(obj_id))
            if not has_box and not has_mask and not has_points:
                missing_obj_ids.append(obj_id)

        if missing_obj_ids:
            missing_names = [
                self._find_object(obj_id).name if self._find_object(obj_id) else str(obj_id)
                for obj_id in missing_obj_ids
            ]
            QMessageBox.warning(
                self,
                "Propagation setup failed",
                "These enabled objects do not have any usable seed-frame prompts or masks: "
                + ", ".join(missing_names),
            )
            return None

        obj_ids = set(enabled_obj_ids)
        if not obj_ids:
            QMessageBox.information(
                self,
                "No prompts",
                "No valid tracker seed inputs are available for this chunk seed.",
            )
            return None

        payload: Dict[int, Dict[str, object]] = {}
        for obj_id in obj_ids:
            box = boxes_to_apply.get(obj_id)
            points_rel: List[List[float]] = []
            labels: List[int] = []
            if box is not None:
                points_rel.append([box.x1_px / w, box.y1_px / h])
                points_rel.append([box.x2_px / w, box.y2_px / h])
                labels.append(2)
                labels.append(3)
            for point in points_to_apply.get(obj_id, []):
                points_rel.append([point.x_px / w, point.y_px / h])
                labels.append(1 if point.is_positive else 0)
            payload[obj_id] = {
                "points_rel": points_rel,
                "labels": labels,
                "mask_input": masks_to_apply.get(obj_id),
            }

        if not payload:
            QMessageBox.information(
                self,
                "No prompts",
                "No valid prompts available for this chunk seed.",
            )
            return None

        return payload

    def _build_segment_payload(
        self,
        frame_idx: int,
        obj_ids: set[int],
        show_no_prompts: bool,
    ) -> Optional[Dict[int, Dict[str, object]]]:
        frame_size = self._get_frame_size(frame_idx)
        if frame_size is None:
            QMessageBox.critical(self, "Segmentation failed", "Failed to load current frame.")
            return None
        w, h = frame_size
        frame_prompts = self.prompts_by_frame_obj.get(frame_idx, {})
        frame_boxes = self.box_prompts_by_frame_obj.get(frame_idx, {})
        payload: Dict[int, Dict[str, object]] = {}
        for obj_id in sorted(obj_ids):
            points = frame_prompts.get(obj_id, [])
            points_rel = [[p.x_px / w, p.y_px / h] for p in points]
            labels = [1 if p.is_positive else 0 for p in points]
            box = frame_boxes.get(obj_id)
            if box is not None:
                points_rel.append([box.x1_px / w, box.y1_px / h])
                points_rel.append([box.x2_px / w, box.y2_px / h])
                labels.append(2)
                labels.append(3)
            if not points_rel:
                continue
            payload[obj_id] = {"points_rel": points_rel, "labels": labels}
        if not payload and show_no_prompts:
            QMessageBox.information(self, "No prompts", "Add point or box prompts for at least one object on this frame.")
        return payload or None

    def _copy_boxes_forward_once(
        self,
        src_frame_idx: int,
        dst_frame_idx: int,
        enabled_obj_ids: set[int],
    ) -> List[int]:
        src_boxes = self.box_prompts_by_frame_obj.get(src_frame_idx, {})
        copied_obj_ids: List[int] = []
        for obj_id in sorted(enabled_obj_ids):
            box = src_boxes.get(obj_id)
            if box is None:
                continue
            dst_boxes = self.box_prompts_by_frame_obj.setdefault(dst_frame_idx, {})
            dst_boxes[obj_id] = BoxPrompt(
                x1_px=box.x1_px,
                y1_px=box.y1_px,
                x2_px=box.x2_px,
                y2_px=box.y2_px,
            )
            self._set_current_box_locked(
                dst_frame_idx,
                obj_id,
                self._is_current_box_locked(src_frame_idx, obj_id),
            )
            self._remove_object_output_from_frame(dst_frame_idx, obj_id)
            copied_obj_ids.append(obj_id)
        if copied_obj_ids:
            self._carry_prompts_forward(src_frame_idx, dst_frame_idx, copied_obj_ids)
        return copied_obj_ids

    def _run_copy_box_propagation_chunk(
        self,
        *,
        enabled_obj_ids: set[int],
        state: PendingPropagationState,
    ) -> None:
        chunk_idx = state.completed_chunks + 1
        seed_frame_idx = state.next_seed_frame_idx
        chunk_last_frame_idx = min(seed_frame_idx + state.n_frames - 1, len(self.frame_paths) - 1)
        if state.target_frame_idx is not None:
            chunk_last_frame_idx = min(chunk_last_frame_idx, state.target_frame_idx)
        if chunk_last_frame_idx <= seed_frame_idx:
            self._clear_pending_propagation_state()
            QMessageBox.information(self, "End of video", "No more frames left to propagate.")
            return

        if not self._ensure_sam_initialized():
            self._clear_pending_propagation_state()
            return

        copied_any = False
        for dst_frame_idx in range(seed_frame_idx + 1, chunk_last_frame_idx + 1):
            copied_obj_ids = self._copy_boxes_forward_once(
                dst_frame_idx - 1,
                dst_frame_idx,
                enabled_obj_ids,
            )
            if not copied_obj_ids:
                continue
            copied_any = True
            if not self._segment_frame_objects(
                frame_idx=dst_frame_idx,
                obj_ids=set(copied_obj_ids),
                show_no_prompts=False,
            ):
                self._clear_pending_propagation_state()
                return

        if not copied_any:
            self._clear_pending_propagation_state()
            QMessageBox.information(
                self,
                "Propagation setup failed",
                "No enabled objects have a box on the current seed frame to copy forward.",
            )
            return

        self._set_current_frame_idx(chunk_last_frame_idx)
        state.remaining_chunks -= 1
        state.completed_chunks += 1
        state.next_seed_frame_idx = chunk_last_frame_idx
        if state.target_frame_idx is not None:
            recomputed = self._compute_target_chunk_count(
                state.next_seed_frame_idx,
                state.target_frame_idx,
                state.n_frames,
            )
            state.remaining_chunks = 0 if recomputed is None else recomputed
            state.total_chunks = state.completed_chunks + state.remaining_chunks

        self._set_status(
            f"Chunk {state.completed_chunks}/{state.total_chunks} complete.",
            progress=state.completed_chunks,
            total=state.total_chunks,
        )
        if state.remaining_chunks <= 0:
            self._set_status(
                f"Propagation complete: {state.total_chunks}/{state.total_chunks} chunks.",
                progress=state.total_chunks,
                total=state.total_chunks,
            )
            self._clear_pending_propagation_state()
            return

        self._run_copy_box_propagation_chunk(enabled_obj_ids=enabled_obj_ids, state=state)

    def _start_propagation_chunk_async(
        self,
        *,
        enabled_obj_ids: set[int],
        state: PendingPropagationState,
    ) -> None:
        if not self._is_tracker_propagation_mode():
            self._run_copy_box_propagation_chunk(enabled_obj_ids=enabled_obj_ids, state=state)
            return
        chunk_idx = state.completed_chunks + 1
        seed_frame_idx = state.next_seed_frame_idx
        use_carryover_sampling = chunk_idx > 1
        prompt_payload = self._build_seed_prompts(
            seed_frame_idx=seed_frame_idx,
            use_carryover_sampling=use_carryover_sampling,
            enabled_obj_ids=enabled_obj_ids,
        )
        if prompt_payload is None:
            self._clear_pending_propagation_state()
            return

        self._propagation_busy = True
        self._propagation_active_chunk_idx = chunk_idx
        self._propagation_active_seed_frame_idx = seed_frame_idx
        self._propagation_seen_obj_ids.clear()
        self._propagation_lost_obj_ids.clear()
        self._propagation_loss_notified = False
        self._set_propagation_ui_enabled(False)
        self._set_status(
            f"Chunk {chunk_idx}/{state.total_chunks} running...",
            progress=state.completed_chunks,
            total=state.total_chunks,
        )
        use_one_session = self.use_one_session_chunked_propagation
        rebuild_one_session = bool(use_one_session and state.completed_chunks == 0)
        if use_one_session and self._one_session_chunk_prompt_version != self._prefetch_prompt_version:
            rebuild_one_session = True

        task_id = self._enqueue_sam_task(
            "propagate",
            {
                "seed_frame_idx": seed_frame_idx,
                "n_frames": state.n_frames,
                "frame_paths": self.frame_paths,
                "prompt_payload": prompt_payload,
                "use_one_session": use_one_session,
                "rebuild_one_session": rebuild_one_session,
            },
        )
        self._sam_task_contexts[task_id] = {
            "kind": "manual",
            "seed_frame_idx": seed_frame_idx,
            "last_emitted_frame_idx": seed_frame_idx - 1,
            "use_one_session": use_one_session,
        }
        self._propagation_task_id = task_id

    def _handle_propagation_frame(self, abs_frame_idx: int, output: SamFrameOutput, session_idx: int, total_frames: int) -> None:
        existing_output = self.outputs_by_frame.get(
            abs_frame_idx,
            SamFrameOutput(obj_ids=[], masks=[], boxes_xywh_norm=[], scores=[], tracker_scores=[]),
        )
        merged_output = self._merge_frame_outputs(existing_output, output)
        self.outputs_by_frame[abs_frame_idx] = merged_output
        self._output_version_by_frame[abs_frame_idx] = self._prefetch_prompt_version
        seed_frame_idx = getattr(self, "_propagation_active_seed_frame_idx", None)
        if seed_frame_idx is not None and abs_frame_idx != seed_frame_idx:
            self._sync_canonical_boxes_from_output(abs_frame_idx, output, preserve_locked=False)
            self._carry_prompts_forward(abs_frame_idx - 1, abs_frame_idx, output.obj_ids)
        if self._propagation_busy:
            self._set_current_frame_idx(abs_frame_idx)
        elif abs_frame_idx == self.current_frame_idx:
            self._render_current_frame()
        state = self._pending_propagation
        if state is not None:
            self._set_status(
                f"Chunk {state.completed_chunks + 1}/{state.total_chunks}: {session_idx + 1}/{total_frames} frames",
                progress=state.completed_chunks,
                total=state.total_chunks,
            )

    def _on_sam_segment_done(self, task_id: str, frame_idx: int, composite: SamFrameOutput) -> None:
        existing_output = self.outputs_by_frame.get(
            frame_idx,
            SamFrameOutput(obj_ids=[], masks=[], boxes_xywh_norm=[], scores=[], tracker_scores=[]),
        )
        merged = self._merge_frame_outputs(existing_output, composite)
        if composite.obj_ids:
            self.outputs_by_frame[frame_idx] = merged
            self._output_version_by_frame[frame_idx] = self._prefetch_prompt_version
            if not self._is_current_box_locked(frame_idx, None):
                self._sync_canonical_boxes_from_output(frame_idx, composite, preserve_locked=False)
            self.segment_mode = True
            self.mode_label.setText("Mode: Segment" if self._use_point_prompt_mode() else "Mode: Box Annotation")
        self._sync_current_box_lock_check()
        self._render_current_frame()
        self._sam_task_results[task_id] = True
        loop = self._sam_waiting.get(task_id)
        if loop is not None:
            loop.quit()
        self._sam_task_contexts.pop(task_id, None)
        self._schedule_prefetch_for_next_frame()

    def _on_sam_text_prompt_done(
        self,
        task_id: str,
        frame_idx: int,
        composite: SamFrameOutput,
        text_prompt: str,
    ) -> None:
        frame_size = self._get_frame_size(frame_idx)
        proposals: List[TextPromptProposal] = []
        if frame_size is not None:
            proposals = build_text_prompt_proposals(
                masks=composite.masks,
                boxes_xywh_norm=composite.boxes_xywh_norm,
                scores=composite.scores,
                frame_size=frame_size,
            )
        self._text_prompt_proposals_frame_idx = frame_idx
        self._text_prompt_last_prompt = text_prompt
        self._text_prompt_proposals = proposals
        self._refresh_text_prompt_list()
        self._render_current_frame()
        self._sam_task_results[task_id] = proposals
        loop = self._sam_waiting.get(task_id)
        if loop is not None:
            loop.quit()
        self._sam_task_contexts.pop(task_id, None)
        if proposals:
            self._set_status(f"Generated {len(proposals)} text prompt proposal(s).", progress=1, total=1)
        else:
            QMessageBox.information(self, "No proposals", f"No objects found for '{text_prompt}' on this frame.")
            self._set_status("No text prompt proposals generated.", progress=1, total=1)

    def _on_sam_propagate_frame(self, task_id: str, abs_frame_idx: int, output: SamFrameOutput, session_idx: int, total_frames: int) -> None:
        context = self._sam_task_contexts.get(task_id, {})
        kind = context.get("kind")
        seed_frame_idx = context.get("seed_frame_idx")
        if kind == "prefetch":
            if self._prefetch_cancel_requested:
                return
            version = context.get("version")
            target_frame_idx = context.get("target_frame_idx")
            if version != self._prefetch_prompt_version or abs_frame_idx != target_frame_idx:
                return
            existing_output = self.outputs_by_frame.get(
                abs_frame_idx,
                SamFrameOutput(obj_ids=[], masks=[], boxes_xywh_norm=[], scores=[], tracker_scores=[]),
            )
            merged_output = self._merge_frame_outputs(existing_output, output)
            self.outputs_by_frame[abs_frame_idx] = merged_output
            self._output_version_by_frame[abs_frame_idx] = self._prefetch_prompt_version
            self._sync_canonical_boxes_from_output(abs_frame_idx, output, preserve_locked=False)
            self._carry_prompts_forward(abs_frame_idx - 1, abs_frame_idx, output.obj_ids)
            self._prefetch_cached_frame_idx = abs_frame_idx
            self._prefetch_cached_seed_idx = seed_frame_idx
            self._prefetch_cached_version = version
            wait_key = f"prefetch-wait:{seed_frame_idx}->{abs_frame_idx}:{version}"
            loop = self._sam_waiting.get(wait_key)
            if loop is not None:
                loop.quit()
            if abs_frame_idx == self.current_frame_idx:
                self._render_current_frame()
            return

        if seed_frame_idx is not None and abs_frame_idx == seed_frame_idx:
            state = self._pending_propagation
            if kind == "manual" and state is not None:
                self._set_status(
                    f"Chunk {state.completed_chunks + 1}/{state.total_chunks}: {session_idx + 1}/{total_frames} frames",
                    progress=state.completed_chunks,
                    total=state.total_chunks,
                )
            return

        existing_output = self.outputs_by_frame.get(
            abs_frame_idx,
            SamFrameOutput(obj_ids=[], masks=[], boxes_xywh_norm=[], scores=[], tracker_scores=[]),
        )
        merged_output = self._merge_frame_outputs(existing_output, output)
        self.outputs_by_frame[abs_frame_idx] = merged_output
        self._output_version_by_frame[abs_frame_idx] = self._prefetch_prompt_version
        if seed_frame_idx is not None and abs_frame_idx != seed_frame_idx:
            self._sync_canonical_boxes_from_output(abs_frame_idx, output, preserve_locked=False)
            self._carry_prompts_forward(abs_frame_idx - 1, abs_frame_idx, output.obj_ids)
        if self._propagation_busy:
            self._set_current_frame_idx(abs_frame_idx)
        elif abs_frame_idx == self.current_frame_idx:
            self._render_current_frame()
        state = self._pending_propagation
        if kind == "manual" and state is not None:
            context["last_emitted_frame_idx"] = abs_frame_idx
            self._set_status(
                f"Chunk {state.completed_chunks + 1}/{state.total_chunks}: {session_idx + 1}/{total_frames} frames",
                progress=state.completed_chunks,
                total=state.total_chunks,
            )

    def _on_sam_propagate_done(self, task_id: str, last_masked_frame_idx: int, chunk_last_frame_idx: int) -> None:
        context = self._sam_task_contexts.get(task_id, {})
        kind = context.get("kind")
        if kind == "prefetch":
            self._prefetch_busy = False
            self._prefetch_cancel_requested = False
            if self._prefetch_pending_restart:
                self._prefetch_pending_restart = False
                self._schedule_prefetch_for_next_frame()
            self._sam_task_contexts.pop(task_id, None)
            return
        if kind == "auto":
            self._sam_task_results[task_id] = (last_masked_frame_idx, chunk_last_frame_idx)
            loop = self._sam_waiting.get(task_id)
            if loop is not None:
                loop.quit()
            self._sam_task_contexts.pop(task_id, None)
            return

        state = self._pending_propagation
        if state is None:
            return
        chunk_seed_frame_idx = context.get("seed_frame_idx", state.next_seed_frame_idx)
        state.remaining_chunks -= 1
        state.completed_chunks += 1
        state.next_seed_frame_idx = last_masked_frame_idx
        self._propagation_seen_obj_ids.clear()
        self._propagation_lost_obj_ids.clear()
        self._propagation_loss_notified = False
        if state.target_frame_idx is not None:
            recomputed = self._compute_target_chunk_count(
                state.next_seed_frame_idx,
                state.target_frame_idx,
                state.n_frames,
            )
            if recomputed is None:
                state.remaining_chunks = 0
            else:
                state.remaining_chunks = recomputed
            state.total_chunks = state.completed_chunks + state.remaining_chunks
        self._set_status(
            f"Chunk {state.completed_chunks}/{state.total_chunks} complete.",
            progress=state.completed_chunks,
            total=state.total_chunks,
        )

        if self._propagation_stop_requested:
            self._propagation_stop_requested = False
            self._propagation_busy = False
            self._set_propagation_ui_enabled(True)
            self._clear_pending_propagation_state()
            self._sam_task_contexts.pop(task_id, None)
            self._set_status("Propagation stopped.", progress=0, total=1)
            return

        if state.remaining_chunks == 0:
            self._set_status(
                f"Propagation complete: {state.total_chunks}/{state.total_chunks} chunks.",
                progress=state.total_chunks,
                total=state.total_chunks,
            )
            self._propagation_busy = False
            self._set_propagation_ui_enabled(True)
            self._clear_pending_propagation_state()
            self._sam_task_contexts.pop(task_id, None)
            return

        self._propagation_busy = False
        self._set_propagation_ui_enabled(True)
        self._start_propagation_chunk_async(
            enabled_obj_ids=self._propagation_enabled_obj_ids or set(),
            state=state,
        )
        self._sam_task_contexts.pop(task_id, None)

    def _on_sam_propagate_stopped(self, task_id: str, last_emitted_frame_idx: int) -> None:
        context = self._sam_task_contexts.get(task_id, {})
        kind = context.get("kind")
        if kind == "prefetch":
            self._prefetch_busy = False
            self._prefetch_cancel_requested = False
            self._prefetch_cached_frame_idx = None
            self._prefetch_cached_seed_idx = None
            self._prefetch_cached_version = None
            self._sam_task_contexts.pop(task_id, None)
            return

        self._propagation_stop_requested = False
        self._propagation_continue_after_chunk = False
        if kind in {"manual", "auto"}:
            self._propagation_busy = False
            self._set_propagation_ui_enabled(True)
            self._clear_pending_propagation_state()

        self._sam_task_results[task_id] = None
        loop = self._sam_waiting.get(task_id)
        if loop is not None:
            loop.quit()
        self._sam_task_contexts.pop(task_id, None)

        if kind == "manual":
            seed_frame_idx = int(context.get("seed_frame_idx", 0))
            if last_emitted_frame_idx >= seed_frame_idx:
                self._set_status(
                    f"Propagation stopped at frame {last_emitted_frame_idx + 1}.",
                    progress=0,
                    total=1,
                )
            else:
                self._set_status("Propagation stopped before next frame.", progress=0, total=1)

    def _on_sam_task_failed(self, task_id: str, message: str, is_warning: bool) -> None:
        context = self._sam_task_contexts.get(task_id, {})
        kind = context.get("kind")
        if kind == "prefetch":
            self._prefetch_busy = False
            self._prefetch_cancel_requested = False
            self._prefetch_cached_frame_idx = None
            self._prefetch_cached_seed_idx = None
            self._prefetch_cached_version = None
            self._sam_task_contexts.pop(task_id, None)
            return
        title = "SAM3 task failed"
        if kind in {"manual", "auto"}:
            title = "Propagation failed"
        elif kind == "text_prompt":
            title = "Text prompt generation failed"
        elif kind == "update_experimental_settings":
            title = "Experimental settings update failed"
        if is_warning:
            QMessageBox.warning(self, title, message)
        else:
            QMessageBox.critical(self, title, message)
        if kind in {"manual", "auto"}:
            self._propagation_busy = False
            self._set_propagation_ui_enabled(True)
            self._clear_pending_propagation_state()
        self._sam_task_results[task_id] = None
        loop = self._sam_waiting.get(task_id)
        if loop is not None:
            loop.quit()
        self._sam_task_contexts.pop(task_id, None)

    def _on_sam_task_finished(self, task_id: str) -> None:
        self._sam_task_results[task_id] = True
        loop = self._sam_waiting.get(task_id)
        if loop is not None:
            loop.quit()
        self._sam_task_contexts.pop(task_id, None)

    def _handle_prefetch_frame(self, abs_frame_idx: int, output: SamFrameOutput, _session_idx: int, _total_frames: int) -> None:
        if self._prefetch_cancel_requested:
            return
        if self._prefetch_active_version is None or self._prefetch_active_version != self._prefetch_prompt_version:
            return
        target_frame_idx = self._prefetch_target_frame_idx
        if target_frame_idx is None or abs_frame_idx != target_frame_idx:
            return
        existing_output = self.outputs_by_frame.get(
            abs_frame_idx,
            SamFrameOutput(obj_ids=[], masks=[], boxes_xywh_norm=[], scores=[], tracker_scores=[]),
        )
        merged_output = self._merge_frame_outputs(existing_output, output)
        self.outputs_by_frame[abs_frame_idx] = merged_output
        self._sync_canonical_boxes_from_output(abs_frame_idx, output, preserve_locked=False)
        self._carry_prompts_forward(abs_frame_idx - 1, abs_frame_idx, output.obj_ids, translate=False)
        self._prefetch_cached_frame_idx = abs_frame_idx
        self._prefetch_cached_seed_idx = self._prefetch_seed_frame_idx
        self._prefetch_cached_version = self._prefetch_active_version
        self._prefetch_provenance_by_frame[abs_frame_idx] = {
            "seed_frame_idx": self._prefetch_seed_frame_idx,
            "obj_ids": [int(obj_id) for obj_id in output.obj_ids],
        }
        if abs_frame_idx == self.current_frame_idx:
            self._render_current_frame()

    def _handle_prefetch_chunk_done(self, _chunk_idx: int, _last_masked_frame_idx: int, _chunk_last_frame_idx: int) -> None:
        pass

    def _handle_prefetch_failed(self, message: str, is_warning: bool) -> None:
        if is_warning:
            self._set_status(f"Prefetch stopped: {message}", progress=0, total=1)
        else:
            self._set_status(f"Prefetch failed: {message}", progress=0, total=1)
        self._prefetch_cached_frame_idx = None
        self._prefetch_cached_seed_idx = None
        self._prefetch_cached_version = None

    def _handle_prefetch_finished(self) -> None:
        self._cleanup_prefetch_worker()
        self._prefetch_busy = False
        self._prefetch_cancel_requested = False
        if self._prefetch_pending_restart:
            self._prefetch_pending_restart = False
            self._schedule_prefetch_for_next_frame()

    def _cleanup_prefetch_worker(self) -> None:
        self._prefetch_worker = None
        self._prefetch_thread = None

    def _handle_propagation_chunk_done(self, chunk_idx: int, last_masked_frame_idx: int, chunk_last_frame_idx: int) -> None:
        state = self._pending_propagation
        if state is None:
            return
        chunk_seed_frame_idx = getattr(self, "_propagation_active_seed_frame_idx", state.next_seed_frame_idx)
        state.remaining_chunks -= 1
        state.completed_chunks += 1
        state.next_seed_frame_idx = last_masked_frame_idx
        self._set_status(
            f"Chunk {chunk_idx}/{state.total_chunks} complete.",
            progress=state.completed_chunks,
            total=state.total_chunks,
        )

        if self._propagation_stop_requested:
            self._propagation_stop_requested = False
            self._clear_pending_propagation_state()
            self._propagation_continue_after_chunk = False
            self._set_status("Propagation stopped.", progress=0, total=1)
            return

        if state.remaining_chunks == 0:
            self._set_status(
                f"Propagation complete: {state.total_chunks}/{state.total_chunks} chunks.",
                progress=state.total_chunks,
                total=state.total_chunks,
            )
            self._propagation_continue_after_chunk = False
            return

        self._propagation_continue_after_chunk = True

    def _handle_propagation_failed(self, message: str, is_warning: bool) -> None:
        if is_warning:
            QMessageBox.warning(self, "Propagation stopped", message)
        else:
            QMessageBox.critical(self, "Propagation failed", message)
        self._clear_pending_propagation_state()
        self._propagation_continue_after_chunk = False

    def _handle_propagation_finished(self) -> None:
        self._cleanup_propagation_worker()
        if self._propagation_continue_after_chunk and self._pending_propagation is not None:
            self._start_propagation_chunk_async(
                enabled_obj_ids=self._propagation_enabled_obj_ids or set(),
                state=self._pending_propagation,
            )
            return
        self._propagation_busy = False
        self._set_propagation_ui_enabled(True)
        if self._pending_propagation is None or self._pending_propagation.remaining_chunks == 0:
            self._clear_pending_propagation_state()

    def _cleanup_propagation_worker(self) -> None:
        self._propagation_worker = None
        self._propagation_thread = None

    def _run_propagation_chunk(
        self,
        seed_frame_idx: int,
        n_frames: int,
        use_carryover_sampling: bool,
        enabled_obj_ids: set[int],
    ) -> Optional[Tuple[int, int]]:
        if self._prefetch_busy:
            self._cancel_prefetch(restart=False)
        prompt_payload = self._build_seed_prompts(
            seed_frame_idx=seed_frame_idx,
            use_carryover_sampling=use_carryover_sampling,
            enabled_obj_ids=enabled_obj_ids,
        )
        if prompt_payload is None:
            return None
        task_id = self._enqueue_sam_task(
            "propagate",
            {
                "seed_frame_idx": seed_frame_idx,
                "n_frames": n_frames,
                "frame_paths": self.frame_paths,
                "prompt_payload": prompt_payload,
            },
        )
        self._sam_task_contexts[task_id] = {
            "kind": "auto",
            "seed_frame_idx": seed_frame_idx,
        }
        result = self._wait_for_sam_task(task_id)
        return result

    def _frame_output_has_masks(self, output: SamFrameOutput) -> bool:
        for mask in output.masks:
            if np.asarray(mask).any():
                return True
        return False

    def _refresh_after_prompt_edit(self) -> None:
        if self.segment_mode:
            if self._active_object_has_any_prompts():
                self.segment_current_frame()
            else:
                self._remove_object_output_from_frame(self.current_frame_idx, self.active_object_id)
                self._render_current_frame()
        else:
            self._render_current_frame()
        self._note_prompt_change_and_prefetch()

    def _active_object_has_any_prompts(self) -> bool:
        if self.active_object_id is None:
            return False
        frame_map = self.prompts_by_frame_obj.get(self.current_frame_idx, {})
        if frame_map.get(self.active_object_id):
            return True
        frame_boxes = self.box_prompts_by_frame_obj.get(self.current_frame_idx, {})
        return self.active_object_id in frame_boxes

    def _is_current_box_locked(self, frame_idx: int, obj_id: Optional[int]) -> bool:
        if obj_id is None:
            return False
        return self.box_locked_by_frame_obj.get(frame_idx, {}).get(obj_id, False)

    def _set_current_box_locked(self, frame_idx: int, obj_id: int, locked: bool) -> None:
        frame_map = self.box_locked_by_frame_obj.setdefault(frame_idx, {})
        if locked:
            frame_map[obj_id] = True
            return
        frame_map.pop(obj_id, None)
        if not frame_map:
            self.box_locked_by_frame_obj.pop(frame_idx, None)

    def _clone_frame_output(self, output: Optional[SamFrameOutput]) -> Optional[SamFrameOutput]:
        if output is None:
            return None
        return SamFrameOutput(
            obj_ids=list(output.obj_ids),
            masks=[np.asarray(mask).copy() for mask in output.masks],
            boxes_xywh_norm=[tuple(b) for b in output.boxes_xywh_norm],
            scores=list(output.scores),
            tracker_scores=list(output.tracker_scores),
        )

    def _stash_undo_state(self) -> None:
        if self.active_object_id is None:
            return
        frame_idx = self.current_frame_idx
        obj_id = self.active_object_id

        frame_map = self.prompts_by_frame_obj.get(frame_idx, {})
        prompts_present = obj_id in frame_map
        prompts = frame_map.get(obj_id, [])
        prompts_copy = [PointPrompt(x_px=p.x_px, y_px=p.y_px, is_positive=p.is_positive) for p in prompts]

        box_map = self.box_prompts_by_frame_obj.get(frame_idx, {})
        box_present = obj_id in box_map
        box = box_map.get(obj_id)
        box_copy = None if box is None else BoxPrompt(
            x1_px=box.x1_px,
            y1_px=box.y1_px,
            x2_px=box.x2_px,
            y2_px=box.y2_px,
        )

        lock_value = self.box_locked_by_frame_obj.get(frame_idx, {}).get(obj_id, False)
        output_copy = self._clone_frame_output(self.outputs_by_frame.get(frame_idx))
        output_version = self._output_version_by_frame.get(frame_idx)

        self._undo_state = {
            "frame_idx": frame_idx,
            "obj_id": obj_id,
            "prompts_present": prompts_present,
            "prompts": prompts_copy,
            "box_present": box_present,
            "box": box_copy,
            "lock_value": lock_value,
            "output": output_copy,
            "output_version": output_version,
        }

    def undo_last_prompt_change(self) -> None:
        state = self._undo_state
        if not state:
            return
        self._undo_state = None

        frame_idx = state["frame_idx"]
        obj_id = state["obj_id"]
        self._set_current_frame_idx(frame_idx)

        if obj_id is not None:
            for i in range(self.object_list.count()):
                item = self.object_list.item(i)
                if item and int(item.data(Qt.UserRole)) == obj_id:
                    self.object_list.setCurrentRow(i)
                    break

        frame_map = self.prompts_by_frame_obj.setdefault(frame_idx, {})
        if state["prompts_present"]:
            frame_map[obj_id] = list(state["prompts"])
        else:
            frame_map.pop(obj_id, None)

        box_map = self.box_prompts_by_frame_obj.setdefault(frame_idx, {})
        if state["box_present"] and state["box"] is not None:
            box_map[obj_id] = state["box"]
        else:
            box_map.pop(obj_id, None)
            if not box_map:
                self.box_prompts_by_frame_obj.pop(frame_idx, None)

        self._set_current_box_locked(frame_idx, obj_id, bool(state["lock_value"]))

        if state["output"] is None:
            self.outputs_by_frame.pop(frame_idx, None)
        else:
            self.outputs_by_frame[frame_idx] = state["output"]

        if state["output_version"] is None:
            self._output_version_by_frame.pop(frame_idx, None)
        else:
            self._output_version_by_frame[frame_idx] = int(state["output_version"])


        self._sync_current_box_lock_check()
        self.refresh_point_list()
        self._render_current_frame()
        self._note_prompt_change_and_prefetch()

    def _remove_object_output_from_frame(self, frame_idx: int, obj_id: Optional[int]) -> None:
        if obj_id is None:
            return
        output = self.outputs_by_frame.get(frame_idx)
        if output is None:
            return

        keep_idx = [i for i, existing_obj_id in enumerate(output.obj_ids) if existing_obj_id != obj_id]
        if len(keep_idx) == len(output.obj_ids):
            return
        if not keep_idx:
            self.outputs_by_frame.pop(frame_idx, None)
            self._output_version_by_frame.pop(frame_idx, None)
            return

        self.outputs_by_frame[frame_idx] = SamFrameOutput(
            obj_ids=[output.obj_ids[i] for i in keep_idx],
            masks=[output.masks[i] for i in keep_idx],
            boxes_xywh_norm=[output.boxes_xywh_norm[i] for i in keep_idx],
            scores=[output.scores[i] for i in keep_idx],
            tracker_scores=[output.tracker_scores[i] for i in keep_idx],
        )

    def _remove_object_annotations_from_frame(self, frame_idx: int, obj_id: Optional[int]) -> None:
        if obj_id is None:
            return
        frame_prompts = self.prompts_by_frame_obj.get(frame_idx)
        if frame_prompts is not None:
            frame_prompts.pop(obj_id, None)
            if not frame_prompts:
                self.prompts_by_frame_obj.pop(frame_idx, None)

        frame_boxes = self.box_prompts_by_frame_obj.get(frame_idx)
        if frame_boxes is not None:
            frame_boxes.pop(obj_id, None)
            if not frame_boxes:
                self.box_prompts_by_frame_obj.pop(frame_idx, None)

        self._set_current_box_locked(frame_idx, obj_id, False)
        self._remove_object_output_from_frame(frame_idx, obj_id)

    def _merge_frame_outputs(
        self,
        base_output: SamFrameOutput,
        new_output: SamFrameOutput,
    ) -> SamFrameOutput:
        merged = SamFrameOutput(
            obj_ids=list(base_output.obj_ids),
            masks=list(base_output.masks),
            boxes_xywh_norm=list(base_output.boxes_xywh_norm),
            scores=list(base_output.scores),
            tracker_scores=list(base_output.tracker_scores),
        )
        obj_to_idx = {obj_id: i for i, obj_id in enumerate(merged.obj_ids)}

        for i, obj_id in enumerate(new_output.obj_ids):
            if i >= len(new_output.masks) or i >= len(new_output.boxes_xywh_norm):
                continue
            score = new_output.scores[i] if i < len(new_output.scores) else 0.0
            tracker_score = new_output.tracker_scores[i] if i < len(new_output.tracker_scores) else 0.0
            if obj_id in obj_to_idx:
                idx = obj_to_idx[obj_id]
                merged.masks[idx] = new_output.masks[i]
                merged.boxes_xywh_norm[idx] = new_output.boxes_xywh_norm[i]
                merged.scores[idx] = score
                merged.tracker_scores[idx] = tracker_score
            else:
                merged.obj_ids.append(obj_id)
                merged.masks.append(new_output.masks[i])
                merged.boxes_xywh_norm.append(new_output.boxes_xywh_norm[i])
                merged.scores.append(score)
                merged.tracker_scores.append(tracker_score)
                obj_to_idx[obj_id] = len(merged.obj_ids) - 1

        return merged

    def _apply_prompts_for_seed(
        self,
        seed_frame_idx: int,
        use_carryover_sampling: bool,
        enabled_obj_ids: set[int],
    ) -> bool:
        frame_prompts = self.prompts_by_frame_obj.get(seed_frame_idx, {})
        frame_boxes = self.box_prompts_by_frame_obj.get(seed_frame_idx, {})

        frame_size = self._get_frame_size(seed_frame_idx)
        if frame_size is None:
            QMessageBox.critical(self, "Propagation setup failed", "Failed to load prompt source frame.")
            return False
        w, h = frame_size

        # Manual prompts are used only for the objects that have them on this seed frame.
        # For carryover chunks, objects without manual prompts fall back to mask-derived carryover prompts.
        prompts_to_apply: Dict[int, List[PointPrompt]] = {}
        boxes_to_apply: Dict[int, BoxPrompt] = {}
        masks_to_apply: Dict[int, np.ndarray] = {}
        missing_obj_ids: List[int] = []
        for obj_id, points in frame_prompts.items():
            if points and obj_id in enabled_obj_ids:
                prompts_to_apply[obj_id] = list(points)
        for obj_id in enabled_obj_ids:
            canonical_box = frame_boxes.get(obj_id)
            if canonical_box is not None:
                boxes_to_apply[obj_id] = canonical_box
        seed_output = self.outputs_by_frame.get(seed_frame_idx)
        if seed_output is not None:
            for i, obj_id in enumerate(seed_output.obj_ids):
                if obj_id in enabled_obj_ids and i < len(seed_output.masks):
                    masks_to_apply[obj_id] = np.asarray(seed_output.masks[i]).astype(np.float32)

        if not use_carryover_sampling:
            sampled_boxes = self._sample_boxes_from_seed_masks(seed_frame_idx, h, w)
            for obj_id in enabled_obj_ids:
                if obj_id in prompts_to_apply or obj_id in boxes_to_apply:
                    continue
                if obj_id in sampled_boxes:
                    boxes_to_apply[obj_id] = sampled_boxes[obj_id]
                else:
                    missing_obj_ids.append(obj_id)
        else:
            sampled_boxes = self._sample_boxes_from_seed_masks(seed_frame_idx, h, w)
            for obj_id, box in sampled_boxes.items():
                if (
                    obj_id in enabled_obj_ids
                    and obj_id not in prompts_to_apply
                    and obj_id not in boxes_to_apply
                ):
                    boxes_to_apply[obj_id] = box

        if missing_obj_ids:
            missing_names = [
                self._find_object(obj_id).name if self._find_object(obj_id) else str(obj_id)
                for obj_id in missing_obj_ids
            ]
            QMessageBox.warning(
                self,
                "Propagation setup failed",
                "These enabled objects have no prompts or existing mask on the seed frame: "
                + ", ".join(missing_names),
            )
            return False

        obj_ids = set(prompts_to_apply.keys()) | set(boxes_to_apply.keys())
        obj_ids |= set(masks_to_apply.keys())
        if not obj_ids:
            QMessageBox.information(
                self,
                "No prompts",
                "No prompts available for this chunk seed. Add prompts on this frame or create a seed mask first.",
            )
            return False

        for obj_id in obj_ids:
            points = prompts_to_apply.get(obj_id, [])
            points_rel = [[p.x_px / w, p.y_px / h] for p in points]
            labels = [1 if p.is_positive else 0 for p in points]
            box = boxes_to_apply.get(obj_id)
            if box is not None:
                points_rel.append([box.x1_px / w, box.y1_px / h])
                points_rel.append([box.x2_px / w, box.y2_px / h])
                labels.append(2)
                labels.append(3)
            if not points_rel:
                continue
            try:
                self.sam_adapter.add_object_points(
                    frame_idx=0,
                    obj_id=obj_id,
                    points_rel=points_rel,
                    labels=labels,
                    mask_input=masks_to_apply.get(obj_id),
                )
            except Exception as exc:
                QMessageBox.critical(self, "Propagation setup failed", str(exc))
                return False

        return True

    def _get_enabled_propagation_obj_ids(self) -> set[int]:
        return self._get_enabled_propagation_obj_ids_from_rows()

    def _get_enabled_propagation_obj_ids_from_rows(self) -> set[int]:
        enabled: set[int] = set()
        for obj_id in self._object_row_widgets.keys():
            if self._is_object_enabled_for_current_frame_propagation(int(obj_id)):
                enabled.add(int(obj_id))
        return enabled

    def _sample_boxes_from_seed_masks(
        self,
        seed_frame_idx: int,
        image_h: int,
        image_w: int,
    ) -> Dict[int, BoxPrompt]:
        sampled_boxes: Dict[int, BoxPrompt] = {}
        output = self.outputs_by_frame.get(seed_frame_idx)
        if output is None:
            return sampled_boxes

        for i, obj_id in enumerate(output.obj_ids):
            if i >= len(output.masks):
                continue
            mask = output.masks[i].astype(np.uint8)
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

    def _carry_prompts_forward(
        self,
        src_frame_idx: int,
        dst_frame_idx: int,
        obj_ids: List[int],
        *,
        translate: bool = True,
    ) -> None:
        src_prompts = self.prompts_by_frame_obj.get(src_frame_idx, {})
        if not src_prompts:
            return
        dst_prompts = self.prompts_by_frame_obj.setdefault(dst_frame_idx, {})
        for obj_id in obj_ids:
            points = src_prompts.get(obj_id)
            if not points:
                continue
            if translate and self.translate_prompts_on_propagation:
                dst_prompts[obj_id] = self._translate_prompts_by_box_delta(
                    src_frame_idx=src_frame_idx,
                    dst_frame_idx=dst_frame_idx,
                    obj_id=obj_id,
                    points=points,
                )
            else:
                dst_prompts[obj_id] = [
                    PointPrompt(x_px=p.x_px, y_px=p.y_px, is_positive=p.is_positive)
                    for p in points
                ]

    def _translate_prompts_by_box_delta(
        self,
        src_frame_idx: int,
        dst_frame_idx: int,
        obj_id: int,
        points: List[PointPrompt],
    ) -> List[PointPrompt]:
        src_box = self.box_prompts_by_frame_obj.get(src_frame_idx, {}).get(obj_id)
        dst_box = self.box_prompts_by_frame_obj.get(dst_frame_idx, {}).get(obj_id)
        frame_size = self._get_frame_size(dst_frame_idx)
        if src_box is None or dst_box is None or frame_size is None:
            return [
                PointPrompt(x_px=p.x_px, y_px=p.y_px, is_positive=p.is_positive)
                for p in points
            ]

        width, height = frame_size
        src_cx = (src_box.x1_px + src_box.x2_px) / 2.0
        src_cy = (src_box.y1_px + src_box.y2_px) / 2.0
        dst_cx = (dst_box.x1_px + dst_box.x2_px) / 2.0
        dst_cy = (dst_box.y1_px + dst_box.y2_px) / 2.0
        dx = dst_cx - src_cx
        dy = dst_cy - src_cy

        translated: List[PointPrompt] = []
        for p in points:
            x_px = int(round(p.x_px + dx))
            y_px = int(round(p.y_px + dy))
            x_px = max(0, min(width - 1, x_px))
            y_px = max(0, min(height - 1, y_px))
            translated.append(PointPrompt(x_px=x_px, y_px=y_px, is_positive=p.is_positive))
        return translated

    def _sync_canonical_boxes_from_output(
        self,
        frame_idx: int,
        output: SamFrameOutput,
        preserve_locked: bool = True,
    ) -> None:
        frame_size = self._get_frame_size(frame_idx)
        if frame_size is None:
            return
        width, height = frame_size
        frame_boxes = self.box_prompts_by_frame_obj.setdefault(frame_idx, {})
        for i, obj_id in enumerate(output.obj_ids):
            if i >= len(output.boxes_xywh_norm):
                continue
            if preserve_locked and self._is_current_box_locked(frame_idx, obj_id):
                continue
            box = self._box_from_norm(output.boxes_xywh_norm[i], width, height)
            if box is None:
                continue
            frame_boxes[obj_id] = box
            self._set_current_box_locked(frame_idx, obj_id, False)

    def _box_from_norm(
        self,
        box_xywh_norm: Tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> Optional[BoxPrompt]:
        x_norm, y_norm, w_norm, h_norm = box_xywh_norm
        x1 = int(round(x_norm * width))
        y1 = int(round(y_norm * height))
        box_w = int(round(w_norm * width))
        box_h = int(round(h_norm * height))
        x2 = x1 + box_w
        y2 = y1 + box_h
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return BoxPrompt(x1_px=x1, y1_px=y1, x2_px=x2, y2_px=y2)

    def _clear_pending_propagation_state(self) -> None:
        self._pending_propagation = None
        self.propagate_btn.setText("Propagate")
        self._propagation_enabled_obj_ids = None
        self._propagation_view_frame_idx = None
        self._propagation_continue_after_chunk = False
        self._propagation_active_chunk_idx = None
        self._propagation_active_seed_frame_idx = None
        self._propagation_task_id = None
        self._one_session_chunk_prompt_version = None
        self._propagation_stop_requested = False
        self._propagation_seen_obj_ids.clear()
        self._propagation_lost_obj_ids.clear()
        self._propagation_loss_notified = False
        if hasattr(self, "stop_propagate_btn"):
            self.stop_propagate_btn.setEnabled(False)
        self._schedule_prefetch_for_next_frame()

    def export_annotations(self) -> None:
        if not self.frame_paths:
            QMessageBox.information(self, "Nothing to export", "Load a frame directory first.")
            return

        export_format, ok = QInputDialog.getItem(
            self,
            "Export Format",
            "Choose export format:",
            ["COCO", "Perk Format"],
            0,
            False,
        )
        if not ok or not export_format:
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not out_dir:
            return

        output_dir = Path(out_dir)
        if export_format == "Perk Format":
            exporter = PerkExporter(output_dir)
            objects = [ExportPerkObjectInfo(obj_id=o.obj_id, name=o.name) for o in self.objects]
            boxes: Dict[int, Dict[int, ExportPerkBoxPrompt]] = {}
            for frame_idx, per_obj in self.box_prompts_by_frame_obj.items():
                boxes[frame_idx] = {}
                for obj_id, box in per_obj.items():
                    boxes[frame_idx][obj_id] = ExportPerkBoxPrompt(
                        x1_px=box.x1_px,
                        y1_px=box.y1_px,
                        x2_px=box.x2_px,
                        y2_px=box.y2_px,
                    )
            csv_path = exporter.export(
                frame_paths=self.frame_paths,
                objects=objects,
                boxes_by_frame_obj=boxes,
            )
            QMessageBox.information(self, "Export complete", f"Saved: {csv_path}")
            return

        exporter = CocoExporter(output_dir)
        objects = [ExportObjectInfo(obj_id=o.obj_id, name=o.name) for o in self.objects]

        prompts: Dict[int, Dict[int, List[ExportPointPrompt]]] = {}
        for frame_idx, per_obj in self.prompts_by_frame_obj.items():
            prompts[frame_idx] = {}
            for obj_id, plist in per_obj.items():
                prompts[frame_idx][obj_id] = [
                    ExportPointPrompt(x_px=p.x_px, y_px=p.y_px, is_positive=p.is_positive)
                    for p in plist
                ]

        outputs: Dict[int, ExportSamFrameOutput] = {}
        for frame_idx, out in self.outputs_by_frame.items():
            outputs[frame_idx] = ExportSamFrameOutput(
                obj_ids=out.obj_ids,
                masks=out.masks,
                boxes_xywh_norm=out.boxes_xywh_norm,
                scores=out.scores,
            )

        json_path = exporter.export(
            frame_paths=self.frame_paths,
            objects=objects,
            prompts_by_frame_obj=prompts,
            outputs_by_frame=outputs,
        )
        QMessageBox.information(self, "Export complete", f"Saved: {json_path}")

    def save_session_dialog(self) -> None:
        if not self.frame_paths:
            QMessageBox.information(self, "Nothing to save", "Load a frame directory first.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Select Session Folder")
        if not directory:
            return
        session_dir = Path(directory)
        self._save_session(session_dir)
        self._session_dir = session_dir
        self._ensure_autosave_running_if_enabled()

    def load_session_dialog(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Session Folder")
        if not directory:
            return
        session_dir = Path(directory)
        session_path = session_dir / "session.json"
        if not session_path.exists():
            QMessageBox.warning(self, "Missing session", "No session.json found in that folder.")
            return
        self._load_session(session_path)
        self._session_dir = session_dir
        self._ensure_autosave_running_if_enabled()

    def _save_session(self, session_dir: Path) -> None:
        if self.image_dir is None:
            return
        masks_dir = session_dir / "masks"
        frame_files = [p.name for p in self.frame_paths]

        data = {
            "version": 6,
            "frame_dir": str(self.image_dir),
            "frame_files": frame_files,
            "current_frame_idx": int(self.current_frame_idx),
            "active_object_id": int(self.active_object_id) if self.active_object_id is not None else None,
            "checkpoint_path": self._read_optional_combo_path(self.checkpoint_combo),
            "flagged_frames": self._sorted_flagged_frames(),
            "objects": [
                {
                    "obj_id": obj.obj_id,
                    "name": obj.name,
                    "color_bgr": list(obj.color_bgr),
                }
                for obj in self.objects
            ],
            "object_view": {
                "hidden_obj_ids": sorted(int(obj_id) for obj_id in self.hidden_obj_ids),
                "solo_object_id": int(self.solo_object_id) if self.solo_object_id is not None else None,
            },
            "prompts": {},
            "boxes": {},
            "box_locks": {},
            "propagation_overrides": {},
            "outputs": {},
            "view": {
                "show_prompts": self.show_prompts,
                "show_segmentations": self.show_segmentations,
                "show_boxes": self.show_boxes,
                "show_box_titles": self.show_box_titles,
                "segmentation_opacity": float(self.segmentation_opacity),
                "box_line_thickness": int(self.box_line_thickness),
            },
            "propagation": {
                "mode": self.propagation_mode,
                "n_frames": int(self.n_propagate_spin.value()),
                "chunks": int(self.chunks_spin.value()),
                "use_target_frame": bool(self.use_target_frame_check.isChecked()),
                "target_frame_idx": int(self.target_frame_spin.value() - 1),
                "translate_prompts": bool(self.translate_prompts_on_propagation),
                "use_point_prompts": bool(self.use_point_prompts_for_propagation),
                "auto_propagate_next": bool(self.auto_propagate_next),
            },
            "experimental": {
                "recondition_every_nth_frame": int(self.recondition_every_nth_frame),
                "recondition_high_conf_thresh": float(self.recondition_high_conf_thresh),
                "recondition_high_iou_thresh": float(self.recondition_high_iou_thresh),
                "use_one_session_chunked_propagation": bool(self.use_one_session_chunked_propagation),
            },
            "prompt_mode_index": int(self.prompt_mode_combo.currentIndex()),
        }

        for frame_idx, per_obj in self.prompts_by_frame_obj.items():
            data["prompts"][str(frame_idx)] = {}
            for obj_id, plist in per_obj.items():
                data["prompts"][str(frame_idx)][str(obj_id)] = [
                    {"x_px": p.x_px, "y_px": p.y_px, "is_positive": p.is_positive}
                    for p in plist
                ]

        for frame_idx, per_obj in self.box_prompts_by_frame_obj.items():
            data["boxes"][str(frame_idx)] = {}
            for obj_id, box in per_obj.items():
                data["boxes"][str(frame_idx)][str(obj_id)] = {
                    "x1_px": box.x1_px,
                    "y1_px": box.y1_px,
                    "x2_px": box.x2_px,
                    "y2_px": box.y2_px,
                }

        for frame_idx, per_obj in self.manual_propagation_overrides_by_frame_obj.items():
            if not per_obj:
                continue
            data["propagation_overrides"][str(frame_idx)] = {
                str(obj_id): bool(enabled)
                for obj_id, enabled in per_obj.items()
            }

        for frame_idx, per_obj in self.box_locked_by_frame_obj.items():
            data["box_locks"][str(frame_idx)] = {
                str(obj_id): bool(locked) for obj_id, locked in per_obj.items()
            }

        for frame_idx, output in self.outputs_by_frame.items():
            entry = {
                "obj_ids": [int(x) for x in output.obj_ids],
                "boxes_xywh_norm": [list(map(float, b)) for b in output.boxes_xywh_norm],
                "scores": [float(s) for s in output.scores],
                "tracker_scores": [float(s) for s in output.tracker_scores],
                "mask_paths": [],
            }
            for obj_id, mask in zip(output.obj_ids, output.masks):
                if mask is None or not np.asarray(mask).any():
                    entry["mask_paths"].append(None)
                    continue
                mask_name = f"frame_{frame_idx:05d}_obj_{int(obj_id)}.png"
                mask_path = masks_dir / mask_name
                write_mask_png(mask_path, np.asarray(mask))
                entry["mask_paths"].append(str(mask_path.relative_to(session_dir)))
            data["outputs"][str(frame_idx)] = entry

        write_session_json(session_dir / "session.json", data)
        self._set_status("Session saved.", progress=1, total=1)

    def _load_session(self, session_path: Path) -> None:
        data = read_session_json(session_path)
        session_dir = session_path.parent
        progress_dialog = QProgressDialog("Loading session...", None, 0, 100, self)
        progress_dialog.setWindowTitle("Loading Session")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setAutoClose(True)
        progress_dialog.setAutoReset(True)

        def update_progress(value: int, text: str) -> None:
            progress_dialog.setLabelText(text)
            progress_dialog.setValue(max(0, min(100, value)))
            QApplication.processEvents()

        progress_dialog.show()
        update_progress(5, "Reading session metadata...")
        frame_dir = Path(data.get("frame_dir", ""))
        if not frame_dir.exists():
            alt_dir = QFileDialog.getExistingDirectory(self, "Select Frame Directory for Session")
            if not alt_dir:
                progress_dialog.close()
                return
            frame_dir = Path(alt_dir)

        frame_files = data.get("frame_files", [])
        if not frame_files:
            progress_dialog.close()
            QMessageBox.warning(self, "Invalid session", "No frame list found in session.")
            return

        update_progress(12, "Preparing frame directory...")
        self.image_dir = frame_dir
        self.frame_paths = [frame_dir / name for name in frame_files]
        self.segment_mode = False
        self.mode_label.setText("Mode: Box Annotation")
        self._clear_pending_propagation_state()
        self._box_draw_start_xy = None
        self._box_draw_start_ui_xy = None
        self._display_image_size = (0, 0)
        self._zoom_multiplier = 1.0
        self._pan_offset_ui = (0.0, 0.0)
        self._pan_drag_last_ui_xy = None
        self._box_edit_active = False
        self._box_drag_mode = None
        self._box_drag_reference_box = None
        self._box_drag_offset_xy = (0, 0)
        self._selected_box_obj_id = None
        self.flagged_frame_indices.clear()
        self._box_rubber_band.hide()
        self._clear_text_prompt_proposals_if_needed()
        self._reset_prefetch_state()

        existing_names = {p.name for p in frame_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
        missing = [name for name in frame_files if name not in existing_names]
        if missing:
            QMessageBox.warning(
                self,
                "Missing frames",
                "Some frames listed in the session were not found in the directory. "
                "Annotations will still load, but rendering may be incomplete.",
            )

        self._checkpoint_path = data.get("checkpoint_path")
        if self._checkpoint_path:
            self.checkpoint_combo.setCurrentText(self._checkpoint_path)
        else:
            self.checkpoint_combo.setCurrentIndex(0)
        experimental = data.get("experimental", {})
        self.recondition_every_nth_frame = int(
            experimental.get(
                "recondition_every_nth_frame",
                DEFAULT_RECONDITION_EVERY_NTH_FRAME,
            )
        )
        self.recondition_high_conf_thresh = float(
            experimental.get(
                "recondition_high_conf_thresh",
                DEFAULT_RECONDITION_HIGH_CONF_THRESH,
            )
        )
        self.recondition_high_iou_thresh = float(
            experimental.get(
                "recondition_high_iou_thresh",
                DEFAULT_RECONDITION_HIGH_IOU_THRESH,
            )
        )
        self.use_one_session_chunked_propagation = bool(
            experimental.get("use_one_session_chunked_propagation", False)
        )
        self._sync_experimental_controls()
        update_progress(20, "Initializing SAM worker...")
        self._initialize_sam_worker(show_errors=False)
        self._apply_experimental_settings_live()

        update_progress(28, "Restoring objects...")
        self.objects.clear()
        self.object_list.clear()
        self._object_row_widgets.clear()
        for entry in data.get("objects", []):
            obj = ObjectInfo(
                obj_id=int(entry["obj_id"]),
                name=entry["name"],
                color_bgr=tuple(entry.get("color_bgr", (255, 255, 255))),
            )
            self.objects.append(obj)
            item = QListWidgetItem()
            item.setData(Qt.UserRole, obj.obj_id)
            item.setForeground(QBrush(QColor(obj.color_bgr[2], obj.color_bgr[1], obj.color_bgr[0])))
            self.object_list.addItem(item)
            self.object_list.setItemWidget(item, self._create_object_row_widget(obj.obj_id))

        object_view = data.get("object_view", {})
        self.hidden_obj_ids = {int(obj_id) for obj_id in object_view.get("hidden_obj_ids", [])}
        solo_object_id = object_view.get("solo_object_id")
        self.solo_object_id = int(solo_object_id) if solo_object_id is not None else None

        self.next_obj_id = 1 + max((obj.obj_id for obj in self.objects), default=0)
        valid_obj_ids = {obj.obj_id for obj in self.objects}
        self.hidden_obj_ids &= valid_obj_ids
        if self.solo_object_id not in valid_obj_ids:
            self.solo_object_id = None
        self.active_object_id = data.get("active_object_id")
        self.flagged_frame_indices = {
            int(frame_idx)
            for frame_idx in data.get("flagged_frames", [])
            if 0 <= int(frame_idx) < len(self.frame_paths)
        }

        update_progress(36, "Restoring prompts...")
        self.prompts_by_frame_obj.clear()
        for frame_idx, per_obj in data.get("prompts", {}).items():
            frame_map: Dict[int, List[PointPrompt]] = {}
            for obj_id, plist in per_obj.items():
                frame_map[int(obj_id)] = [
                    PointPrompt(
                        x_px=int(p["x_px"]),
                        y_px=int(p["y_px"]),
                        is_positive=bool(p["is_positive"]),
                    )
                    for p in plist
                ]
            self.prompts_by_frame_obj[int(frame_idx)] = frame_map

        update_progress(46, "Restoring boxes...")
        self.box_prompts_by_frame_obj.clear()
        for frame_idx, per_obj in data.get("boxes", {}).items():
            frame_map: Dict[int, BoxPrompt] = {}
            for obj_id, box in per_obj.items():
                frame_map[int(obj_id)] = BoxPrompt(
                    x1_px=int(box["x1_px"]),
                    y1_px=int(box["y1_px"]),
                    x2_px=int(box["x2_px"]),
                    y2_px=int(box["y2_px"]),
                )
            self.box_prompts_by_frame_obj[int(frame_idx)] = frame_map

        update_progress(54, "Restoring box locks...")
        self.box_locked_by_frame_obj.clear()
        for frame_idx, per_obj in data.get("box_locks", {}).items():
            self.box_locked_by_frame_obj[int(frame_idx)] = {
                int(obj_id): bool(locked) for obj_id, locked in per_obj.items()
            }

        self.manual_propagation_overrides_by_frame_obj.clear()
        for frame_idx, per_obj in data.get("propagation_overrides", {}).items():
            overrides = {
                int(obj_id): bool(enabled) for obj_id, enabled in per_obj.items()
            }
            if overrides:
                self.manual_propagation_overrides_by_frame_obj[int(frame_idx)] = overrides

        update_progress(60, "Restoring masks and outputs...")
        self.outputs_by_frame.clear()
        outputs_items = list(data.get("outputs", {}).items())
        total_outputs = max(1, len(outputs_items))
        for output_idx, (frame_idx, entry) in enumerate(outputs_items, start=1):
            frame_idx_int = int(frame_idx)
            obj_ids = [int(x) for x in entry.get("obj_ids", [])]
            boxes = [tuple(map(float, b)) for b in entry.get("boxes_xywh_norm", [])]
            scores = [float(s) for s in entry.get("scores", [])]
            tracker_scores = [float(s) for s in entry.get("tracker_scores", [])]
            if len(tracker_scores) < len(obj_ids):
                tracker_scores.extend([0.0] * (len(obj_ids) - len(tracker_scores)))
            mask_paths = entry.get("mask_paths", [])
            masks: List[np.ndarray] = []
            frame_size = self._get_frame_size(frame_idx_int)
            fallback_shape = (1, 1) if frame_size is None else (frame_size[1], frame_size[0])
            for idx in range(len(obj_ids)):
                path = mask_paths[idx] if idx < len(mask_paths) else None
                if path is None:
                    masks.append(np.zeros(fallback_shape, dtype=bool))
                    continue
                try:
                    masks.append(read_mask_png(session_dir / path))
                except Exception:
                    masks.append(np.zeros(fallback_shape, dtype=bool))
            self.outputs_by_frame[frame_idx_int] = SamFrameOutput(
                obj_ids=obj_ids,
                masks=masks,
                boxes_xywh_norm=boxes,
                scores=scores,
                tracker_scores=tracker_scores,
            )
            self._output_version_by_frame[frame_idx_int] = self._prefetch_prompt_version
            update_progress(60 + int(20 * output_idx / total_outputs), f"Restoring masks and outputs... {output_idx}/{total_outputs}")

        if self.outputs_by_frame:
            self.segment_mode = True
            self.mode_label.setText("Mode: Segment" if self._use_point_prompt_mode() else "Mode: Box Annotation")

        update_progress(84, "Restoring view settings...")
        view = data.get("view", {})
        self.show_prompts = bool(view.get("show_prompts", True))
        self.show_segmentations = bool(view.get("show_segmentations", True))
        self.show_boxes = bool(view.get("show_boxes", True))
        self.show_box_titles = bool(view.get("show_box_titles", True))
        self.segmentation_opacity = float(view.get("segmentation_opacity", self.segmentation_opacity))
        self.box_line_thickness = int(view.get("box_line_thickness", self.box_line_thickness))
        self.show_prompts_check.setChecked(self.show_prompts)
        self.show_segmentations_check.setChecked(self.show_segmentations)
        self.show_boxes_check.setChecked(self.show_boxes)
        self.show_box_titles_check.setChecked(self.show_box_titles)
        self.segmentation_opacity_spin.setValue(self.segmentation_opacity)
        self.box_line_thickness_spin.setValue(self.box_line_thickness)

        update_progress(90, "Restoring propagation settings...")
        propagation = data.get("propagation", {})
        self.n_propagate_spin.setValue(int(propagation.get("n_frames", self.n_propagate_spin.value())))
        self.chunks_spin.setValue(int(propagation.get("chunks", self.chunks_spin.value())))
        use_target = bool(propagation.get("use_target_frame", False))
        target_idx = int(propagation.get("target_frame_idx", self.current_frame_idx))
        self.translate_prompts_on_propagation = bool(propagation.get("translate_prompts", True))
        self.translate_prompts_check.setChecked(self.translate_prompts_on_propagation)
        self.use_point_prompts_for_propagation = bool(propagation.get("use_point_prompts", True))
        self.use_point_prompts_for_propagation_check.setChecked(self.use_point_prompts_for_propagation)
        self.propagation_mode = str(propagation.get("mode", "tracker"))
        self._sync_propagation_mode_controls()
        self.auto_propagate_next = bool(propagation.get("auto_propagate_next", False))
        self.auto_propagate_next_check.setChecked(self.auto_propagate_next)
        self.use_target_frame = use_target
        self.use_target_frame_check.setChecked(use_target)
        if self.frame_paths:
            target_idx = max(0, min(len(self.frame_paths) - 1, target_idx))
        self.target_frame_spin.setValue(target_idx + 1)

        self.prompt_mode_combo.setCurrentIndex(int(data.get("prompt_mode_index", 2)))

        update_progress(96, "Finalizing session...")
        self.current_frame_idx = int(data.get("current_frame_idx", 0))
        if self.current_frame_idx < 0 or self.current_frame_idx >= len(self.frame_paths):
            self.current_frame_idx = 0

        self._sync_frame_navigation_controls()
        self._refresh_flagged_frame_list()
        self._refresh_object_list_visuals()
        if self.active_object_id is not None:
            for i in range(self.object_list.count()):
                item = self.object_list.item(i)
                if item and item.data(Qt.UserRole) == self.active_object_id:
                    self.object_list.setCurrentRow(i)
                    break
        self._sync_current_box_lock_check()
        self.refresh_point_list()
        self._render_current_frame()
        self._set_status("Session loaded.", progress=1, total=1)
        self._output_version_by_frame = {
            frame_idx: self._prefetch_prompt_version for frame_idx in self.outputs_by_frame.keys()
        }
        self._undo_state = None
        self._last_prompt_edit_frame = None
        update_progress(100, "Session loaded.")
        progress_dialog.close()

    def _render_current_frame(self) -> None:
        if not self.frame_paths:
            self.image_label.setText("Load a frame directory")
            self.frame_label.setText("Frame: -/-")
            return
        self._sync_selected_box_state()

        frame_path = self.frame_paths[self.current_frame_idx]
        img = self._read_frame_bgr(self.current_frame_idx)
        if img is None:
            self.image_label.setText(f"Failed to read: {frame_path.name}")
            return

        if self.current_frame_idx in self.outputs_by_frame:
            img = self._draw_output_overlay(img, self.outputs_by_frame[self.current_frame_idx])

        img = self._draw_text_prompt_proposals_overlay(img)
        img = self._draw_points_overlay(img)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        label_w = max(1, self.image_label.width())
        label_h = max(1, self.image_label.height())
        fit_scale = min(label_w / w, label_h / h)
        self._display_scale = fit_scale * self._zoom_multiplier
        scaled_w = max(1, int(round(w * self._display_scale)))
        scaled_h = max(1, int(round(h * self._display_scale)))
        self._clamp_pan_offset(label_w=label_w, label_h=label_h, scaled_w=scaled_w, scaled_h=scaled_h)

        scaled = pixmap.scaled(scaled_w, scaled_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        x_off, y_off = self._compute_display_offset(label_w, label_h, scaled_w, scaled_h)
        self._display_pixmap = scaled
        self._display_image_size = (scaled_w, scaled_h)
        self._display_offset = (x_off, y_off)

        canvas = QPixmap(label_w, label_h)
        canvas.fill(Qt.black)
        painter = None
        try:
            from PySide6.QtGui import QPainter
            painter = QPainter(canvas)
            painter.drawPixmap(x_off, y_off, scaled)
        finally:
            if painter is not None:
                painter.end()

        self.image_label.setPixmap(canvas)
        flagged_suffix = " [Flagged]" if self.current_frame_idx in self.flagged_frame_indices else ""
        self.frame_label.setText(f"Frame: {self.current_frame_idx + 1}/{len(self.frame_paths)}{flagged_suffix}")
        self._sync_frame_navigation_controls()
        self._sync_flagged_frame_selection()
        self.refresh_point_list()
        self._sync_current_box_lock_check()

    def _draw_output_overlay(self, img: np.ndarray, output: SamFrameOutput) -> np.ndarray:
        overlay = img.copy()
        h, w = img.shape[:2]
        mask_alpha = float(max(0.0, min(1.0, self.segmentation_opacity)))

        for i, obj_id in enumerate(output.obj_ids):
            if not self._is_object_visible(obj_id):
                continue
            obj = self._find_object(obj_id)
            color = obj.color_bgr if obj else (255, 255, 0)

            mask = output.masks[i].astype(np.uint8)
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            if self.show_segmentations and mask_alpha > 0.0:
                color_img = np.zeros_like(img)
                color_img[:] = color
                blended = cv2.addWeighted(overlay, 1.0 - mask_alpha, color_img, mask_alpha, 0)
                overlay = np.where(mask[..., None].astype(bool), blended, overlay)

        return overlay

    def _draw_text_prompt_proposals_overlay(self, img: np.ndarray) -> np.ndarray:
        if self._text_prompt_proposals_frame_idx != self.current_frame_idx or not self._text_prompt_proposals:
            return img
        overlay = img.copy()
        mask_alpha = 0.25
        proposal_color = (0, 200, 255)
        for idx, proposal in enumerate(self._text_prompt_proposals):
            mask = np.asarray(proposal.mask).astype(np.uint8)
            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            color_img = np.zeros_like(img)
            color_img[:] = proposal_color
            blended = cv2.addWeighted(overlay, 1.0 - mask_alpha, color_img, mask_alpha, 0)
            overlay = np.where(mask[..., None].astype(bool), blended, overlay)
            x1, y1, x2, y2 = proposal.box_xyxy_px
            cv2.rectangle(overlay, (x1, y1), (x2, y2), proposal_color, 1)
            cv2.putText(
                overlay,
                f"T{idx + 1}:{proposal.score:.2f}",
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                proposal_color,
                1,
                cv2.LINE_AA,
            )
        return overlay

    def _draw_points_overlay(self, img: np.ndarray) -> np.ndarray:
        frame_boxes = self.box_prompts_by_frame_obj.get(self.current_frame_idx, {})
        selected_box = self._is_selected_box_prompt()
        for obj_id, box in frame_boxes.items():
            if not self._is_object_visible(obj_id):
                continue
            obj = self._find_object(obj_id)
            color = obj.color_bgr if obj else (255, 255, 255)
            if self.show_boxes:
                if selected_box and obj_id == self.active_object_id:
                    outline_pad = max(1, int(self.box_line_thickness))
                    self._draw_box_outline(
                        img,
                        BoxPrompt(
                            x1_px=max(0, box.x1_px - outline_pad),
                            y1_px=max(0, box.y1_px - outline_pad),
                            x2_px=box.x2_px + outline_pad,
                            y2_px=box.y2_px + outline_pad,
                        ),
                        (255, 255, 255),
                        1,
                        dashed=False,
                    )
                self._draw_box_outline(img, box, color, self.box_line_thickness, dashed=False)
                if self.show_box_titles:
                    label = obj.name if obj else str(obj_id)
                    if self._is_current_box_locked(self.current_frame_idx, obj_id):
                        label += " [L]"
                    cv2.putText(
                        img,
                        label,
                        (box.x1_px, max(20, box.y1_px - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        self.box_line_thickness,
                        cv2.LINE_AA,
                    )
            if self.show_boxes and self._should_draw_box_handles(obj_id):
                self._draw_box_handles(img, box, color)

        if self.show_prompts:
            frame_map = self.prompts_by_frame_obj.get(self.current_frame_idx, {})
            selected_prompt_idx = self._get_selected_point_prompt_idx()
            for obj_id, plist in frame_map.items():
                if not self._is_object_visible(obj_id):
                    continue
                obj = self._find_object(obj_id)
                color = obj.color_bgr if obj else (255, 255, 255)
                for idx, p in enumerate(plist):
                    radius = 4
                    if obj_id == self.active_object_id and idx == selected_prompt_idx:
                        cv2.circle(img, (p.x_px, p.y_px), radius + 1, (255, 255, 255), 1)
                    if p.is_positive:
                        cv2.circle(img, (p.x_px, p.y_px), radius, color, -1)
                    else:
                        if obj_id == self.active_object_id and idx == selected_prompt_idx:
                            cv2.line(
                                img,
                                (p.x_px - (radius + 1), p.y_px - (radius + 1)),
                                (p.x_px + (radius + 1), p.y_px + (radius + 1)),
                                (255, 255, 255),
                                1,
                            )
                            cv2.line(
                                img,
                                (p.x_px - (radius + 1), p.y_px + (radius + 1)),
                                (p.x_px + (radius + 1), p.y_px - (radius + 1)),
                                (255, 255, 255),
                                1,
                            )
                        cv2.line(
                            img,
                            (p.x_px - radius, p.y_px - radius),
                            (p.x_px + radius, p.y_px + radius),
                            color,
                            2,
                        )
                        cv2.line(
                            img,
                            (p.x_px - radius, p.y_px + radius),
                            (p.x_px + radius, p.y_px - radius),
                            color,
                            2,
                        )

        return img

    def _get_selected_point_prompt_idx(self) -> Optional[int]:
        row = self.point_list.currentRow()
        if row < 0 or row >= len(self._active_prompt_rows):
            return None
        prompt_type, idx = self._active_prompt_rows[row]
        if prompt_type != "point":
            return None
        return idx

    def _is_selected_box_prompt(self) -> bool:
        row = self.point_list.currentRow()
        if row < 0 or row >= len(self._active_prompt_rows):
            return False
        prompt_type, _idx = self._active_prompt_rows[row]
        return prompt_type == "box"

    def _is_box_mode(self) -> bool:
        return not self._use_point_prompt_mode()

    def is_box_mode_active(self) -> bool:
        return self._is_box_mode()

    def _begin_box_interaction(
        self,
        x_px: int,
        y_px: int,
    ) -> Tuple[Optional[BoxPrompt], bool]:
        existing_box = self.box_prompts_by_frame_obj.get(self.current_frame_idx, {}).get(self.active_object_id)
        if existing_box is None:
            self._selected_box_obj_id = None
            self._box_draw_start_xy = (x_px, y_px)
            self._box_drag_mode = "new"
            self._box_drag_reference_box = None
            self._box_drag_offset_xy = (0, 0)
            return BoxPrompt(x_px, y_px, x_px, y_px), False

        handle_hit = self._hit_test_box_handle(existing_box, x_px, y_px, handles_only=True)
        inside_box = self._point_in_box(existing_box, x_px, y_px)

        if self._selected_box_obj_id != self.active_object_id:
            if inside_box:
                self._selected_box_obj_id = self.active_object_id
                self._box_draw_start_xy = None
                self._box_drag_mode = None
                self._box_drag_reference_box = None
                self._box_drag_offset_xy = (0, 0)
                return None, False
            self._selected_box_obj_id = None
            self._box_draw_start_xy = (x_px, y_px)
            self._box_drag_mode = "new"
            self._box_drag_reference_box = None
            self._box_drag_offset_xy = (0, 0)
            return BoxPrompt(x_px, y_px, x_px, y_px), False

        self._selected_box_obj_id = self.active_object_id
        if handle_hit is not None:
            self._box_drag_mode = handle_hit
            self._box_drag_reference_box = existing_box
            self._box_draw_start_xy = (x_px, y_px)
            self._box_drag_offset_xy = (0, 0)
            return existing_box, True

        if inside_box:
            self._box_drag_mode = "move"
            self._box_drag_reference_box = existing_box
            self._box_draw_start_xy = (x_px, y_px)
            self._box_drag_offset_xy = (x_px - existing_box.x1_px, y_px - existing_box.y1_px)
            return existing_box, True

        self._selected_box_obj_id = None
        self._box_draw_start_xy = (x_px, y_px)
        self._box_drag_mode = "new"
        self._box_drag_reference_box = None
        self._box_drag_offset_xy = (0, 0)
        return BoxPrompt(x_px, y_px, x_px, y_px), False

    def _hit_test_box_handle(
        self,
        box: BoxPrompt,
        x_px: int,
        y_px: int,
        handles_only: bool = False,
    ) -> Optional[str]:
        threshold_px = self._box_handle_radius_px(box) + 4
        handle_points = self._get_box_handle_points(box)
        best_name: Optional[str] = None
        best_dist_sq: Optional[int] = None
        for name, (cx, cy) in handle_points.items():
            dist_sq = (cx - x_px) ** 2 + (cy - y_px) ** 2
            if dist_sq > threshold_px ** 2:
                continue
            if best_dist_sq is None or dist_sq < best_dist_sq:
                best_name = name
                best_dist_sq = dist_sq

        if best_name is not None:
            return best_name

        if handles_only:
            return None
        if self._point_in_box(box, x_px, y_px):
            return "move"
        return None

    def _get_box_handle_points(self, box: BoxPrompt) -> Dict[str, Tuple[int, int]]:
        mid_x = (box.x1_px + box.x2_px) // 2
        mid_y = (box.y1_px + box.y2_px) // 2
        return {
            "tl": (box.x1_px, box.y1_px),
            "tr": (box.x2_px, box.y1_px),
            "bl": (box.x1_px, box.y2_px),
            "br": (box.x2_px, box.y2_px),
            "left": (box.x1_px, mid_y),
            "right": (box.x2_px, mid_y),
            "top": (mid_x, box.y1_px),
            "bottom": (mid_x, box.y2_px),
        }

    def _box_handle_radius_px(self, box: BoxPrompt) -> int:
        box_w = max(1, box.x2_px - box.x1_px)
        box_h = max(1, box.y2_px - box.y1_px)
        min_dim = min(box_w, box_h)
        return max(2, min(6, min_dim // 6))

    def _point_in_box(self, box: BoxPrompt, x_px: int, y_px: int) -> bool:
        return box.x1_px <= x_px <= box.x2_px and box.y1_px <= y_px <= box.y2_px

    def _find_box_at_point(self, x_px: int, y_px: int) -> Optional[Tuple[int, BoxPrompt]]:
        frame_boxes = self.box_prompts_by_frame_obj.get(self.current_frame_idx, {})
        best_hit: Optional[Tuple[int, BoxPrompt]] = None
        best_area: Optional[int] = None
        for obj_id, box in frame_boxes.items():
            if not self._is_object_visible(obj_id):
                continue
            if not self._point_in_box(box, x_px, y_px):
                continue
            area = max(1, box.x2_px - box.x1_px) * max(1, box.y2_px - box.y1_px)
            if best_area is None or area < best_area:
                best_hit = (obj_id, box)
                best_area = area
        return best_hit

    def _build_box_from_drag(
        self,
        current_xy: Tuple[int, int],
    ) -> Optional[BoxPrompt]:
        if self._box_drag_mode is None or self._box_draw_start_xy is None:
            return None

        if self._box_drag_mode == "new":
            x1, y1 = self._box_draw_start_xy
            x2, y2 = current_xy
            if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
                return None
            return BoxPrompt(
                x1_px=min(x1, x2),
                y1_px=min(y1, y2),
                x2_px=max(x1, x2),
                y2_px=max(y1, y2),
            )

        ref_box = self._box_drag_reference_box
        if ref_box is None:
            return None
        x_px, y_px = current_xy

        if self._box_drag_mode == "move":
            return self._build_moved_box(ref_box, x_px, y_px)

        return self._build_resized_box(ref_box, self._box_drag_mode, x_px, y_px)

    def _build_resized_box(
        self,
        ref_box: BoxPrompt,
        drag_mode: str,
        x_px: int,
        y_px: int,
    ) -> Optional[BoxPrompt]:
        x1 = ref_box.x1_px
        y1 = ref_box.y1_px
        x2 = ref_box.x2_px
        y2 = ref_box.y2_px

        if "left" in drag_mode or drag_mode in {"tl", "bl"}:
            x1 = x_px
        if "right" in drag_mode or drag_mode in {"tr", "br"}:
            x2 = x_px
        if "top" in drag_mode or drag_mode in {"tl", "tr"}:
            y1 = y_px
        if "bottom" in drag_mode or drag_mode in {"bl", "br"}:
            y2 = y_px

        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
            return None
        return BoxPrompt(x1_px=x1, y1_px=y1, x2_px=x2, y2_px=y2)

    def _build_moved_box(
        self,
        ref_box: BoxPrompt,
        x_px: int,
        y_px: int,
    ) -> BoxPrompt:
        frame_size = self._get_current_frame_size()
        if frame_size is None:
            return ref_box
        width, height = frame_size

        box_w = ref_box.x2_px - ref_box.x1_px
        box_h = ref_box.y2_px - ref_box.y1_px
        off_x, off_y = self._box_drag_offset_xy
        new_x1 = x_px - off_x
        new_y1 = y_px - off_y
        new_x1 = max(0, min(width - 1 - box_w, new_x1))
        new_y1 = max(0, min(height - 1 - box_h, new_y1))
        return BoxPrompt(
            x1_px=int(new_x1),
            y1_px=int(new_y1),
            x2_px=int(new_x1 + box_w),
            y2_px=int(new_y1 + box_h),
        )

    def _update_box_rubber_band_for_box(
        self,
        box: Optional[BoxPrompt],
    ) -> None:
        if box is None:
            return
        start_ui = self._map_image_to_ui_xy(box.x1_px, box.y1_px)
        end_ui = self._map_image_to_ui_xy(box.x2_px, box.y2_px)
        if start_ui is None or end_ui is None:
            return
        rect = QRect(QPoint(*start_ui), QPoint(*end_ui)).normalized()
        self._box_rubber_band.setGeometry(rect)

    def _draw_box_outline(
        self,
        img: np.ndarray,
        box: BoxPrompt,
        color: Tuple[int, int, int],
        thickness: int,
        dashed: bool,
    ) -> None:
        if not dashed:
            cv2.rectangle(img, (box.x1_px, box.y1_px), (box.x2_px, box.y2_px), color, thickness)
            return

        dash_px = 10
        gap_px = 6
        self._draw_dashed_line(img, (box.x1_px, box.y1_px), (box.x2_px, box.y1_px), color, thickness, dash_px, gap_px)
        self._draw_dashed_line(img, (box.x2_px, box.y1_px), (box.x2_px, box.y2_px), color, thickness, dash_px, gap_px)
        self._draw_dashed_line(img, (box.x2_px, box.y2_px), (box.x1_px, box.y2_px), color, thickness, dash_px, gap_px)
        self._draw_dashed_line(img, (box.x1_px, box.y2_px), (box.x1_px, box.y1_px), color, thickness, dash_px, gap_px)

    def _draw_dashed_line(
        self,
        img: np.ndarray,
        start_xy: Tuple[int, int],
        end_xy: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int,
        dash_px: int,
        gap_px: int,
    ) -> None:
        x1, y1 = start_xy
        x2, y2 = end_xy
        length = max(abs(x2 - x1), abs(y2 - y1))
        if length <= 0:
            return

        step = max(1, dash_px + gap_px)
        for offset in range(0, length + 1, step):
            seg_start = offset / length
            seg_end = min(offset + dash_px, length) / length
            sx = int(round(x1 + (x2 - x1) * seg_start))
            sy = int(round(y1 + (y2 - y1) * seg_start))
            ex = int(round(x1 + (x2 - x1) * seg_end))
            ey = int(round(y1 + (y2 - y1) * seg_end))
            cv2.line(img, (sx, sy), (ex, ey), color, thickness)

    def _should_draw_box_handles(self, obj_id: int) -> bool:
        return (
            self._is_box_mode()
            and self.active_object_id == obj_id
            and self._selected_box_obj_id == obj_id
        )

    def _draw_box_handles(
        self,
        img: np.ndarray,
        box: BoxPrompt,
        color: Tuple[int, int, int],
    ) -> None:
        handle_radius = self._box_handle_radius_px(box)
        outline_color = (255, 255, 255)
        for cx, cy in self._get_box_handle_points(box).values():
            cv2.circle(img, (cx, cy), handle_radius + 1, outline_color, -1)
            cv2.circle(img, (cx, cy), handle_radius, color, -1)

    def _sync_selected_box_state(self) -> None:
        if self.active_object_id is None:
            self._selected_box_obj_id = None
            return
        frame_boxes = self.box_prompts_by_frame_obj.get(self.current_frame_idx, {})
        if self._selected_box_obj_id != self.active_object_id:
            self._selected_box_obj_id = None
            return
        if self.active_object_id not in frame_boxes:
            self._selected_box_obj_id = None

    def _find_object(self, obj_id: int) -> Optional[ObjectInfo]:
        for obj in self.objects:
            if obj.obj_id == obj_id:
                return obj
        return None

    def _map_ui_to_image_xy(self, ui_x: float, ui_y: float) -> Optional[Tuple[int, int]]:
        if not self.frame_paths:
            return None
        frame_size = self._get_current_frame_size()
        if frame_size is None:
            return None
        w, h = frame_size

        x_off, y_off = self._display_offset
        rel_x = ui_x - x_off
        rel_y = ui_y - y_off

        disp_w, disp_h = self._display_image_size
        if rel_x < 0 or rel_y < 0 or rel_x >= disp_w or rel_y >= disp_h:
            return None

        if self._display_scale <= 0:
            return None

        x_img = int(rel_x / self._display_scale)
        y_img = int(rel_y / self._display_scale)

        x_img = max(0, min(w - 1, x_img))
        y_img = max(0, min(h - 1, y_img))
        return x_img, y_img

    def _map_ui_to_image_xy_clamped(self, ui_x: float, ui_y: float) -> Optional[Tuple[int, int]]:
        if not self.frame_paths or self._display_scale <= 0:
            return None
        frame_size = self._get_current_frame_size()
        if frame_size is None:
            return None
        w, h = frame_size

        x_off, y_off = self._display_offset
        rel_x = ui_x - x_off
        rel_y = ui_y - y_off
        x_img = int(rel_x / self._display_scale)
        y_img = int(rel_y / self._display_scale)

        x_img = max(0, min(w - 1, x_img))
        y_img = max(0, min(h - 1, y_img))
        return x_img, y_img

    def _map_image_to_ui_xy(self, x_img: int, y_img: int) -> Optional[Tuple[int, int]]:
        if self._display_scale <= 0 or not self.frame_paths:
            return None
        frame_size = self._get_current_frame_size()
        if frame_size is None:
            return None
        w, h = frame_size
        x_img = max(0, min(w - 1, x_img))
        y_img = max(0, min(h - 1, y_img))

        x_off, y_off = self._display_offset
        x_ui = int(round(x_img * self._display_scale + x_off))
        y_ui = int(round(y_img * self._display_scale + y_off))
        return x_ui, y_ui

    def _compute_display_offset(
        self,
        label_w: int,
        label_h: int,
        scaled_w: int,
        scaled_h: int,
    ) -> Tuple[int, int]:
        pan_x, pan_y = self._pan_offset_ui
        x_off = int(round((label_w - scaled_w) / 2.0 + pan_x))
        y_off = int(round((label_h - scaled_h) / 2.0 + pan_y))
        return x_off, y_off

    def _clamp_pan_offset(
        self,
        label_w: Optional[int] = None,
        label_h: Optional[int] = None,
        scaled_w: Optional[int] = None,
        scaled_h: Optional[int] = None,
    ) -> None:
        if not self.frame_paths:
            self._pan_offset_ui = (0.0, 0.0)
            return
        frame_size = self._get_current_frame_size()
        if frame_size is None:
            self._pan_offset_ui = (0.0, 0.0)
            return
        frame_w, frame_h = frame_size
        label_w = label_w if label_w is not None else max(1, self.image_label.width())
        label_h = label_h if label_h is not None else max(1, self.image_label.height())
        fit_scale = min(label_w / frame_w, label_h / frame_h)
        scale = fit_scale * self._zoom_multiplier
        scaled_w = scaled_w if scaled_w is not None else max(1, int(round(frame_w * scale)))
        scaled_h = scaled_h if scaled_h is not None else max(1, int(round(frame_h * scale)))

        pan_x, pan_y = self._pan_offset_ui
        if scaled_w <= label_w:
            pan_x = 0.0
        else:
            centered_x = (label_w - scaled_w) / 2.0
            min_pan_x = (label_w - scaled_w) - centered_x
            max_pan_x = -centered_x
            pan_x = min(max_pan_x, max(min_pan_x, pan_x))

        if scaled_h <= label_h:
            pan_y = 0.0
        else:
            centered_y = (label_h - scaled_h) / 2.0
            min_pan_y = (label_h - scaled_h) - centered_y
            max_pan_y = -centered_y
            pan_y = min(max_pan_y, max(min_pan_y, pan_y))

        self._pan_offset_ui = (pan_x, pan_y)

    def _set_pan_for_anchor(
        self,
        anchor_xy: Tuple[int, int],
        ui_x: float,
        ui_y: float,
    ) -> None:
        frame_size = self._get_current_frame_size()
        if frame_size is None:
            return
        frame_w, frame_h = frame_size
        label_w = max(1, self.image_label.width())
        label_h = max(1, self.image_label.height())
        fit_scale = min(label_w / frame_w, label_h / frame_h)
        scale = fit_scale * self._zoom_multiplier
        scaled_w = max(1, int(round(frame_w * scale)))
        scaled_h = max(1, int(round(frame_h * scale)))
        centered_x = (label_w - scaled_w) / 2.0
        centered_y = (label_h - scaled_h) / 2.0
        anchor_x, anchor_y = anchor_xy
        pan_x = ui_x - anchor_x * scale - centered_x
        pan_y = ui_y - anchor_y * scale - centered_y
        self._pan_offset_ui = (pan_x, pan_y)
        self._clamp_pan_offset(label_w=label_w, label_h=label_h, scaled_w=scaled_w, scaled_h=scaled_h)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.frame_paths:
            self._clamp_pan_offset()
            self._render_current_frame()

    def _get_current_frame_size(self) -> Optional[Tuple[int, int]]:
        return self._get_frame_size(self.current_frame_idx)

    def _get_frame_size(self, frame_idx: int) -> Optional[Tuple[int, int]]:
        if not self.frame_paths or frame_idx < 0 or frame_idx >= len(self.frame_paths):
            return None
        frame = cv2.imread(str(self.frame_paths[frame_idx]))
        if frame is not None:
            h, w = frame.shape[:2]
            return w, h
        try:
            with PilImage.open(str(self.frame_paths[frame_idx])) as img:
                w, h = img.size
            return w, h
        except Exception:
            return None

    def _read_frame_bgr(self, frame_idx: int) -> Optional[np.ndarray]:
        if not self.frame_paths or frame_idx < 0 or frame_idx >= len(self.frame_paths):
            return None
        frame = cv2.imread(str(self.frame_paths[frame_idx]))
        if frame is not None:
            return frame
        try:
            with PilImage.open(str(self.frame_paths[frame_idx])) as img:
                rgb = img.convert("RGB")
            return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    def _color_for_obj(self, obj_id: int) -> Tuple[int, int, int]:
        palette = [
            (56, 56, 255),
            (56, 255, 56),
            (255, 56, 56),
            (56, 255, 255),
            (255, 56, 255),
            (255, 255, 56),
            (180, 80, 255),
            (80, 180, 255),
        ]
        return palette[(obj_id - 1) % len(palette)]

    def closeEvent(self, event):
        try:
            self._autosave_timer.stop()
            if self._sam_thread is not None:
                self._sam_thread.quit()
                self._sam_thread.wait()
        except Exception:
            pass
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    win = AnnotatorMainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
