#!/usr/bin/env python3
"""PySide6 surgical video annotation app using SAM3 as assistive annotator."""

from __future__ import annotations

import sys
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image as PilImage
from PySide6.QtCore import QObject, QPoint, QRect, QSize, Qt, QThread, Signal, Slot, QEventLoop, QMetaObject, Q_ARG
from PySide6.QtGui import QAction, QBrush, QColor, QImage, QKeySequence, QPixmap
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
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRubberBand,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from sam3.model.sam3_video_predictor import Sam3VideoPredictor
from tools.exporters.coco_export import CocoExporter, ObjectInfo as ExportObjectInfo
from tools.exporters.coco_export import PointPrompt as ExportPointPrompt
from tools.exporters.coco_export import SamFrameOutput as ExportSamFrameOutput

MAX_POINTS_PER_OBJECT = 6
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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


@dataclass
class PendingPropagationState:
    remaining_chunks: int
    next_seed_frame_idx: int
    n_frames: int
    total_chunks: int
    sample_points_per_object: int
    carryover_mode: str
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
    propagate_frame = Signal(str, int, object, int, int)
    propagate_done = Signal(str, int, int)
    task_failed = Signal(str, str, bool)
    task_finished = Signal(str)

    def __init__(self, checkpoint_path: Optional[str], bpe_path: Optional[str]) -> None:
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.bpe_path = bpe_path
        self.sam_adapter: Optional[Sam3Adapter] = None
        self._queue = deque()
        self._busy = False
        self._cancel_prefetch = False

    @Slot()
    def initialize(self) -> None:
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
        if priority:
            self._queue.appendleft((task_id, task_type, payload))
        else:
            self._queue.append((task_id, task_type, payload))
        if not self._busy:
            self._process_next()

    @Slot()
    def cancel_prefetch(self) -> None:
        self._cancel_prefetch = True

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
                self.task_finished.emit(task_id)
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
        composite = SamFrameOutput(obj_ids=[], masks=[], boxes_xywh_norm=[], scores=[])
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
            )
            composite = _merge_outputs_for_worker(composite, obj_output)
        self.segment_done.emit(task_id, frame_idx, composite)

    def _run_propagate(self, task_id: str, payload: object, is_prefetch: bool) -> None:
        data = payload or {}
        seed_frame_idx = int(data.get("seed_frame_idx", -1))
        n_frames = int(data.get("n_frames", 0))
        frame_paths = data.get("frame_paths", [])
        prompt_payload = data.get("prompt_payload", {})
        if seed_frame_idx < 0 or n_frames <= 1:
            self.task_failed.emit(task_id, "No forward frames available from current seed frame.", True)
            return
        abs_start = seed_frame_idx
        abs_end = min(abs_start + n_frames, len(frame_paths))
        actual_n = abs_end - abs_start
        if actual_n <= 1:
            self.task_failed.emit(task_id, "No forward frames available from current seed frame.", True)
            return

        imgs_pil = [PilImage.open(str(frame_paths[i])) for i in range(abs_start, abs_end)]
        self.sam_adapter.start_session(imgs_pil)
        for obj_id, p in prompt_payload.items():
            points_rel = p.get("points_rel", [])
            labels = p.get("labels", [])
            mask_input = p.get("mask_input")
            if not points_rel:
                continue
            self.sam_adapter.add_object_points(
                frame_idx=0,
                obj_id=int(obj_id),
                points_rel=points_rel,
                labels=labels,
                mask_input=mask_input,
            )

        session_outputs = self.sam_adapter.propagate_n_frames(
            start_frame_idx=0,
            max_frames=actual_n,
        )
        last_masked_frame_idx: Optional[int] = None
        for session_idx, output in session_outputs.items():
            if is_prefetch and self._cancel_prefetch:
                continue
            abs_frame_idx = abs_start + int(session_idx)
            self.propagate_frame.emit(task_id, abs_frame_idx, output, int(session_idx), actual_n)
            for mask in output.masks:
                if np.asarray(mask).any():
                    last_masked_frame_idx = abs_frame_idx
                    break

        if is_prefetch:
            self._cancel_prefetch = False

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
    )
    obj_to_idx = {obj_id: i for i, obj_id in enumerate(merged.obj_ids)}
    for i, obj_id in enumerate(new_output.obj_ids):
        if i >= len(new_output.masks) or i >= len(new_output.boxes_xywh_norm):
            continue
        score = new_output.scores[i] if i < len(new_output.scores) else 0.0
        if obj_id in obj_to_idx:
            idx = obj_to_idx[obj_id]
            merged.masks[idx] = new_output.masks[i]
            merged.boxes_xywh_norm[idx] = new_output.boxes_xywh_norm[i]
            merged.scores[idx] = score
        else:
            merged.obj_ids.append(obj_id)
            merged.masks.append(new_output.masks[i])
            merged.boxes_xywh_norm.append(new_output.boxes_xywh_norm[i])
            merged.scores.append(score)
            obj_to_idx[obj_id] = len(merged.obj_ids) - 1
    return merged


class ClickableImageLabel(QLabel):
    def __init__(self, parent: "AnnotatorMainWindow"):
        super().__init__()
        self.parent_window = parent
        self.setAlignment(Qt.AlignCenter)
        # Keep layout stable while allowing full fit-to-panel rendering.
        self.setMinimumSize(1, 1)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setScaledContents(False)
        self.setStyleSheet("background-color: #111; border: 1px solid #444; color: #ddd;")
        self.setMouseTracking(True)

    def sizeHint(self):
        return QSize(960, 540)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.parent_window.on_pan_press(event.position().x(), event.position().y())
            return
        if event.button() != Qt.LeftButton:
            return
        self.parent_window.on_image_press(event.position().x(), event.position().y())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.RightButton:
            self.parent_window.on_pan_drag(event.position().x(), event.position().y())
            return
        self.parent_window.on_image_hover(event.position().x(), event.position().y())
        self.parent_window.on_image_drag(event.position().x(), event.position().y())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.parent_window.on_pan_release()
            return
        if event.button() != Qt.LeftButton:
            return
        self.parent_window.on_image_release(event.position().x(), event.position().y())

    def leaveEvent(self, _event):
        self.parent_window.on_image_hover(None, None)

    def wheelEvent(self, event):
        self.parent_window.on_image_wheel(
            event.position().x(),
            event.position().y(),
            event.angleDelta().y(),
        )
        event.accept()


class Sam3Adapter:
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        bpe_path: Optional[str] = None,
    ) -> None:
        self.predictor = Sam3VideoPredictor(
            checkpoint_path=checkpoint_path,
            bpe_path=bpe_path,
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

    def _parse_output(self, outputs: dict) -> SamFrameOutput:
        obj_ids = [int(x) for x in outputs.get("out_obj_ids", [])]
        masks = [np.asarray(m).astype(bool) for m in outputs.get("out_binary_masks", [])]
        boxes = [tuple(map(float, b)) for b in outputs.get("out_boxes_xywh", [])]
        scores = [float(x) for x in outputs.get("out_probs", [])]
        return SamFrameOutput(obj_ids=obj_ids, masks=masks, boxes_xywh_norm=boxes, scores=scores)


class AnnotatorMainWindow(QMainWindow):
    task_requested = Signal(str, str, object, bool)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM3 Surgical Video Annotator")
        self.resize(1600, 900)

        self.sam_adapter: Optional[Sam3Adapter] = None
        self._checkpoint_path: Optional[str] = None
        self._bpe_path: Optional[str] = None
        self.frame_paths: List[Path] = []
        self.current_frame_idx: int = 0
        self.image_dir: Optional[Path] = None

        self.objects: List[ObjectInfo] = []
        self.active_object_id: Optional[int] = None
        self.next_obj_id: int = 1

        self.prompts_by_frame_obj: Dict[int, Dict[int, List[PointPrompt]]] = {}
        self.box_prompts_by_frame_obj: Dict[int, Dict[int, BoxPrompt]] = {}
        self.box_locked_by_frame_obj: Dict[int, Dict[int, bool]] = {}
        self.outputs_by_frame: Dict[int, SamFrameOutput] = {}
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
        self._box_draw_start_xy: Optional[Tuple[int, int]] = None
        self._box_draw_start_ui_xy: Optional[Tuple[int, int]] = None
        self._box_edit_active: bool = False
        self._box_drag_mode: Optional[str] = None
        self._box_drag_reference_box: Optional[BoxPrompt] = None
        self._box_drag_offset_xy: Tuple[int, int] = (0, 0)
        self._selected_box_obj_id: Optional[int] = None
        self.show_advanced_controls: bool = False
        self.show_prompts: bool = True
        self.show_segmentations: bool = True
        self.show_boxes: bool = True
        self.auto_propagate_next: bool = False
        self.translate_prompts_on_propagation: bool = True
        self.segmentation_opacity: float = 0.6
        self.box_line_thickness: int = 2

        self._setup_ui()

    def _setup_ui(self) -> None:
        load_action = QAction("Load Frame Directory", self)
        load_action.triggered.connect(self.load_frame_directory)
        self.menuBar().addAction(load_action)

        export_action = QAction("Export COCO", self)
        export_action.triggered.connect(self.export_annotations)
        self.menuBar().addAction(export_action)

        root = QWidget()
        root_layout = QHBoxLayout(root)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        left_placeholder = QWidget()
        left_placeholder.setMinimumWidth(200)

        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)

        nav_row = QHBoxLayout()
        self.prev_btn = QPushButton("Prev")
        self.prev_btn.clicked.connect(self.go_prev_frame)
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.go_next_frame)
        self.fit_view_btn = QPushButton("Fit to Screen")
        self.fit_view_btn.clicked.connect(self.fit_current_frame_to_view)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(1)
        self.frame_slider.setMaximum(1)
        self.frame_slider.setValue(1)
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        self.frame_jump_spin = QSpinBox()
        self.frame_jump_spin.setMinimum(1)
        self.frame_jump_spin.setMaximum(1)
        self.frame_jump_spin.setValue(1)
        self.frame_jump_spin.valueChanged.connect(self._on_frame_jump_changed)
        self.frame_label = QLabel("Frame: -/-")
        nav_row.addWidget(self.prev_btn)
        nav_row.addWidget(self.next_btn)
        nav_row.addWidget(self.fit_view_btn)
        nav_row.addWidget(self.frame_slider, 1)
        nav_row.addWidget(QLabel("Go to:"))
        nav_row.addWidget(self.frame_jump_spin)
        nav_row.addWidget(self.frame_label)
        center_layout.addLayout(nav_row)

        self.image_label = ClickableImageLabel(self)
        center_layout.addWidget(self.image_label)
        self._box_rubber_band = QRubberBand(QRubberBand.Rectangle, self.image_label)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        control_tabs = QTabWidget()
        right_layout.addWidget(control_tabs)

        prompt_tab = QWidget()
        prompt_layout = QVBoxLayout(prompt_tab)
        processing_tab = QWidget()
        processing_layout = QVBoxLayout(processing_tab)

        model_form = QFormLayout()
        self.checkpoint_combo = QComboBox()
        self.checkpoint_combo.addItem("Use default HF checkpoint")
        self.checkpoint_combo.setEditable(True)
        self.checkpoint_combo.lineEdit().setPlaceholderText("Optional local checkpoint path")
        model_form.addRow("Checkpoint", self.checkpoint_combo)

        self.bpe_combo = QComboBox()
        self.bpe_combo.addItem("Use default BPE")
        self.bpe_combo.setEditable(True)
        self.bpe_combo.lineEdit().setPlaceholderText("Optional local BPE path")
        model_form.addRow("BPE", self.bpe_combo)
        prompt_layout.addLayout(model_form)

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

        self.auto_propagate_next_check = QCheckBox("Auto Propagate Next Frame")
        self.auto_propagate_next_check.toggled.connect(self._on_auto_propagate_next_toggled)
        processing_layout.addWidget(self.auto_propagate_next_check)

        self.translate_prompts_check = QCheckBox("Translate Point Prompts on Propagation")
        self.translate_prompts_check.setChecked(self.translate_prompts_on_propagation)
        self.translate_prompts_check.toggled.connect(self._on_translate_prompts_toggled)
        processing_layout.addWidget(self.translate_prompts_check)

        self.prompt_mode_label = QLabel("Advanced Prompt Mode")
        prompt_layout.addWidget(self.prompt_mode_label)
        self.prompt_mode_combo = QComboBox()
        self.prompt_mode_combo.addItems(["Positive (+)", "Negative (-)", "Box (drag)"])
        self.prompt_mode_combo.setCurrentIndex(2)
        self.prompt_mode_combo.currentIndexChanged.connect(self._on_prompt_mode_changed)
        prompt_layout.addWidget(self.prompt_mode_combo)

        self.annotation_list_label = QLabel("Current Frame Annotations")
        prompt_layout.addWidget(self.annotation_list_label)
        self.lock_current_box_check = QCheckBox("Lock Current Box")
        self.lock_current_box_check.toggled.connect(self._on_lock_current_box_toggled)
        prompt_layout.addWidget(self.lock_current_box_check)
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

        self.advanced_controls_check = QCheckBox("Show Advanced Prompt Controls")
        self.advanced_controls_check.toggled.connect(self._on_advanced_controls_toggled)
        prompt_layout.addWidget(self.advanced_controls_check)

        processing_layout.addWidget(QLabel("View"))
        self.show_prompts_check = QCheckBox("Show Prompts")
        self.show_prompts_check.setChecked(self.show_prompts)
        self.show_prompts_check.toggled.connect(self._on_view_settings_changed)
        processing_layout.addWidget(self.show_prompts_check)

        self.show_segmentations_check = QCheckBox("Show Segmentations")
        self.show_segmentations_check.setChecked(self.show_segmentations)
        self.show_segmentations_check.toggled.connect(self._on_view_settings_changed)
        processing_layout.addWidget(self.show_segmentations_check)

        self.show_boxes_check = QCheckBox("Show Boxes")
        self.show_boxes_check.setChecked(self.show_boxes)
        self.show_boxes_check.toggled.connect(self._on_view_settings_changed)
        processing_layout.addWidget(self.show_boxes_check)

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
        processing_layout.addLayout(view_form)

        propagate_row = QHBoxLayout()
        self.propagate_btn = QPushButton("Propagate")
        self.propagate_btn.clicked.connect(self.propagate_next_frame)
        self.n_propagate_spin = QSpinBox()
        self.n_propagate_spin.setMinimum(2)
        self.n_propagate_spin.setMaximum(9999)
        self.n_propagate_spin.setValue(2)
        self.n_propagate_spin.setToolTip("Number of frames to propagate (including current frame)")
        self.chunks_spin = QSpinBox()
        self.chunks_spin.setMinimum(1)
        self.chunks_spin.setMaximum(9999)
        self.chunks_spin.setValue(1)
        self.chunks_spin.setToolTip("Number of propagation chunks to run")
        self.sample_points_spin = QSpinBox()
        self.sample_points_spin.setMinimum(1)
        self.sample_points_spin.setMaximum(8)
        self.sample_points_spin.setValue(4)
        self.sample_points_spin.setToolTip("Carryover sampled points per object (from last masks)")
        self.carryover_mode_combo = QComboBox()
        self.carryover_mode_combo.addItems(["Sample Points", "Mask AABB Box"])
        self.carryover_mode_combo.setToolTip("Carryover prompt mode for chunks after the first")
        self.pause_between_chunks_check = QCheckBox("Pause Between Chunks")
        self.pause_between_chunks_check.setChecked(True)
        propagate_row.addWidget(self.propagate_btn)
        propagate_row.addWidget(QLabel("N frames:"))
        propagate_row.addWidget(self.n_propagate_spin)
        propagate_row.addWidget(QLabel("Chunks:"))
        propagate_row.addWidget(self.chunks_spin)
        propagate_row.addWidget(QLabel("Sample pts:"))
        propagate_row.addWidget(self.sample_points_spin)
        propagate_row.addWidget(QLabel("Carryover:"))
        propagate_row.addWidget(self.carryover_mode_combo)
        propagate_row.addWidget(self.pause_between_chunks_check)
        processing_layout.addLayout(propagate_row)

        self.mode_label = QLabel("Mode: Prompt")
        processing_layout.addWidget(self.mode_label)
        prompt_layout.addStretch(1)
        processing_layout.addStretch(1)

        control_tabs.addTab(prompt_tab, "Prompting")
        control_tabs.addTab(processing_tab, "Segment / Propagate")

        self._advanced_widgets = [
            self.prompt_mode_label,
            self.prompt_mode_combo,
            self.segment_btn,
            self.show_prompts_check,
        ]
        self._set_advanced_controls_visible(False)
        self.mode_label.setText("Mode: Box Annotation")

        splitter.addWidget(left_placeholder)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([200, 1100, 500])

        self.setCentralWidget(root)
        self._setup_shortcuts()
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

    def _on_translate_prompts_toggled(self, checked: bool) -> None:
        self.translate_prompts_on_propagation = checked

    def _on_lock_current_box_toggled(self, checked: bool) -> None:
        if self.active_object_id is None:
            self.lock_current_box_check.blockSignals(True)
            self.lock_current_box_check.setChecked(False)
            self.lock_current_box_check.blockSignals(False)
            return
        frame_boxes = self.box_prompts_by_frame_obj.get(self.current_frame_idx, {})
        if self.active_object_id not in frame_boxes:
            self.lock_current_box_check.blockSignals(True)
            self.lock_current_box_check.setChecked(False)
            self.lock_current_box_check.blockSignals(False)
            return
        self._set_current_box_locked(self.current_frame_idx, self.active_object_id, checked)
        self.refresh_point_list()
        self._render_current_frame()

    def _on_advanced_controls_toggled(self, checked: bool) -> None:
        self.show_advanced_controls = checked
        self._set_advanced_controls_visible(checked)
        self.mode_label.setText("Mode: Prompt" if self._use_point_prompt_mode() else "Mode: Box Annotation")
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

        next_action = QAction(self)
        next_action.setShortcut(QKeySequence(Qt.Key_Right))
        next_action.triggered.connect(self.go_next_frame_shortcut)
        self.addAction(next_action)

        propagate_action = QAction(self)
        propagate_action.setShortcut(QKeySequence("P"))
        propagate_action.triggered.connect(self.propagate_next_frame)
        self.addAction(propagate_action)

    def _refresh_object_list_visuals(self) -> None:
        active_label = "Active: None"
        for i in range(self.object_list.count()):
            item = self.object_list.item(i)
            obj_id = int(item.data(Qt.UserRole))
            if obj_id == self.active_object_id:
                item.setBackground(QBrush(QColor("#1f4f7a")))
                item.setForeground(QBrush(QColor("#ffffff")))
                active_label = f"Active: {item.text()}"
            else:
                item.setBackground(QBrush())
                item.setForeground(QBrush())
        self.active_object_label.setText(active_label)

    def _on_view_settings_changed(self, _value=None) -> None:
        self.show_prompts = self.show_prompts_check.isChecked()
        self.show_segmentations = self.show_segmentations_check.isChecked()
        self.show_boxes = self.show_boxes_check.isChecked()
        self.segmentation_opacity = float(self.segmentation_opacity_spin.value())
        self.box_line_thickness = int(self.box_line_thickness_spin.value())
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

    def _note_prompt_change_and_prefetch(self) -> None:
        self._prefetch_prompt_version += 1
        self._prefetch_cached_frame_idx = None
        self._prefetch_cached_seed_idx = None
        self._prefetch_cached_version = None
        self._schedule_prefetch_for_next_frame()

    def _has_valid_prefetch(self, target_frame_idx: int) -> bool:
        return (
            self._prefetch_cached_frame_idx == target_frame_idx
            and self._prefetch_cached_seed_idx == target_frame_idx - 1
            and self._prefetch_cached_version == self._prefetch_prompt_version
        )

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
        if self._propagation_busy:
            return
        if not self.frame_paths or self.current_frame_idx >= len(self.frame_paths) - 1:
            return
        if self._has_valid_prefetch(self.current_frame_idx + 1):
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
            sample_points_per_object=self.sample_points_spin.value(),
            carryover_mode=self.carryover_mode_combo.currentText(),
            enabled_obj_ids=enabled_obj_ids,
        )
        if prompt_payload is None:
            return

        target_frame_idx = self.current_frame_idx + 1
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
        self.point_list.setEnabled(enabled)
        self.remove_point_btn.setEnabled(enabled)
        self.clear_points_btn.setEnabled(enabled)
        self.lock_current_box_check.setEnabled(enabled)
        self.prompt_mode_combo.setEnabled(enabled)
        self.advanced_controls_check.setEnabled(enabled)
        self.prev_btn.setEnabled(enabled)
        self.next_btn.setEnabled(enabled)
        self.fit_view_btn.setEnabled(enabled)
        self.frame_slider.setEnabled(enabled)
        self.frame_jump_spin.setEnabled(enabled)

    def _sync_current_box_lock_check(self) -> None:
        has_box = False
        checked = False
        if self.active_object_id is not None:
            frame_boxes = self.box_prompts_by_frame_obj.get(self.current_frame_idx, {})
            has_box = self.active_object_id in frame_boxes
            checked = self._is_current_box_locked(self.current_frame_idx, self.active_object_id)
        self.lock_current_box_check.blockSignals(True)
        self.lock_current_box_check.setEnabled(has_box)
        self.lock_current_box_check.setChecked(checked if has_box else False)
        self.lock_current_box_check.blockSignals(False)

    def _use_point_prompt_mode(self) -> bool:
        return self.show_advanced_controls and self.prompt_mode_combo.currentIndex() != 2

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
        self._bpe_path = self._read_optional_combo_path(self.bpe_combo)

        if not self._initialize_sam_worker(show_errors=True):
            return
        self._enqueue_sam_task("close_session", {}, priority=True)

        self.image_dir = dir_path
        self.frame_paths = frame_paths
        self.current_frame_idx = 0
        self.objects.clear()
        self.object_list.clear()
        self.next_obj_id = 1
        self.active_object_id = None
        self.prompts_by_frame_obj.clear()
        self.box_prompts_by_frame_obj.clear()
        self.box_locked_by_frame_obj.clear()
        self.outputs_by_frame.clear()
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
        self.show_advanced_controls = False
        self.advanced_controls_check.setChecked(False)
        self.prompt_mode_combo.setCurrentIndex(2)
        self._set_advanced_controls_visible(False)
        self._sync_frame_navigation_controls()
        self._refresh_object_list_visuals()
        self._sync_current_box_lock_check()

        self._render_current_frame()
        self._set_status(f"Loaded {len(self.frame_paths)} frames from {dir_path}", progress=1, total=1)
        self._reset_prefetch_state()

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

        obj_id = self.next_obj_id
        self.next_obj_id += 1
        color = self._color_for_obj(obj_id)
        obj = ObjectInfo(obj_id=obj_id, name=name, color_bgr=color)
        self.objects.append(obj)

        item = QListWidgetItem(f"{obj.name} (id={obj.obj_id})")
        item.setData(Qt.UserRole, obj.obj_id)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        self.object_list.addItem(item)
        self.object_list.setCurrentItem(item)
        self._refresh_object_list_visuals()

    def remove_active_object(self) -> None:
        if self.active_object_id is None:
            return

        obj_id = self.active_object_id
        self.objects = [o for o in self.objects if o.obj_id != obj_id]

        for frame_map in self.prompts_by_frame_obj.values():
            frame_map.pop(obj_id, None)
        for frame_map in self.box_prompts_by_frame_obj.values():
            frame_map.pop(obj_id, None)
        for frame_map in self.box_locked_by_frame_obj.values():
            frame_map.pop(obj_id, None)

        for frame_idx, out in list(self.outputs_by_frame.items()):
            keep_idx = [i for i, oid in enumerate(out.obj_ids) if oid != obj_id]
            if len(keep_idx) == len(out.obj_ids):
                continue
            self.outputs_by_frame[frame_idx] = SamFrameOutput(
                obj_ids=[out.obj_ids[i] for i in keep_idx],
                masks=[out.masks[i] for i in keep_idx],
                boxes_xywh_norm=[out.boxes_xywh_norm[i] for i in keep_idx],
                scores=[out.scores[i] for i in keep_idx],
            )

        for i in range(self.object_list.count()):
            item = self.object_list.item(i)
            if int(item.data(Qt.UserRole)) == obj_id:
                self.object_list.takeItem(i)
                break

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

        if not self._ensure_sam_initialized():
            return False
        if self._has_valid_prefetch(min(len(self.frame_paths) - 1, self.current_frame_idx + 1)):
            return True

        prompt_payload = self._build_seed_prompts(
            seed_frame_idx=self.current_frame_idx,
            use_carryover_sampling=False,
            sample_points_per_object=self.sample_points_spin.value(),
            carryover_mode=self.carryover_mode_combo.currentText(),
            enabled_obj_ids=enabled_obj_ids,
        )
        if prompt_payload is None:
            return False

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
                "target_frame_idx": self.current_frame_idx + 1,
                "seed_frame_idx": self.current_frame_idx,
                "version": self._prefetch_prompt_version,
            }
            self._prefetch_busy = True
            self._prefetch_seed_frame_idx = self.current_frame_idx
            self._prefetch_target_frame_idx = self.current_frame_idx + 1
            self._prefetch_active_version = self._prefetch_prompt_version
            self._set_status("Prefetching next frame...", progress=0, total=1)
        if self._prefetch_busy:
            task_id = f"prefetch-wait:{self.current_frame_idx}->{self.current_frame_idx + 1}:{self._prefetch_prompt_version}"
            self._sam_task_contexts[task_id] = {"kind": "prefetch_wait"}
            self._sam_waiting[task_id] = QEventLoop()
            self._sam_waiting[task_id].exec()
            self._sam_waiting.pop(task_id, None)
            self._sam_task_contexts.pop(task_id, None)
        return self._has_valid_prefetch(min(len(self.frame_paths) - 1, self.current_frame_idx + 1))

    def _set_current_frame_idx(self, frame_idx: int) -> None:
        if not self.frame_paths:
            return
        self.current_frame_idx = max(0, min(len(self.frame_paths) - 1, frame_idx))
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

    def _on_frame_slider_changed(self, value: int) -> None:
        if not self.frame_paths:
            return
        self._set_current_frame_idx(value - 1)

    def _on_frame_jump_changed(self, value: int) -> None:
        if not self.frame_paths:
            return
        self._set_current_frame_idx(value - 1)

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

        points.append(PointPrompt(x_px=x_px, y_px=y_px, is_positive=self.current_prompt_positive))
        self.refresh_point_list()

        if self.segment_mode:
            self.segment_current_frame()
        else:
            self._note_prompt_change_and_prefetch()
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

        frame_boxes = self.box_prompts_by_frame_obj.setdefault(self.current_frame_idx, {})
        frame_boxes[self.active_object_id] = box
        self._selected_box_obj_id = self.active_object_id
        self.refresh_point_list()
        is_locked = self._is_current_box_locked(self.current_frame_idx, self.active_object_id)
        self._sync_current_box_lock_check()
        if not is_locked:
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
        worker = SamWorker(self._checkpoint_path, self._bpe_path)
        thread = QThread()
        worker.moveToThread(thread)
        worker.initialized.connect(self._on_sam_worker_initialized)
        worker.segment_done.connect(self._on_sam_segment_done)
        worker.propagate_frame.connect(self._on_sam_propagate_frame)
        worker.propagate_done.connect(self._on_sam_propagate_done)
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
        if not obj_ids:
            if show_no_prompts:
                QMessageBox.information(self, "No prompts", "Add point or box prompts for at least one object on this frame.")
            return False
        if not self._ensure_sam_initialized():
            return False
        payload = self._build_segment_payload(self.current_frame_idx, obj_ids, show_no_prompts)
        if payload is None:
            return False
        task_id = self._enqueue_sam_task(
            "segment",
            {
                "frame_idx": self.current_frame_idx,
                "frame_path": self.frame_paths[self.current_frame_idx],
                "payload": payload,
            },
        )
        result = self._wait_for_sam_task(task_id)
        return bool(result)

    def propagate_next_frame(self) -> None:
        if not self._ensure_sam_initialized():
            return
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
        if self._pending_propagation is None:
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
            self._pending_propagation = PendingPropagationState(
                remaining_chunks=self.chunks_spin.value(),
                next_seed_frame_idx=self.current_frame_idx,
                n_frames=self.n_propagate_spin.value(),
                total_chunks=self.chunks_spin.value(),
                sample_points_per_object=self.sample_points_spin.value(),
                carryover_mode=self.carryover_mode_combo.currentText(),
            )
            self._set_status(
                f"Propagation starting: 0/{self._pending_propagation.total_chunks} chunks.",
                progress=0,
                total=self._pending_propagation.total_chunks,
            )
            self._propagation_enabled_obj_ids = enabled_obj_ids
            self._propagation_view_frame_idx = self.current_frame_idx
        elif self.pause_between_chunks_check.isChecked():
            confirm = QMessageBox.question(
                self,
                "Continue propagation?",
                "Continue to next chunk from current masks?\nChoose No to stop and refine prompts/masks first.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if confirm != QMessageBox.Yes:
                self._clear_pending_propagation_state()
                self._set_status("Propagation stopped for manual correction.", progress=0, total=1)
                return
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
        sample_points_per_object: int,
        carryover_mode: str,
        enabled_obj_ids: set[int],
    ) -> Optional[Dict[int, Dict[str, object]]]:
        frame_prompts = self.prompts_by_frame_obj.get(seed_frame_idx, {})
        frame_boxes = self.box_prompts_by_frame_obj.get(seed_frame_idx, {})

        frame_size = self._get_frame_size(seed_frame_idx)
        if frame_size is None:
            QMessageBox.critical(self, "Propagation setup failed", "Failed to load prompt source frame.")
            return None
        w, h = frame_size

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
        elif use_carryover_sampling:
            if carryover_mode == "Mask AABB Box":
                sampled_boxes = self._sample_boxes_from_seed_masks(seed_frame_idx, h, w)
                for obj_id, box in sampled_boxes.items():
                    if (
                        obj_id in enabled_obj_ids
                        and obj_id not in prompts_to_apply
                        and obj_id not in boxes_to_apply
                    ):
                        boxes_to_apply[obj_id] = box
            else:
                sampled = self._sample_prompts_from_seed_masks(seed_frame_idx, sample_points_per_object, h, w)
                for obj_id, points in sampled.items():
                    if (
                        points
                        and obj_id in enabled_obj_ids
                        and obj_id not in prompts_to_apply
                        and obj_id not in boxes_to_apply
                    ):
                        prompts_to_apply[obj_id] = points

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
            return None

        obj_ids = set(prompts_to_apply.keys()) | set(boxes_to_apply.keys())
        obj_ids |= set(masks_to_apply.keys())
        if not obj_ids:
            QMessageBox.information(
                self,
                "No prompts",
                "No prompts available for this chunk seed. Add prompts on this frame or create a seed mask first.",
            )
            return None

        payload: Dict[int, Dict[str, object]] = {}
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

    def _start_propagation_chunk_async(
        self,
        *,
        enabled_obj_ids: set[int],
        state: PendingPropagationState,
    ) -> None:
        chunk_idx = state.completed_chunks + 1
        seed_frame_idx = state.next_seed_frame_idx
        use_carryover_sampling = chunk_idx > 1
        prompt_payload = self._build_seed_prompts(
            seed_frame_idx=seed_frame_idx,
            use_carryover_sampling=use_carryover_sampling,
            sample_points_per_object=state.sample_points_per_object,
            carryover_mode=state.carryover_mode,
            enabled_obj_ids=enabled_obj_ids,
        )
        if prompt_payload is None:
            self._clear_pending_propagation_state()
            return

        self._propagation_busy = True
        self._propagation_active_chunk_idx = chunk_idx
        self._propagation_active_seed_frame_idx = seed_frame_idx
        self._set_propagation_ui_enabled(False)
        self._set_status(
            f"Chunk {chunk_idx}/{state.total_chunks} running...",
            progress=state.completed_chunks,
            total=state.total_chunks,
        )

        task_id = self._enqueue_sam_task(
            "propagate",
            {
                "seed_frame_idx": seed_frame_idx,
                "n_frames": state.n_frames,
                "frame_paths": self.frame_paths,
                "prompt_payload": prompt_payload,
            },
        )
        self._sam_task_contexts[task_id] = {
            "kind": "manual",
            "seed_frame_idx": seed_frame_idx,
        }
        self._propagation_task_id = task_id

    def _handle_propagation_frame(self, abs_frame_idx: int, output: SamFrameOutput, session_idx: int, total_frames: int) -> None:
        existing_output = self.outputs_by_frame.get(
            abs_frame_idx,
            SamFrameOutput(obj_ids=[], masks=[], boxes_xywh_norm=[], scores=[]),
        )
        merged_output = self._merge_frame_outputs(existing_output, output)
        self.outputs_by_frame[abs_frame_idx] = merged_output
        seed_frame_idx = getattr(self, "_propagation_active_seed_frame_idx", None)
        if seed_frame_idx is not None and abs_frame_idx != seed_frame_idx:
            self._sync_canonical_boxes_from_output(abs_frame_idx, output, preserve_locked=False)
            self._carry_prompts_forward(abs_frame_idx - 1, abs_frame_idx, output.obj_ids)
        if abs_frame_idx == self.current_frame_idx:
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
            SamFrameOutput(obj_ids=[], masks=[], boxes_xywh_norm=[], scores=[]),
        )
        merged = self._merge_frame_outputs(existing_output, composite)
        if composite.obj_ids:
            self.outputs_by_frame[frame_idx] = merged
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

    def _on_sam_propagate_frame(self, task_id: str, abs_frame_idx: int, output: SamFrameOutput, session_idx: int, total_frames: int) -> None:
        context = self._sam_task_contexts.get(task_id, {})
        kind = context.get("kind")
        if kind == "prefetch":
            if self._prefetch_cancel_requested:
                return
            version = context.get("version")
            target_frame_idx = context.get("target_frame_idx")
            seed_frame_idx = context.get("seed_frame_idx")
            if version != self._prefetch_prompt_version or abs_frame_idx != target_frame_idx:
                return
            existing_output = self.outputs_by_frame.get(
                abs_frame_idx,
                SamFrameOutput(obj_ids=[], masks=[], boxes_xywh_norm=[], scores=[]),
            )
            merged_output = self._merge_frame_outputs(existing_output, output)
            self.outputs_by_frame[abs_frame_idx] = merged_output
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

        existing_output = self.outputs_by_frame.get(
            abs_frame_idx,
            SamFrameOutput(obj_ids=[], masks=[], boxes_xywh_norm=[], scores=[]),
        )
        merged_output = self._merge_frame_outputs(existing_output, output)
        self.outputs_by_frame[abs_frame_idx] = merged_output
        seed_frame_idx = context.get("seed_frame_idx")
        if seed_frame_idx is not None and abs_frame_idx != seed_frame_idx:
            self._sync_canonical_boxes_from_output(abs_frame_idx, output, preserve_locked=False)
            self._carry_prompts_forward(abs_frame_idx - 1, abs_frame_idx, output.obj_ids)
        if abs_frame_idx == self.current_frame_idx:
            self._render_current_frame()
        state = self._pending_propagation
        if kind == "manual" and state is not None:
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
        self._set_status(
            f"Chunk {state.completed_chunks}/{state.total_chunks} complete.",
            progress=state.completed_chunks,
            total=state.total_chunks,
        )

        pause_between_chunks = self.pause_between_chunks_check.isChecked()
        if pause_between_chunks and state.remaining_chunks > 0:
            chunk_span = f"{chunk_seed_frame_idx + 1}-{chunk_last_frame_idx + 1}"
            if self._propagation_view_frame_idx is not None:
                self._set_current_frame_idx(self._propagation_view_frame_idx)
            self.propagate_btn.setText("Continue Propagate")
            self._set_status(
                f"Chunk {state.completed_chunks}/{state.total_chunks} complete over frames {chunk_span}. Review and click Continue Propagate.",
                progress=state.completed_chunks,
                total=state.total_chunks,
            )
            self._propagation_continue_after_chunk = False
            self._propagation_busy = False
            self._set_propagation_ui_enabled(True)
            return

        if state.remaining_chunks == 0:
            if self._propagation_view_frame_idx is not None:
                self._set_current_frame_idx(self._propagation_view_frame_idx)
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
        if is_warning:
            QMessageBox.warning(self, "Propagation stopped", message)
        else:
            QMessageBox.critical(self, "Propagation failed", message)
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
            SamFrameOutput(obj_ids=[], masks=[], boxes_xywh_norm=[], scores=[]),
        )
        merged_output = self._merge_frame_outputs(existing_output, output)
        self.outputs_by_frame[abs_frame_idx] = merged_output
        self._sync_canonical_boxes_from_output(abs_frame_idx, output, preserve_locked=False)
        self._carry_prompts_forward(abs_frame_idx - 1, abs_frame_idx, output.obj_ids)
        self._prefetch_cached_frame_idx = abs_frame_idx
        self._prefetch_cached_seed_idx = self._prefetch_seed_frame_idx
        self._prefetch_cached_version = self._prefetch_active_version
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

        pause_between_chunks = self.pause_between_chunks_check.isChecked()
        if pause_between_chunks and state.remaining_chunks > 0:
            chunk_span = f"{chunk_seed_frame_idx + 1}-{chunk_last_frame_idx + 1}"
            if self._propagation_view_frame_idx is not None:
                self._set_current_frame_idx(self._propagation_view_frame_idx)
            self.propagate_btn.setText("Continue Propagate")
            self._set_status(
                f"Chunk {chunk_idx}/{state.total_chunks} complete over frames {chunk_span}. Review and click Continue Propagate.",
                progress=state.completed_chunks,
                total=state.total_chunks,
            )
            self._propagation_continue_after_chunk = False
            return

        if state.remaining_chunks == 0:
            if self._propagation_view_frame_idx is not None:
                self._set_current_frame_idx(self._propagation_view_frame_idx)
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
        sample_points_per_object: int,
        carryover_mode: str,
        enabled_obj_ids: set[int],
    ) -> Optional[Tuple[int, int]]:
        if self._prefetch_busy:
            self._cancel_prefetch(restart=False)
        prompt_payload = self._build_seed_prompts(
            seed_frame_idx=seed_frame_idx,
            use_carryover_sampling=use_carryover_sampling,
            sample_points_per_object=sample_points_per_object,
            carryover_mode=carryover_mode,
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
            return

        self.outputs_by_frame[frame_idx] = SamFrameOutput(
            obj_ids=[output.obj_ids[i] for i in keep_idx],
            masks=[output.masks[i] for i in keep_idx],
            boxes_xywh_norm=[output.boxes_xywh_norm[i] for i in keep_idx],
            scores=[output.scores[i] for i in keep_idx],
        )

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
        )
        obj_to_idx = {obj_id: i for i, obj_id in enumerate(merged.obj_ids)}

        for i, obj_id in enumerate(new_output.obj_ids):
            if i >= len(new_output.masks) or i >= len(new_output.boxes_xywh_norm):
                continue
            score = new_output.scores[i] if i < len(new_output.scores) else 0.0
            if obj_id in obj_to_idx:
                idx = obj_to_idx[obj_id]
                merged.masks[idx] = new_output.masks[i]
                merged.boxes_xywh_norm[idx] = new_output.boxes_xywh_norm[i]
                merged.scores[idx] = score
            else:
                merged.obj_ids.append(obj_id)
                merged.masks.append(new_output.masks[i])
                merged.boxes_xywh_norm.append(new_output.boxes_xywh_norm[i])
                merged.scores.append(score)
                obj_to_idx[obj_id] = len(merged.obj_ids) - 1

        return merged

    def _apply_prompts_for_seed(
        self,
        seed_frame_idx: int,
        use_carryover_sampling: bool,
        sample_points_per_object: int,
        carryover_mode: str,
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
        elif use_carryover_sampling:
            if carryover_mode == "Mask AABB Box":
                sampled_boxes = self._sample_boxes_from_seed_masks(seed_frame_idx, h, w)
                for obj_id, box in sampled_boxes.items():
                    if (
                        obj_id in enabled_obj_ids
                        and obj_id not in prompts_to_apply
                        and obj_id not in boxes_to_apply
                    ):
                        boxes_to_apply[obj_id] = box
            else:
                sampled = self._sample_prompts_from_seed_masks(seed_frame_idx, sample_points_per_object, h, w)
                for obj_id, points in sampled.items():
                    if (
                        points
                        and obj_id in enabled_obj_ids
                        and obj_id not in prompts_to_apply
                        and obj_id not in boxes_to_apply
                    ):
                        prompts_to_apply[obj_id] = points

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
        enabled: set[int] = set()
        for i in range(self.object_list.count()):
            item = self.object_list.item(i)
            if item.checkState() == Qt.Checked:
                enabled.add(int(item.data(Qt.UserRole)))
        return enabled

    def _sample_prompts_from_seed_masks(
        self,
        seed_frame_idx: int,
        sample_points_per_object: int,
        image_h: int,
        image_w: int,
    ) -> Dict[int, List[PointPrompt]]:
        sampled_prompts: Dict[int, List[PointPrompt]] = {}
        output = self.outputs_by_frame.get(seed_frame_idx)
        if output is None:
            return sampled_prompts

        for i, obj_id in enumerate(output.obj_ids):
            if i >= len(output.masks):
                continue
            mask = output.masks[i].astype(np.uint8)
            if mask.shape[:2] != (image_h, image_w):
                mask = cv2.resize(mask, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
            ys, xs = np.where(mask > 0)
            if xs.size == 0:
                continue
            k = min(sample_points_per_object, int(xs.size))
            if k <= 0:
                continue
            if k == 1:
                pick_idx = np.array([xs.size // 2], dtype=int)
            else:
                pick_idx = np.linspace(0, xs.size - 1, num=k, dtype=int)
            sampled_prompts[obj_id] = [
                PointPrompt(x_px=int(xs[j]), y_px=int(ys[j]), is_positive=True)
                for j in pick_idx
            ]
        return sampled_prompts

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
    ) -> None:
        src_prompts = self.prompts_by_frame_obj.get(src_frame_idx, {})
        if not src_prompts:
            return
        dst_prompts = self.prompts_by_frame_obj.setdefault(dst_frame_idx, {})
        for obj_id in obj_ids:
            points = src_prompts.get(obj_id)
            if not points:
                continue
            if self.translate_prompts_on_propagation:
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
        delta_x = dst_cx - src_cx
        delta_y = dst_cy - src_cy

        translated: List[PointPrompt] = []
        for p in points:
            x_px = int(round(p.x_px + delta_x))
            y_px = int(round(p.y_px + delta_y))
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
        self._schedule_prefetch_for_next_frame()

    def export_annotations(self) -> None:
        if not self.frame_paths:
            QMessageBox.information(self, "Nothing to export", "Load a frame directory first.")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not out_dir:
            return

        exporter = CocoExporter(Path(out_dir))
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
        self.frame_label.setText(f"Frame: {self.current_frame_idx + 1}/{len(self.frame_paths)}")
        self._sync_frame_navigation_controls()
        self.refresh_point_list()
        self._sync_current_box_lock_check()

    def _draw_output_overlay(self, img: np.ndarray, output: SamFrameOutput) -> np.ndarray:
        overlay = img.copy()
        h, w = img.shape[:2]
        mask_alpha = float(max(0.0, min(1.0, self.segmentation_opacity)))

        for i, obj_id in enumerate(output.obj_ids):
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

    def _draw_points_overlay(self, img: np.ndarray) -> np.ndarray:
        frame_boxes = self.box_prompts_by_frame_obj.get(self.current_frame_idx, {})
        for obj_id, box in frame_boxes.items():
            obj = self._find_object(obj_id)
            color = obj.color_bgr if obj else (255, 255, 255)
            if self.show_boxes:
                self._draw_box_outline(img, box, color, self.box_line_thickness, dashed=False)
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
                obj = self._find_object(obj_id)
                color = obj.color_bgr if obj else (255, 255, 255)
                for idx, p in enumerate(plist):
                    radius = 6
                    if obj_id == self.active_object_id and idx == selected_prompt_idx:
                        cv2.circle(img, (p.x_px, p.y_px), radius + 3, (255, 255, 255), 2)
                    if p.is_positive:
                        cv2.circle(img, (p.x_px, p.y_px), radius, color, -1)
                    else:
                        cv2.circle(img, (p.x_px, p.y_px), radius, color, 2)
                        cv2.line(img, (p.x_px - radius, p.y_px - radius), (p.x_px + radius, p.y_px + radius), color, 2)
                        cv2.line(img, (p.x_px - radius, p.y_px + radius), (p.x_px + radius, p.y_px - radius), color, 2)

        return img

    def _get_selected_point_prompt_idx(self) -> Optional[int]:
        row = self.point_list.currentRow()
        if row < 0 or row >= len(self._active_prompt_rows):
            return None
        prompt_type, idx = self._active_prompt_rows[row]
        if prompt_type != "point":
            return None
        return idx

    def _is_box_mode(self) -> bool:
        return not self._use_point_prompt_mode()

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
        threshold_px = 12
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

    def _point_in_box(self, box: BoxPrompt, x_px: int, y_px: int) -> bool:
        return box.x1_px <= x_px <= box.x2_px and box.y1_px <= y_px <= box.y2_px

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
        handle_radius = 6
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
