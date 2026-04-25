#!/usr/bin/env python3
"""PySide6 surgical video annotation app using SAM3 as assistive annotator."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image as PilImage
from PySide6.QtCore import QObject, QPoint, QRect, QSize, Qt, QThread, Signal, QEvent, QEventLoop, QMetaObject, QTimer
from PySide6.QtGui import QAction, QBrush, QColor, QGuiApplication, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSlider,
    QAbstractSpinBox,
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
    QMenu,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QRubberBand,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from annotator.widgets import ArrowSpinBox, ClickableImageLabel

from annotator.exporters.coco_export import CocoExporter, ObjectInfo as ExportObjectInfo
from annotator.exporters.coco_export import PointPrompt as ExportPointPrompt
from annotator.exporters.coco_export import SamFrameOutput as ExportSamFrameOutput
from annotator.exporters.perk_export import BoxPrompt as ExportPerkBoxPrompt
from annotator.exporters.perk_export import ObjectInfo as ExportPerkObjectInfo
from annotator.exporters.perk_export import PerkExporter
from annotator.models import (
    BoxPrompt,
    DEFAULT_BOX_LINE_THICKNESS,
    DEFAULT_PROPAGATION_CHUNK_SIZE,
    DEFAULT_PROPAGATION_CHUNKS,
    DEFAULT_RECONDITION_EVERY_NTH_FRAME,
    DEFAULT_RECONDITION_HIGH_CONF_THRESH,
    DEFAULT_RECONDITION_HIGH_IOU_THRESH,
    DEFAULT_SEGMENTATION_OPACITY,
    DEFAULT_SMART_PROPAGATION_RECOVERY_CHUNK_SIZE,
    DEFAULT_SMART_PROPAGATION_REWIND_FRAMES,
    ExperimentalSettings,
    ObjectInfo,
    PendingPropagationState,
    PointPrompt,
    PROPAGATION_MODE_COPY_BOXES,
    PROPAGATION_MODE_TRACKER,
    PropagationSettings,
    SamFrameOutput,
    ViewSettings,
    normalize_propagation_mode,
)
from annotator.persistence.session_models import SessionPayload
from annotator.persistence.session_repository import SessionRepository
from annotator.prompts.text_prompt_grounding import (
    TextPromptProposal,
    build_text_prompt_proposals,
    next_prompt_object_name,
)
from annotator.propagation.smart_propagation import (
    SmartPropagationSettings,
    detect_smart_propagation_restart,
)
from annotator.propagation.runtime import (
    PrefetchState,
    PropagationRuntimeState,
    SamTaskContext,
    TASK_KIND_MANUAL_PROPAGATION,
    TASK_KIND_PREFETCH,
    TASK_KIND_PREFETCH_WAIT,
    TASK_KIND_TEXT_PROMPT,
    TASK_KIND_UPDATE_EXPERIMENTAL_SETTINGS,
)
from annotator.propagation.frame_outputs import (
    merge_frame_outputs,
    remove_object_from_output,
    sample_boxes_from_output_masks,
)
from annotator.propagation.prompt_payloads import (
    build_propagation_seed_payload,
    build_segment_prompt_payload,
    clone_point_prompts,
    prompt_payload_inputs_to_task_payload,
    translate_prompts_by_box_delta,
)
from annotator.research.controller import ResearchController
from annotator.sam import SamWorker

MAX_POINTS_PER_OBJECT = 6
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
WINDOW_SCREEN_FRACTION = 0.85
CANVAS_SCREEN_FRACTION = (0.7, 0.6)
CANVAS_SCREEN_MARGIN_PX = 40
DEFAULT_CHECKPOINT_OPTION_LABEL = "Use default HF checkpoint"
CHECKPOINT_PRESET_OPTIONS: List[Tuple[str, Optional[str]]] = [
    (DEFAULT_CHECKPOINT_OPTION_LABEL, None),
    ("sam3.1_multiplex (local)", r"C:\Users\denes\Downloads\sam3.1_multiplex.pt"),
]
SESSION_REPOSITORY = SessionRepository()


class AnnotatorMainWindow(QMainWindow):
    """Primary Qt window that orchestrates annotation state, rendering, and SAM workflows."""

    task_requested = Signal(str, str, object, bool)
    def __init__(self, research_mode_enabled: bool = False):
        """Initialize window state, runtime helpers, and all UI owned by the annotator."""
        super().__init__()
        self.setWindowTitle("SAM3 Surgical Video Annotator")
        self._apply_window_sizing()

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
        self._propagation_runtime = PropagationRuntimeState()
        self._sam_thread: Optional[QThread] = None
        self._sam_worker: Optional[SamWorker] = None
        self._sam_ready: bool = False
        self._sam_task_counter: int = 0
        self._sam_task_contexts: Dict[str, SamTaskContext] = {}
        self._sam_waiting: Dict[str, QEventLoop] = {}
        self._sam_task_results: Dict[str, object] = {}
        self._prefetch_state = PrefetchState()
        self._output_version_by_frame: Dict[int, int] = {}
        self._undo_state: Optional[dict] = None
        self._last_prompt_edit_frame: Optional[int] = None
        self._one_session_chunk_cache_generation: Optional[int] = None
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
        self.propagation_mode: str = PROPAGATION_MODE_TRACKER
        self.translate_prompts_on_propagation: bool = True
        self.use_point_prompts_for_propagation: bool = True
        self.use_target_frame: bool = False
        self.use_one_session_chunked_propagation: bool = False
        self.smart_propagation_enabled: bool = False
        self.smart_propagation_rewind_frames: int = DEFAULT_SMART_PROPAGATION_REWIND_FRAMES
        self.smart_propagation_recovery_chunk_size: int = DEFAULT_SMART_PROPAGATION_RECOVERY_CHUNK_SIZE
        self.segmentation_opacity: float = DEFAULT_SEGMENTATION_OPACITY
        self.box_line_thickness: int = DEFAULT_BOX_LINE_THICKNESS
        self.recondition_every_nth_frame: int = DEFAULT_RECONDITION_EVERY_NTH_FRAME
        self.recondition_high_conf_thresh: float = DEFAULT_RECONDITION_HIGH_CONF_THRESH
        self.recondition_high_iou_thresh: float = DEFAULT_RECONDITION_HIGH_IOU_THRESH
        self._research_mode_enabled: bool = bool(research_mode_enabled)
        self._research_controller = ResearchController(self, enabled=self._research_mode_enabled)

        self._setup_ui()
        if self._research_mode_enabled:
            QApplication.instance().installEventFilter(self)
            self._research_controller.start_ui_updates()
        QTimer.singleShot(0, self._apply_canvas_sizing)

    def eventFilter(self, watched: QObject, event) -> bool:
        """Capture research telemetry and return focus to the canvas after text-entry keys."""
        if self._research_mode_enabled:
            self._research_controller.handle_widget_event(
                watched=watched,
                event=event,
                current_frame_idx=self.current_frame_idx,
                has_frames=bool(self.frame_paths),
                image_label=self.image_label,
            )
            canvas_xy = None
            if watched is self.image_label and event.type() == QEvent.MouseMove and hasattr(event, "position"):
                position = event.position()
                canvas_xy = self._map_ui_to_image_xy(position.x(), position.y())
            self._research_controller.record_mouse_position(
                watched=watched,
                event=event,
                current_frame_idx=self.current_frame_idx,
                has_frames=bool(self.frame_paths),
                canvas_xy=canvas_xy,
            )
        if event.type() == QEvent.KeyPress and event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Escape):
            handled = super().eventFilter(watched, event)
            QTimer.singleShot(0, self._focus_canvas)
            return handled
        return super().eventFilter(watched, event)

    @property
    def _prefetch_busy(self) -> bool:
        """Compatibility shim exposing ``PrefetchState.busy`` on the window."""
        return self._prefetch_state.busy

    @_prefetch_busy.setter
    def _prefetch_busy(self, value: bool) -> None:
        """Compatibility shim updating ``PrefetchState.busy`` on the window."""
        self._prefetch_state.busy = bool(value)

    @property
    def _prefetch_cancel_requested(self) -> bool:
        """Compatibility shim exposing ``PrefetchState.cancel_requested`` on the window."""
        return self._prefetch_state.cancel_requested

    @_prefetch_cancel_requested.setter
    def _prefetch_cancel_requested(self, value: bool) -> None:
        """Compatibility shim updating ``PrefetchState.cancel_requested`` on the window."""
        self._prefetch_state.cancel_requested = bool(value)

    @property
    def _prefetch_pending_restart(self) -> bool:
        """Compatibility shim exposing ``PrefetchState.pending_restart`` on the window."""
        return self._prefetch_state.pending_restart

    @_prefetch_pending_restart.setter
    def _prefetch_pending_restart(self, value: bool) -> None:
        """Compatibility shim updating ``PrefetchState.pending_restart`` on the window."""
        self._prefetch_state.pending_restart = bool(value)

    @property
    def _prefetch_target_frame_idx(self) -> Optional[int]:
        """Compatibility shim exposing ``PrefetchState.target_frame_idx`` on the window."""
        return self._prefetch_state.target_frame_idx

    @_prefetch_target_frame_idx.setter
    def _prefetch_target_frame_idx(self, value: Optional[int]) -> None:
        """Compatibility shim updating ``PrefetchState.target_frame_idx`` on the window."""
        self._prefetch_state.target_frame_idx = value

    @property
    def _prefetch_seed_frame_idx(self) -> Optional[int]:
        """Compatibility shim exposing ``PrefetchState.seed_frame_idx`` on the window."""
        return self._prefetch_state.seed_frame_idx

    @_prefetch_seed_frame_idx.setter
    def _prefetch_seed_frame_idx(self, value: Optional[int]) -> None:
        """Compatibility shim updating ``PrefetchState.seed_frame_idx`` on the window."""
        self._prefetch_state.seed_frame_idx = value

    @property
    def _prefetch_prompt_version(self) -> int:
        """Compatibility shim exposing ``PrefetchState.cache_generation`` on the window."""
        return self._prefetch_state.cache_generation

    @_prefetch_prompt_version.setter
    def _prefetch_prompt_version(self, value: int) -> None:
        """Compatibility shim updating ``PrefetchState.cache_generation`` on the window."""
        self._prefetch_state.cache_generation = int(value)

    @property
    def _prefetch_active_version(self) -> Optional[int]:
        """Compatibility shim exposing ``PrefetchState.active_generation`` on the window."""
        return self._prefetch_state.active_generation

    @_prefetch_active_version.setter
    def _prefetch_active_version(self, value: Optional[int]) -> None:
        """Compatibility shim updating ``PrefetchState.active_generation`` on the window."""
        self._prefetch_state.active_generation = value

    @property
    def _prefetch_cached_frame_idx(self) -> Optional[int]:
        """Compatibility shim exposing ``PrefetchState.cached_frame_idx`` on the window."""
        return self._prefetch_state.cached_frame_idx

    @_prefetch_cached_frame_idx.setter
    def _prefetch_cached_frame_idx(self, value: Optional[int]) -> None:
        """Compatibility shim updating ``PrefetchState.cached_frame_idx`` on the window."""
        self._prefetch_state.cached_frame_idx = value

    @property
    def _prefetch_cached_seed_idx(self) -> Optional[int]:
        """Compatibility shim exposing ``PrefetchState.cached_seed_idx`` on the window."""
        return self._prefetch_state.cached_seed_idx

    @_prefetch_cached_seed_idx.setter
    def _prefetch_cached_seed_idx(self, value: Optional[int]) -> None:
        """Compatibility shim updating ``PrefetchState.cached_seed_idx`` on the window."""
        self._prefetch_state.cached_seed_idx = value

    @property
    def _prefetch_cached_version(self) -> Optional[int]:
        """Compatibility shim exposing ``PrefetchState.cached_generation`` on the window."""
        return self._prefetch_state.cached_generation

    @_prefetch_cached_version.setter
    def _prefetch_cached_version(self, value: Optional[int]) -> None:
        """Compatibility shim updating ``PrefetchState.cached_generation`` on the window."""
        self._prefetch_state.cached_generation = value

    @property
    def _prefetch_provenance_by_frame(self):
        """Compatibility shim exposing cached prefetch provenance keyed by frame index."""
        return self._prefetch_state.provenance_by_frame

    @property
    def _propagation_busy(self) -> bool:
        """Compatibility shim exposing ``PropagationRuntimeState.busy`` on the window."""
        return self._propagation_runtime.busy

    @_propagation_busy.setter
    def _propagation_busy(self, value: bool) -> None:
        """Compatibility shim updating ``PropagationRuntimeState.busy`` on the window."""
        self._propagation_runtime.busy = bool(value)

    @property
    def _propagation_enabled_obj_ids(self) -> Optional[set[int]]:
        """Compatibility shim exposing ``PropagationRuntimeState.enabled_obj_ids``."""
        return self._propagation_runtime.enabled_obj_ids

    @_propagation_enabled_obj_ids.setter
    def _propagation_enabled_obj_ids(self, value: Optional[set[int]]) -> None:
        """Compatibility shim updating ``PropagationRuntimeState.enabled_obj_ids``."""
        self._propagation_runtime.enabled_obj_ids = None if value is None else set(value)

    @property
    def _propagation_view_frame_idx(self) -> Optional[int]:
        """Compatibility shim exposing ``PropagationRuntimeState.view_frame_idx``."""
        return self._propagation_runtime.view_frame_idx

    @_propagation_view_frame_idx.setter
    def _propagation_view_frame_idx(self, value: Optional[int]) -> None:
        """Compatibility shim updating ``PropagationRuntimeState.view_frame_idx``."""
        self._propagation_runtime.view_frame_idx = value

    @property
    def _propagation_continue_after_chunk(self) -> bool:
        """Compatibility shim exposing ``PropagationRuntimeState.continue_after_chunk``."""
        return self._propagation_runtime.continue_after_chunk

    @_propagation_continue_after_chunk.setter
    def _propagation_continue_after_chunk(self, value: bool) -> None:
        """Compatibility shim updating ``PropagationRuntimeState.continue_after_chunk``."""
        self._propagation_runtime.continue_after_chunk = bool(value)

    @property
    def _propagation_active_chunk_idx(self) -> Optional[int]:
        """Compatibility shim exposing ``PropagationRuntimeState.active_chunk_idx``."""
        return self._propagation_runtime.active_chunk_idx

    @_propagation_active_chunk_idx.setter
    def _propagation_active_chunk_idx(self, value: Optional[int]) -> None:
        """Compatibility shim updating ``PropagationRuntimeState.active_chunk_idx``."""
        self._propagation_runtime.active_chunk_idx = value

    @property
    def _propagation_active_seed_frame_idx(self) -> Optional[int]:
        """Compatibility shim exposing ``PropagationRuntimeState.active_seed_frame_idx``."""
        return self._propagation_runtime.active_seed_frame_idx

    @_propagation_active_seed_frame_idx.setter
    def _propagation_active_seed_frame_idx(self, value: Optional[int]) -> None:
        """Compatibility shim updating ``PropagationRuntimeState.active_seed_frame_idx``."""
        self._propagation_runtime.active_seed_frame_idx = value

    @property
    def _propagation_task_id(self) -> Optional[str]:
        """Compatibility shim exposing ``PropagationRuntimeState.task_id`` on the window."""
        return self._propagation_runtime.task_id

    @_propagation_task_id.setter
    def _propagation_task_id(self, value: Optional[str]) -> None:
        """Compatibility shim updating ``PropagationRuntimeState.task_id`` on the window."""
        self._propagation_runtime.task_id = value

    @property
    def _propagation_stop_requested(self) -> bool:
        """Compatibility shim exposing ``PropagationRuntimeState.stop_requested``."""
        return self._propagation_runtime.stop_requested

    @_propagation_stop_requested.setter
    def _propagation_stop_requested(self, value: bool) -> None:
        """Compatibility shim updating ``PropagationRuntimeState.stop_requested``."""
        self._propagation_runtime.stop_requested = bool(value)

    @property
    def _propagation_seen_obj_ids(self) -> set[int]:
        """Compatibility shim exposing the set of objects seen during the current run."""
        return self._propagation_runtime.seen_obj_ids

    @property
    def _propagation_lost_obj_ids(self) -> set[int]:
        """Compatibility shim exposing objects marked lost during smart propagation."""
        return self._propagation_runtime.lost_obj_ids

    @property
    def _smart_propagation_waiting_for_recovery_obj_ids(self) -> set[int]:
        """Compatibility shim exposing objects waiting for recovery after a rewind."""
        return self._propagation_runtime.waiting_for_recovery_obj_ids

    @property
    def _propagation_loss_notified(self) -> bool:
        """Compatibility shim exposing ``PropagationRuntimeState.loss_notified``."""
        return self._propagation_runtime.loss_notified

    @_propagation_loss_notified.setter
    def _propagation_loss_notified(self, value: bool) -> None:
        """Compatibility shim updating ``PropagationRuntimeState.loss_notified``."""
        self._propagation_runtime.loss_notified = bool(value)

    @property
    def _propagation_smart_restart_requested(self) -> bool:
        """Compatibility shim exposing ``PropagationRuntimeState.smart_restart_requested``."""
        return self._propagation_runtime.smart_restart_requested

    @_propagation_smart_restart_requested.setter
    def _propagation_smart_restart_requested(self, value: bool) -> None:
        """Compatibility shim updating ``PropagationRuntimeState.smart_restart_requested``."""
        self._propagation_runtime.smart_restart_requested = bool(value)

    @property
    def _smart_propagation_triggered_loss_keys(self) -> set[tuple[int, int]]:
        """Compatibility shim exposing the set of loss events that already triggered rewinds."""
        return self._propagation_runtime.triggered_loss_keys

    @property
    def _one_session_chunk_prompt_version(self) -> Optional[int]:
        """Compatibility shim exposing the prompt generation for one-session chunk reuse."""
        return self._one_session_chunk_cache_generation

    @_one_session_chunk_prompt_version.setter
    def _one_session_chunk_prompt_version(self, value: Optional[int]) -> None:
        """Compatibility shim updating the prompt generation for one-session chunk reuse."""
        self._one_session_chunk_cache_generation = value

    def _focus_canvas(self) -> None:
        """Return keyboard focus to the canvas so navigation shortcuts stay active."""
        if hasattr(self, "image_label") and self.image_label is not None:
            self.image_label.setFocus(Qt.ShortcutFocusReason)

    def _install_focus_return_on_widget(self, widget: Optional[QWidget]) -> None:
        """Install focus-return filters on editors whose Enter/Escape should hand focus back."""
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
        """Return the usable screen rectangle, with a desktop-sized fallback for safety."""
        screen = self.windowHandle().screen() if self.windowHandle() else QGuiApplication.primaryScreen()
        if screen is None:
            return QRect(0, 0, 1600, 900)
        return screen.availableGeometry()

    def _apply_window_sizing(self) -> None:
        """Size the main window as a fraction of the available desktop area."""
        geom = self._get_available_screen_geometry()
        avail_w = max(1, geom.width())
        avail_h = max(1, geom.height())
        win_w = max(1, int(avail_w * WINDOW_SCREEN_FRACTION))
        win_h = max(1, int(avail_h * WINDOW_SCREEN_FRACTION))
        self.resize(win_w, win_h)

    def _apply_canvas_sizing(self) -> None:
        """Reserve a large stable canvas while accounting for the side-panel widths."""
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
        """Assemble the full window from focused builder methods."""
        self._build_menu()
        root = QWidget()
        root_layout = QHBoxLayout(root)
        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)
        left_panel = self._build_left_review_panel()
        center_panel = self._build_center_panel()
        right_panel = self._build_right_panel()
        self._register_advanced_widgets()
        self._set_advanced_controls_visible(True)
        self.mode_label.setText("Mode: Box Annotation")
        self._configure_root_splitter(splitter, left_panel, center_panel, right_panel)
        self.setCentralWidget(root)
        self._setup_shortcuts()
        self._install_focus_return_widgets()
        self._sync_propagation_mode_controls()
        self._build_status_bar()
        if self._research_mode_enabled:
            self._enable_research_mouse_tracking(self)

    def _enable_research_mouse_tracking(self, root: QWidget) -> None:
        """Enable passive mouse-move events for research whole-window position logging."""
        root.setMouseTracking(True)
        for child in root.findChildren(QWidget):
            child.setMouseTracking(True)

    def _build_menu(self) -> None:
        """Create the application menu bar with grouped workflow actions."""
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("File")
        load_action = QAction("Load Frame Directory...", self)
        load_action.triggered.connect(self.load_frame_directory)
        file_menu.addAction(load_action)
        load_session_action = QAction("Load Session...", self)
        load_session_action.triggered.connect(self.load_session_dialog)
        file_menu.addAction(load_session_action)
        save_action = QAction("Save Session...", self)
        save_action.triggered.connect(self.save_session_dialog)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        export_action = QAction("Export Annotations...", self)
        export_action.triggered.connect(self.export_annotations)
        file_menu.addAction(export_action)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        edit_menu = menu_bar.addMenu("Edit")
        undo_action = QAction("Undo", self)
        undo_action.setShortcut(QKeySequence.Undo)
        undo_action.triggered.connect(self.undo_last_prompt_change)
        edit_menu.addAction(undo_action)
        delete_prompt_action = QAction("Delete Selected Prompt/Box", self)
        delete_prompt_action.setShortcut(QKeySequence(Qt.Key_Delete))
        delete_prompt_action.triggered.connect(self._delete_current_prompt_shortcut)
        edit_menu.addAction(delete_prompt_action)

        view_menu = menu_bar.addMenu("View")
        fit_action = QAction("Fit to Screen", self)
        fit_action.triggered.connect(self.fit_current_frame_to_view)
        view_menu.addAction(fit_action)

        navigate_menu = menu_bar.addMenu("Navigate")
        prev_frame_action = QAction("Previous Frame", self)
        prev_frame_action.setShortcut(QKeySequence(Qt.Key_Left))
        prev_frame_action.triggered.connect(self.go_prev_frame)
        navigate_menu.addAction(prev_frame_action)
        next_frame_action = QAction("Next Frame", self)
        next_frame_action.setShortcut(QKeySequence(Qt.Key_Right))
        next_frame_action.triggered.connect(self.go_next_frame_shortcut)
        navigate_menu.addAction(next_frame_action)
        navigate_menu.addSeparator()
        toggle_flag_action = QAction("Flag / Unflag Current Frame", self)
        toggle_flag_action.setShortcut(QKeySequence("F"))
        toggle_flag_action.triggered.connect(self.toggle_current_frame_flag)
        navigate_menu.addAction(toggle_flag_action)
        prev_flag_action = QAction("Previous Flagged Frame", self)
        prev_flag_action.setShortcut(QKeySequence("Shift+Left"))
        prev_flag_action.triggered.connect(self.go_prev_flagged_frame)
        navigate_menu.addAction(prev_flag_action)
        next_flag_action = QAction("Next Flagged Frame", self)
        next_flag_action.setShortcut(QKeySequence("Shift+Right"))
        next_flag_action.triggered.connect(self.go_next_flagged_frame)
        navigate_menu.addAction(next_flag_action)

        tools_menu = menu_bar.addMenu("Tools")
        segment_action = QAction("Segment Current Frame", self)
        segment_action.triggered.connect(self.segment_current_frame)
        tools_menu.addAction(segment_action)
        propagate_action = QAction("Propagate", self)
        propagate_action.setShortcut(QKeySequence("P"))
        propagate_action.triggered.connect(self.propagate_next_frame)
        tools_menu.addAction(propagate_action)

        help_menu = menu_bar.addMenu("Help")
        hotkeys_action = QAction("Hotkeys List", self)
        hotkeys_action.triggered.connect(self.show_hotkeys_list)
        help_menu.addAction(hotkeys_action)
        walkthrough_action = QAction("Program Explanation", self)
        walkthrough_action.triggered.connect(self.show_instructions_walkthrough)
        help_menu.addAction(walkthrough_action)
        help_menu.addSeparator()
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def _build_left_review_panel(self) -> QWidget:
        """Build the flagged-frame review panel shown on the left side."""
        left_panel = QWidget()
        left_panel.setMinimumWidth(180)
        left_panel.setMaximumWidth(260)
        self._left_review_panel = left_panel
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("Flagged Frames"))

        flag_button_row = QHBoxLayout()
        self.flag_toggle_btn = QPushButton("Flag / Unflag")
        self.flag_toggle_btn.clicked.connect(self.toggle_current_frame_flag)
        flag_button_row.addWidget(self.flag_toggle_btn)
        left_layout.addLayout(flag_button_row)

        jump_button_row = QHBoxLayout()
        self.prev_flag_btn = QPushButton("Prev Flag")
        self.prev_flag_btn.clicked.connect(self.go_prev_flagged_frame)
        self.next_flag_btn = QPushButton("Next Flag")
        self.next_flag_btn.clicked.connect(self.go_next_flagged_frame)
        jump_button_row.addWidget(self.prev_flag_btn)
        jump_button_row.addWidget(self.next_flag_btn)
        left_layout.addLayout(jump_button_row)

        self.flagged_frames_list = QListWidget()
        self.flagged_frames_list.currentItemChanged.connect(self._on_flagged_frame_selection_changed)
        left_layout.addWidget(self.flagged_frames_list)
        return left_panel

    def _build_center_panel(self) -> QWidget:
        """Build the frame navigation row and interactive image canvas."""
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
        return center_panel

    def _build_right_panel(self) -> QWidget:
        """Build the tabbed control column for prompting and propagation controls."""
        right_panel = QWidget()
        self._right_panel = right_panel
        right_panel.setMinimumWidth(550)
        right_layout = QVBoxLayout(right_panel)
        control_tabs = QTabWidget()
        right_layout.addWidget(control_tabs)
        control_tabs.addTab(self._build_prompt_tab(), "Prompting")
        control_tabs.addTab(self._build_processing_tab(), "Segment / Propagate")
        control_tabs.addTab(self._build_experimental_tab(), "Experimental Features")
        return right_panel

    def _build_prompt_tab(self) -> QWidget:
        """Build the object, prompt, and view controls tab."""
        prompt_tab = QWidget()
        prompt_layout = QVBoxLayout(prompt_tab)
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
        prompt_layout.addStretch(1)
        return prompt_tab

    def _build_processing_tab(self) -> QWidget:
        """Build the checkpoint, segmentation, and propagation controls tab."""
        processing_tab = QWidget()
        processing_layout = QVBoxLayout(processing_tab)

        model_form = QFormLayout()
        self.checkpoint_combo = QComboBox()
        for label, checkpoint_path in CHECKPOINT_PRESET_OPTIONS:
            self.checkpoint_combo.addItem(label, checkpoint_path)
        self.checkpoint_combo.setEditable(True)
        self.checkpoint_combo.lineEdit().setPlaceholderText("Optional local checkpoint path")
        model_form.addRow("Checkpoint", self.checkpoint_combo)
        processing_layout.addLayout(model_form)

        self.auto_propagate_next_check = QCheckBox("Auto Propagate Next Frame")
        self.auto_propagate_next_check.toggled.connect(self._on_auto_propagate_next_toggled)
        processing_layout.addWidget(self.auto_propagate_next_check)

        propagation_mode_form = QFormLayout()
        self.propagation_mode_combo = QComboBox()
        self.propagation_mode_combo.addItem("Tracker", PROPAGATION_MODE_TRACKER)
        self.propagation_mode_combo.addItem("Copy Boxes", PROPAGATION_MODE_COPY_BOXES)
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

        self.segment_btn = QPushButton("Segment")
        self.segment_btn.clicked.connect(self.segment_current_frame)
        processing_layout.addWidget(self.segment_btn)

        propagate_row = QHBoxLayout()
        self.propagate_btn = QPushButton("Propagate")
        self.propagate_btn.clicked.connect(self.propagate_next_frame)
        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setMinimum(2)
        self.chunk_size_spin.setMaximum(9999)
        self.chunk_size_spin.setValue(DEFAULT_PROPAGATION_CHUNK_SIZE)
        self.chunk_size_spin.setToolTip("Frames per propagation chunk, including the current frame")
        self.chunk_size_spin.valueChanged.connect(self._on_propagation_target_changed)
        self.chunks_spin = QSpinBox()
        self.chunks_spin.setMinimum(1)
        self.chunks_spin.setMaximum(9999)
        self.chunks_spin.setValue(1)
        self.chunks_spin.setToolTip("Number of propagation chunks to run")
        propagate_row.addWidget(self.propagate_btn)
        propagate_row.addWidget(QLabel("Chunk Size:"))
        propagate_row.addWidget(self.chunk_size_spin)
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
        processing_layout.addStretch(1)
        return processing_tab

    def _build_experimental_tab(self) -> QWidget:
        """Build the tracker heuristic controls that are intentionally optional."""
        experimental_tab = QWidget()
        experimental_layout = QVBoxLayout(experimental_tab)
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

        self.smart_propagation_check = QCheckBox("Enable Smart Propagation")
        self.smart_propagation_check.setChecked(self.smart_propagation_enabled)
        self.smart_propagation_check.setToolTip(
            "On target-frame tracker runs, rewind a few frames and run a larger recovery chunk after an object disappears."
        )
        self.smart_propagation_check.toggled.connect(self._on_experimental_settings_changed)
        experimental_layout.addWidget(self.smart_propagation_check)

        self.smart_propagation_rewind_spin = QSpinBox()
        self.smart_propagation_rewind_spin.setMinimum(0)
        self.smart_propagation_rewind_spin.setMaximum(9999)
        self.smart_propagation_rewind_spin.setValue(self.smart_propagation_rewind_frames)
        self.smart_propagation_rewind_spin.valueChanged.connect(self._on_experimental_settings_changed)
        experimental_form.addRow("Smart Rewind Frames", self.smart_propagation_rewind_spin)

        self.smart_propagation_chunk_size_spin = QSpinBox()
        self.smart_propagation_chunk_size_spin.setMinimum(2)
        self.smart_propagation_chunk_size_spin.setMaximum(9999)
        self.smart_propagation_chunk_size_spin.setValue(self.smart_propagation_recovery_chunk_size)
        self.smart_propagation_chunk_size_spin.valueChanged.connect(self._on_experimental_settings_changed)
        experimental_form.addRow("Recovery Chunk Size", self.smart_propagation_chunk_size_spin)

        experimental_layout.addLayout(experimental_form)
        experimental_layout.addStretch(1)
        return experimental_tab

    def _register_advanced_widgets(self) -> None:
        """Track widgets that can be globally shown or hidden as advanced controls."""
        self._advanced_widgets = [
            self.prompt_mode_label,
            self.prompt_mode_combo,
            self.segment_btn,
            self.show_prompts_check,
        ]

    def _configure_root_splitter(
        self,
        splitter: QSplitter,
        left_panel: QWidget,
        center_panel: QWidget,
        right_panel: QWidget,
    ) -> None:
        """Attach the three main panels and size them for the current screen."""
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

    def _install_focus_return_widgets(self) -> None:
        """Install enter/escape-to-canvas behavior on text-entry widgets."""
        for widget in (
            self.checkpoint_combo,
            self.frame_jump_spin,
            self.text_prompt_input,
            self.autosave_minutes_spin,
            self.segmentation_opacity_spin,
            self.box_line_thickness_spin,
            self.chunk_size_spin,
            self.chunks_spin,
            self.target_frame_spin,
            self.propagation_mode_combo,
            self.recondition_every_nth_frame_spin,
            self.recondition_high_conf_thresh_spin,
            self.recondition_high_iou_thresh_spin,
            self.use_one_session_chunked_propagation_check,
            self.smart_propagation_check,
            self.smart_propagation_rewind_spin,
            self.smart_propagation_chunk_size_spin,
        ):
            self._install_focus_return_on_widget(widget)

    def _build_status_bar(self) -> None:
        """Create the shared status widgets and optional research controls."""
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
        if not self._research_mode_enabled:
            return
        self._research_status_label = QLabel("Research: --:--:--")
        self._research_pause_btn = QPushButton("Pause")
        self._research_resume_btn = QPushButton("Resume")
        self._research_hide_btn = QPushButton("Hide Timer")
        self.statusBar().addPermanentWidget(self._research_status_label)
        self.statusBar().addPermanentWidget(self._research_pause_btn)
        self.statusBar().addPermanentWidget(self._research_resume_btn)
        self.statusBar().addPermanentWidget(self._research_hide_btn)
        self._research_controller.bind_status_widgets(
            status_label=self._research_status_label,
            pause_button=self._research_pause_btn,
            resume_button=self._research_resume_btn,
            hide_button=self._research_hide_btn,
        )

    def _on_prompt_mode_changed(self, idx: int) -> None:
        """Switch the sign of future point prompts based on the selected prompt mode."""
        self.current_prompt_positive = idx == 0

    def _on_auto_propagate_next_toggled(self, checked: bool) -> None:
        """Enable or disable one-step auto propagation and stop stale prefetch when disabling."""
        self.auto_propagate_next = checked
        if not checked:
            self._cancel_prefetch(restart=False)

    def _on_propagation_mode_changed(self, _idx: int) -> None:
        """Persist the selected propagation mode and invalidate stale prefetched outputs."""
        if not hasattr(self, "propagation_mode_combo"):
            return
        data = self.propagation_mode_combo.currentData()
        self.propagation_mode = normalize_propagation_mode(data)
        self._cancel_prefetch(restart=False)
        self._prefetch_state.clear_cache()
        self._sync_propagation_mode_controls()

    def _on_translate_prompts_toggled(self, checked: bool) -> None:
        """Store whether point prompts should shift with propagated boxes."""
        self.translate_prompts_on_propagation = checked

    def _on_use_point_prompts_for_propagation_toggled(self, checked: bool) -> None:
        """Store whether tracker propagation seeds should include point prompts."""
        self.use_point_prompts_for_propagation = checked

    def _on_use_one_session_chunked_propagation_toggled(self, checked: bool) -> None:
        """Toggle one-session chunk reuse and refresh dependent experimental controls."""
        self.use_one_session_chunked_propagation = checked
        if not checked:
            self._one_session_chunk_cache_generation = None
        self._sync_experimental_controls()

    def _on_experimental_settings_changed(self, _value) -> None:
        """Pull advanced tracker settings from widgets and push them to live state."""
        if hasattr(self, "recondition_every_nth_frame_spin"):
            self.recondition_every_nth_frame = int(self.recondition_every_nth_frame_spin.value())
        if hasattr(self, "recondition_high_conf_thresh_spin"):
            self.recondition_high_conf_thresh = float(self.recondition_high_conf_thresh_spin.value())
        if hasattr(self, "recondition_high_iou_thresh_spin"):
            self.recondition_high_iou_thresh = float(self.recondition_high_iou_thresh_spin.value())
        if hasattr(self, "smart_propagation_check"):
            self.smart_propagation_enabled = bool(self.smart_propagation_check.isChecked())
        if hasattr(self, "smart_propagation_rewind_spin"):
            self.smart_propagation_rewind_frames = int(self.smart_propagation_rewind_spin.value())
        if hasattr(self, "smart_propagation_chunk_size_spin"):
            self.smart_propagation_recovery_chunk_size = int(self.smart_propagation_chunk_size_spin.value())
        self._sync_experimental_controls()
        self._apply_experimental_settings_live()

    def _apply_experimental_settings_live(self) -> None:
        """Push tracker tuning changes to the live SAM worker without rebuilding the session."""
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
        self._sam_task_contexts[task_id] = SamTaskContext(
            kind=TASK_KIND_UPDATE_EXPERIMENTAL_SETTINGS
        )
        self._wait_for_sam_task(task_id)

    def _sync_experimental_controls(self) -> None:
        """Keep advanced widgets aligned with the current experimental feature state."""
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
        smart_controls_enabled = not bool(self.use_one_session_chunked_propagation)
        if hasattr(self, "smart_propagation_check"):
            self.smart_propagation_check.blockSignals(True)
            self.smart_propagation_check.setChecked(bool(self.smart_propagation_enabled))
            self.smart_propagation_check.setEnabled(smart_controls_enabled)
            tooltip = (
                "On target-frame tracker runs, rewind a few frames and run a larger recovery chunk after an object disappears."
            )
            if not smart_controls_enabled:
                tooltip += " Disabled while one-session chunked propagation is enabled."
            self.smart_propagation_check.setToolTip(tooltip)
            self.smart_propagation_check.blockSignals(False)
        if hasattr(self, "smart_propagation_rewind_spin"):
            self.smart_propagation_rewind_spin.blockSignals(True)
            self.smart_propagation_rewind_spin.setValue(int(self.smart_propagation_rewind_frames))
            self.smart_propagation_rewind_spin.setEnabled(smart_controls_enabled)
            self.smart_propagation_rewind_spin.blockSignals(False)
        if hasattr(self, "smart_propagation_chunk_size_spin"):
            self.smart_propagation_chunk_size_spin.blockSignals(True)
            self.smart_propagation_chunk_size_spin.setValue(int(self.smart_propagation_recovery_chunk_size))
            self.smart_propagation_chunk_size_spin.setEnabled(smart_controls_enabled)
            self.smart_propagation_chunk_size_spin.blockSignals(False)

    def _on_propagation_target_toggled(self, checked: bool) -> None:
        """Switch between explicit chunk-count mode and target-frame planning mode."""
        self.use_target_frame = checked
        self.target_frame_spin.setEnabled(checked)
        self.chunks_spin.setEnabled(not checked)
        if checked and self.frame_paths:
            current_value = self.current_frame_idx + 1
            if self.target_frame_spin.value() <= current_value:
                self.target_frame_spin.setValue(min(len(self.frame_paths), current_value + 1))
        self._update_target_chunk_label()

    def _on_propagation_target_changed(self, _value: int) -> None:
        """Refresh the derived chunk count label after target-frame edits."""
        self._update_target_chunk_label()

    def _compute_target_chunk_count(
        self,
        current_frame_idx: int,
        target_frame_idx: int,
        n_frames: int,
    ) -> Optional[int]:
        """Compute how many propagation chunks are needed to reach a target frame."""
        if n_frames <= 1 or target_frame_idx <= current_frame_idx:
            return None
        stride = n_frames - 1
        distance = target_frame_idx - current_frame_idx
        return int(math.ceil(distance / stride))

    def _update_target_chunk_label(self) -> None:
        """Update the read-only label that explains the computed target-frame chunk count."""
        if not hasattr(self, "computed_chunks_label"):
            return
        if not self.use_target_frame or not self.frame_paths:
            self.computed_chunks_label.setText("Chunks: -")
            return
        target_idx = self.target_frame_spin.value() - 1
        chunks = self._compute_target_chunk_count(
            self.current_frame_idx,
            target_idx,
            int(self.chunk_size_spin.value()),
        )
        if chunks is None:
            self.computed_chunks_label.setText("Chunks: -")
        else:
            self.computed_chunks_label.setText(f"Chunks: {chunks}")

    def _smart_propagation_settings(self) -> SmartPropagationSettings:
        """Package the current smart-restart controls for the pure decision helper."""
        return SmartPropagationSettings(
            enabled=bool(self.smart_propagation_enabled),
            rewind_frames=int(self.smart_propagation_rewind_frames),
            recovery_chunk_size=int(self.smart_propagation_recovery_chunk_size),
        )

    def _is_smart_propagation_available_for_run(self) -> bool:
        """Return whether the current propagation configuration can use recovery rewinds."""
        settings = self._smart_propagation_settings()
        return (
            settings.enabled
            and self.use_target_frame_check.isChecked()
            and self._is_tracker_propagation_mode()
            and not self.use_one_session_chunked_propagation
        )

    def request_stop_propagation(self) -> None:
        """Stop the active propagation run or clear a queued run before it starts."""
        if not self._propagation_busy:
            if self._pending_propagation is not None:
                self._clear_pending_propagation_state()
                self._set_status("Propagation stopped.", progress=0, total=1)
            return
        self._propagation_stop_requested = True
        if self._sam_worker is not None:
            self._sam_worker.cancel_propagation()

    def _on_autosave_toggled(self, checked: bool) -> None:
        """Start or stop the autosave timer after verifying a session folder exists."""
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
        """Restart autosave immediately when the interval changes while enabled."""
        if self.autosave_check.isChecked():
            self._start_autosave_timer()

    def _start_autosave_timer(self) -> None:
        """Arm the autosave timer using the currently selected minutes interval."""
        interval_ms = int(self.autosave_minutes_spin.value() * 60 * 1000)
        self._autosave_timer.stop()
        self._autosave_timer.start(interval_ms)

    def _ensure_autosave_running_if_enabled(self) -> None:
        """Resume autosave after saving/loading a session that now has a target directory."""
        if self.autosave_check.isChecked() and self._session_dir is not None:
            self._start_autosave_timer()

    def _handle_autosave(self) -> None:
        """Persist the current session snapshot when the autosave timer fires."""
        if self._session_dir is None or not self.frame_paths:
            return
        self._save_session(self._session_dir)

    def _on_object_lock_toggled(self, obj_id: int, checked: bool) -> None:
        """Lock or unlock one object's current-frame box and refresh dependent UI."""
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
        """Apply the same lock state to all visible boxes on the current frame."""
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
        """Show or hide the widgets registered as advanced controls."""
        for widget in self._advanced_widgets:
            widget.setVisible(visible)

    def _on_object_selection_changed(self, current: Optional[QListWidgetItem], _previous: Optional[QListWidgetItem]) -> None:
        """Update active-object state after the user selects a different object row."""
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
        """Register keyboard shortcuts for navigation, prompting, and propagation workflows."""
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
        """Return whether keyboard focus is inside a text-editing control."""
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
        """Delete the selected prompt unless a text-entry control currently owns focus."""
        if self._focus_widget_is_text_entry():
            return
        self.remove_selected_prompt()

    def _toggle_auto_propagate_shortcut(self) -> None:
        """Toggle auto-propagate unless the user is typing into an editor widget."""
        if self._focus_widget_is_text_entry():
            return
        self.auto_propagate_next_check.toggle()

    def _cycle_propagation_mode_shortcut(self) -> None:
        """Cycle through propagation modes unless keyboard focus is inside an editor."""
        if self._focus_widget_is_text_entry():
            return
        count = self.propagation_mode_combo.count()
        if count <= 1:
            return
        next_idx = (self.propagation_mode_combo.currentIndex() + 1) % count
        self.propagation_mode_combo.setCurrentIndex(next_idx)

    def show_hotkeys_list(self) -> None:
        """Show the built-in shortcut reference dialog."""
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
        """Show the basic workflow walkthrough for first-time annotator users."""
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

    def show_about_dialog(self) -> None:
        """Show a concise app-purpose dialog from the Help menu."""
        QMessageBox.about(
            self,
            "About SAM3 Annotator",
            "\n".join(
                [
                    "SAM3 Annotator",
                    "",
                    "Interactive surgical video annotation tool using SAM3-assisted prompting, "
                    "segmentation, propagation, review flags, session persistence, and export workflows.",
                ]
            ),
        )

    def _flagged_frame_display_text(self, frame_idx: int) -> str:
        """Build the list-row label for one flagged frame."""
        label = f"Frame {frame_idx + 1}"
        if 0 <= frame_idx < len(self.frame_paths):
            label += f" - {self.frame_paths[frame_idx].name}"
        return label

    def _sorted_flagged_frames(self) -> List[int]:
        """Return flagged frames in ascending order, filtering out stale indices."""
        return sorted(
            frame_idx
            for frame_idx in self.flagged_frame_indices
            if 0 <= frame_idx < len(self.frame_paths)
        )

    def _refresh_flagged_frame_list(self) -> None:
        """Rebuild the flagged-frame list widget from the current flagged-frame set."""
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
        """Select the list row for the current frame if it is flagged."""
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
        """Jump to the previous or next flagged frame, wrapping around at the ends."""
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
        """Toggle the review flag on the current frame and refresh the side panel."""
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
        """Jump to the previous flagged frame."""
        self._jump_to_flagged_frame(-1)

    def go_next_flagged_frame(self) -> None:
        """Jump to the next flagged frame."""
        self._jump_to_flagged_frame(1)

    def _on_flagged_frame_selection_changed(
        self,
        current: Optional[QListWidgetItem],
        _previous: Optional[QListWidgetItem],
    ) -> None:
        """Jump to the frame chosen in the flagged-frame review list."""
        if current is None:
            return
        frame_idx = current.data(Qt.UserRole)
        if frame_idx is None:
            return
        self._set_current_frame_idx(int(frame_idx))

    def _refresh_object_list_visuals(self) -> None:
        """Synchronize object-row widgets with current selection, visibility, and seed state."""
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
        """Pull render-only view settings from widgets and redraw the current frame."""
        self.show_prompts = self.show_prompts_check.isChecked()
        self.show_segmentations = self.show_segmentations_check.isChecked()
        self.show_boxes = self.show_boxes_check.isChecked()
        self.show_box_titles = self.show_box_titles_check.isChecked()
        self.segmentation_opacity = float(self.segmentation_opacity_spin.value())
        self.box_line_thickness = int(self.box_line_thickness_spin.value())
        self._render_current_frame()

    def _current_view_settings(self) -> ViewSettings:
        """Capture the current render-only display settings for persistence."""
        return ViewSettings(
            show_prompts=bool(self.show_prompts),
            show_segmentations=bool(self.show_segmentations),
            show_boxes=bool(self.show_boxes),
            show_box_titles=bool(self.show_box_titles),
            segmentation_opacity=float(self.segmentation_opacity),
            box_line_thickness=int(self.box_line_thickness),
        )

    def _apply_view_settings(self, settings: ViewSettings) -> None:
        """Restore persisted render settings into both state and widgets."""
        self.show_prompts = bool(settings.show_prompts)
        self.show_segmentations = bool(settings.show_segmentations)
        self.show_boxes = bool(settings.show_boxes)
        self.show_box_titles = bool(settings.show_box_titles)
        self.segmentation_opacity = float(settings.segmentation_opacity)
        self.box_line_thickness = int(settings.box_line_thickness)
        self.show_prompts_check.setChecked(self.show_prompts)
        self.show_segmentations_check.setChecked(self.show_segmentations)
        self.show_boxes_check.setChecked(self.show_boxes)
        self.show_box_titles_check.setChecked(self.show_box_titles)
        self.segmentation_opacity_spin.setValue(self.segmentation_opacity)
        self.box_line_thickness_spin.setValue(self.box_line_thickness)

    def _current_propagation_settings(self) -> PropagationSettings:
        """Capture current propagation controls for session persistence."""
        return PropagationSettings(
            mode=self.propagation_mode,
            chunk_size=int(self.chunk_size_spin.value()),
            chunks=int(self.chunks_spin.value()),
            use_target_frame=bool(self.use_target_frame_check.isChecked()),
            target_frame_idx=int(self.target_frame_spin.value() - 1),
            translate_prompts=bool(self.translate_prompts_on_propagation),
            use_point_prompts=bool(self.use_point_prompts_for_propagation),
            auto_propagate_next=bool(self.auto_propagate_next),
        )

    def _apply_propagation_settings(self, settings: PropagationSettings) -> None:
        """Restore persisted propagation controls and resync dependent widgets."""
        self.chunk_size_spin.setValue(int(settings.chunk_size))
        self.chunks_spin.setValue(int(settings.chunks))
        self.translate_prompts_on_propagation = bool(settings.translate_prompts)
        self.translate_prompts_check.setChecked(self.translate_prompts_on_propagation)
        self.use_point_prompts_for_propagation = bool(settings.use_point_prompts)
        self.use_point_prompts_for_propagation_check.setChecked(self.use_point_prompts_for_propagation)
        self.propagation_mode = normalize_propagation_mode(settings.mode)
        self._sync_propagation_mode_controls()
        self.auto_propagate_next = bool(settings.auto_propagate_next)
        self.auto_propagate_next_check.setChecked(self.auto_propagate_next)
        self.use_target_frame = bool(settings.use_target_frame)
        self.use_target_frame_check.setChecked(self.use_target_frame)
        target_idx = int(settings.target_frame_idx)
        if self.frame_paths:
            target_idx = max(0, min(len(self.frame_paths) - 1, target_idx))
        self.target_frame_spin.setValue(target_idx + 1)

    def _current_experimental_settings(self) -> ExperimentalSettings:
        """Capture current advanced tracker settings for session persistence."""
        return ExperimentalSettings(
            recondition_every_nth_frame=int(self.recondition_every_nth_frame),
            recondition_high_conf_thresh=float(self.recondition_high_conf_thresh),
            recondition_high_iou_thresh=float(self.recondition_high_iou_thresh),
            use_one_session_chunked_propagation=bool(self.use_one_session_chunked_propagation),
            smart_propagation_enabled=bool(self.smart_propagation_enabled),
            smart_propagation_rewind_frames=int(self.smart_propagation_rewind_frames),
            smart_propagation_recovery_chunk_size=int(self.smart_propagation_recovery_chunk_size),
        )

    def _apply_experimental_settings(self, settings: ExperimentalSettings) -> None:
        """Restore persisted advanced tracker settings and refresh the controls."""
        self.recondition_every_nth_frame = int(settings.recondition_every_nth_frame)
        self.recondition_high_conf_thresh = float(settings.recondition_high_conf_thresh)
        self.recondition_high_iou_thresh = float(settings.recondition_high_iou_thresh)
        self.use_one_session_chunked_propagation = bool(settings.use_one_session_chunked_propagation)
        self.smart_propagation_enabled = bool(settings.smart_propagation_enabled)
        self.smart_propagation_rewind_frames = int(settings.smart_propagation_rewind_frames)
        self.smart_propagation_recovery_chunk_size = int(settings.smart_propagation_recovery_chunk_size)
        self._sync_experimental_controls()

    def _build_session_payload(self) -> SessionPayload:
        """Snapshot the current annotator state into the typed persistence model."""
        return SessionPayload(
            frame_dir=str(self.image_dir),
            frame_files=[path.name for path in self.frame_paths],
            current_frame_idx=int(self.current_frame_idx),
            active_object_id=int(self.active_object_id) if self.active_object_id is not None else None,
            checkpoint_path=self._read_optional_combo_path(self.checkpoint_combo),
            flagged_frames=self._sorted_flagged_frames(),
            objects=[
                ObjectInfo(
                    obj_id=int(obj.obj_id),
                    name=obj.name,
                    color_bgr=tuple(int(channel) for channel in obj.color_bgr),
                )
                for obj in self.objects
            ],
            hidden_obj_ids=sorted(int(obj_id) for obj_id in self.hidden_obj_ids),
            solo_object_id=int(self.solo_object_id) if self.solo_object_id is not None else None,
            prompts_by_frame_obj={
                int(frame_idx): {
                    int(obj_id): [
                        PointPrompt(
                            x_px=int(point.x_px),
                            y_px=int(point.y_px),
                            is_positive=bool(point.is_positive),
                        )
                        for point in point_list
                    ]
                    for obj_id, point_list in per_obj.items()
                }
                for frame_idx, per_obj in self.prompts_by_frame_obj.items()
            },
            boxes_by_frame_obj={
                int(frame_idx): {
                    int(obj_id): BoxPrompt(
                        x1_px=int(box.x1_px),
                        y1_px=int(box.y1_px),
                        x2_px=int(box.x2_px),
                        y2_px=int(box.y2_px),
                    )
                    for obj_id, box in per_obj.items()
                }
                for frame_idx, per_obj in self.box_prompts_by_frame_obj.items()
            },
            box_locks_by_frame_obj={
                int(frame_idx): {
                    int(obj_id): bool(locked)
                    for obj_id, locked in per_obj.items()
                }
                for frame_idx, per_obj in self.box_locked_by_frame_obj.items()
            },
            propagation_overrides_by_frame_obj={
                int(frame_idx): {
                    int(obj_id): bool(enabled)
                    for obj_id, enabled in per_obj.items()
                }
                for frame_idx, per_obj in self.manual_propagation_overrides_by_frame_obj.items()
                if per_obj
            },
            outputs_by_frame={
                int(frame_idx): SamFrameOutput(
                    obj_ids=[int(obj_id) for obj_id in output.obj_ids],
                    masks=[
                        np.asarray(mask).copy() if mask is not None else np.zeros((1, 1), dtype=bool)
                        for mask in output.masks
                    ],
                    boxes_xywh_norm=[tuple(map(float, box)) for box in output.boxes_xywh_norm],
                    scores=[float(score) for score in output.scores],
                    tracker_scores=[float(score) for score in output.tracker_scores],
                )
                for frame_idx, output in self.outputs_by_frame.items()
            },
            view_settings=self._current_view_settings(),
            propagation_settings=self._current_propagation_settings(),
            experimental_settings=self._current_experimental_settings(),
            prompt_mode_index=int(self.prompt_mode_combo.currentIndex()),
        )

    def _object_display_text(self, obj_id: int) -> str:
        """Build the object-row label including tracker score and visibility markers."""
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
        """Return the current frame's tracker confidence for one object, if present."""
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
        """Create the inline controls used for one object row in the object list."""
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
        if self._research_mode_enabled:
            self._enable_research_mouse_tracking(row_widget)
        return row_widget

    def _is_object_visible(self, obj_id: int) -> bool:
        """Return whether an object should currently appear in overlays and review UI."""
        if self.solo_object_id is not None:
            return obj_id == self.solo_object_id
        return obj_id not in self.hidden_obj_ids

    def _object_has_current_frame_seed(self, obj_id: int) -> bool:
        """Return whether the current frame has any seed prompt or output for this object."""
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
        """Store or clear the per-frame manual override for propagation eligibility."""
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
        """Resolve default propagation enablement plus any per-frame manual override."""
        if not self._object_has_current_frame_seed(obj_id):
            return False
        override = self.manual_propagation_overrides_by_frame_obj.get(self.current_frame_idx, {}).get(obj_id)
        if override is None:
            return True
        return bool(override)

    def _on_object_propagate_toggled(self, obj_id: int, checked: bool) -> None:
        """Persist the propagation checkbox state for one object on the current frame."""
        self._set_active_object_by_id(obj_id)
        if not self._object_has_current_frame_seed(obj_id):
            self._set_manual_propagation_override(self.current_frame_idx, obj_id, None)
            return
        if checked:
            self._set_manual_propagation_override(self.current_frame_idx, obj_id, None)
        else:
            self._set_manual_propagation_override(self.current_frame_idx, obj_id, False)

    def _set_active_object_by_id(self, obj_id: int) -> None:
        """Select the matching object row so the rest of the UI follows that object."""
        for i in range(self.object_list.count()):
            item = self.object_list.item(i)
            if item is not None and int(item.data(Qt.UserRole)) == obj_id:
                self.object_list.setCurrentItem(item)
                break

    def rename_object(self, obj_id: int) -> None:
        """Prompt for and apply a new display name for one object."""
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
        """Toggle solo visibility mode for one object."""
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
        """Toggle hidden/visible state for one object."""
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
        """Redraw overlays so the selected prompt highlight stays in sync."""
        if self.frame_paths:
            self._render_current_frame()

    def _set_status(self, text: str, *, progress: Optional[int] = None, total: Optional[int] = None, indeterminate: bool = False) -> None:
        """Update the shared status text and progress indicator in the status bar."""
        self._status_label.setText(text)
        if indeterminate:
            self._status_progress.setRange(0, 0)
            return
        if total is not None:
            self._status_progress.setRange(0, max(1, total))
        if progress is not None:
            self._status_progress.setValue(progress)

    def _on_sam_worker_initialized(self, ok: bool, message: str) -> None:
        """Finish the blocking worker-start handshake after the thread reports readiness."""
        self._sam_ready = ok
        self._sam_init_error = message
        loop = self._sam_waiting.get("__init__")
        if loop is not None:
            loop.quit()

    def _enqueue_sam_task(self, task_type: str, payload: Dict[str, object], priority: bool = False) -> str:
        """Allocate a task id and queue work onto the background SAM worker thread."""
        self._sam_task_counter += 1
        task_id = f"{task_type}:{self._sam_task_counter}"
        if self._sam_worker is None:
            return task_id
        self.task_requested.emit(task_id, task_type, payload, priority)
        return task_id

    def _wait_for_sam_task(self, task_id: str) -> object:
        """Block the current UI flow until an asynchronous SAM task records a result."""
        loop = QEventLoop()
        self._sam_waiting[task_id] = loop
        loop.exec()
        self._sam_waiting.pop(task_id, None)
        return self._sam_task_results.pop(task_id, None)

    def _reset_prefetch_state(self) -> None:
        """Cancel in-flight prefetch work and reset cached prefetch state/provenance."""
        prefetch = self._prefetch_state
        self._cancel_prefetch(restart=False)
        prefetch.reset()

    def _is_tracker_propagation_mode(self) -> bool:
        """Return whether propagation is currently using the tracker backend."""
        return self.propagation_mode == PROPAGATION_MODE_TRACKER

    def _sync_propagation_mode_controls(self) -> None:
        """Keep propagation-mode widgets selected and enabled for the current backend."""
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
        """Invalidate stale next-frame cache after prompt edits and maybe start a new prefetch."""
        prefetch = self._prefetch_state
        if self.active_object_id is not None:
            self._invalidate_prefetched_next_frame_for_objects(
                seed_frame_idx=self.current_frame_idx,
                obj_ids={self.active_object_id},
            )
        prefetch.note_prompt_change()
        self._last_prompt_edit_frame = self.current_frame_idx
        next_idx = min(len(self.frame_paths) - 1, self.current_frame_idx + 1) if self.frame_paths else None
        if next_idx is not None:
            self._output_version_by_frame.pop(next_idx, None)
        self._schedule_prefetch_for_next_frame()

    def _has_valid_prefetch(self, target_frame_idx: int) -> bool:
        """Return whether the cached prefetch result is still valid for a target frame."""
        return self._prefetch_state.has_valid_cache(target_frame_idx)

    def _invalidate_prefetched_next_frame_for_objects(self, seed_frame_idx: int, obj_ids: set[int]) -> None:
        """Remove stale prefetched annotations for objects whose seed prompts just changed."""
        if not self.frame_paths or not obj_ids:
            return
        next_idx = seed_frame_idx + 1
        if next_idx >= len(self.frame_paths):
            return
        provenance = self._prefetch_state.provenance_by_frame.get(next_idx)
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
            self._prefetch_state.provenance_by_frame.pop(next_idx, None)
        if self.current_frame_idx == next_idx:
            self.refresh_point_list()
            self._sync_current_box_lock_check()
            self._render_current_frame()

    def _cancel_prefetch(self, *, restart: bool) -> None:
        """Request prefetch cancellation and optionally mark that it should be restarted."""
        prefetch = self._prefetch_state
        if not prefetch.busy:
            prefetch.pending_restart = False
            return
        prefetch.cancel_requested = True
        prefetch.pending_restart = restart
        if self._sam_worker is not None:
            QMetaObject.invokeMethod(self._sam_worker, "cancel_prefetch", Qt.QueuedConnection)


    def _schedule_prefetch_for_next_frame(self) -> None:
        """Kick off background one-step propagation to speed up next-frame navigation."""
        prefetch = self._prefetch_state
        propagation = self._propagation_runtime
        if not self.auto_propagate_next:
            return
        if not self._is_tracker_propagation_mode():
            return
        if propagation.busy:
            return
        if not self.frame_paths or self.current_frame_idx >= len(self.frame_paths) - 1:
            return
        next_idx = min(len(self.frame_paths) - 1, self.current_frame_idx + 1)
        if (
            next_idx in self.outputs_by_frame
            and self._output_version_by_frame.get(next_idx, -1) == prefetch.cache_generation
        ):
            return
        if self._has_valid_prefetch(next_idx):
            return
        if prefetch.busy:
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
        prefetch.begin(seed_frame_idx=self.current_frame_idx, target_frame_idx=target_frame_idx)

        task_id = self._enqueue_sam_task(
            "prefetch",
            {
                "seed_frame_idx": self.current_frame_idx,
                "n_frames": 2,
                "frame_paths": self.frame_paths,
                "prompt_payload": prompt_payload,
            },
        )
        self._sam_task_contexts[task_id] = SamTaskContext(
            kind=TASK_KIND_PREFETCH,
            target_frame_idx=target_frame_idx,
            seed_frame_idx=self.current_frame_idx,
            cache_generation=prefetch.cache_generation,
        )

    def on_image_hover(self, ui_x: Optional[float], ui_y: Optional[float]) -> None:
        """Show image-space cursor coordinates while hovering over the canvas."""
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
        """Enable or disable UI that must remain stable while propagation mutates state."""
        propagation = self._propagation_runtime
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
        smart_controls_enabled = enabled and not self.use_one_session_chunked_propagation
        self.smart_propagation_check.setEnabled(smart_controls_enabled)
        self.smart_propagation_rewind_spin.setEnabled(smart_controls_enabled)
        self.smart_propagation_chunk_size_spin.setEnabled(smart_controls_enabled)
        running = propagation.busy
        pending = self._pending_propagation is not None
        self.stop_propagate_btn.setEnabled(running or pending)
        self.use_target_frame_check.setEnabled(enabled)
        self.target_frame_spin.setEnabled(enabled and self.use_target_frame_check.isChecked())
        self.chunks_spin.setEnabled(enabled and not self.use_target_frame_check.isChecked())
        self.translate_prompts_check.setEnabled(enabled and self._is_tracker_propagation_mode())
        self.use_point_prompts_for_propagation_check.setEnabled(enabled and self._is_tracker_propagation_mode())
        self._sync_current_box_lock_check()

    def _sync_current_box_lock_check(self) -> None:
        """Synchronize the global lock-all checkbox from visible boxes on this frame."""
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
        """Return whether the current prompt mode expects point clicks instead of boxes."""
        return self.prompt_mode_combo.currentIndex() != 2

    def load_frame_directory(self) -> None:
        """Load a fresh frame directory, reset annotator state, and start a new run."""
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
        self._prefetch_state.provenance_by_frame.clear()
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
        self.propagation_mode = PROPAGATION_MODE_TRACKER
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
        self._start_new_research_experiment(dir_path, start_paused=False)

    def _read_optional_combo_path(self, combo: QComboBox) -> Optional[str]:
        """Resolve an editable combo box selection into an optional filesystem path."""
        text = combo.currentText().strip()
        if not text or text.lower().startswith("use default"):
            return None
        selected_data = combo.currentData()
        if isinstance(selected_data, str) and combo.currentIndex() >= 0:
            selected_label = combo.itemText(combo.currentIndex()).strip()
            if text == selected_label:
                return selected_data
        return text

    def _set_optional_combo_path(self, combo: QComboBox, value: Optional[str]) -> None:
        """Restore an optional filesystem path into an editable combo box."""
        if not value:
            combo.setCurrentIndex(0)
            return

        value_str = str(value).strip()
        for idx in range(combo.count()):
            item_data = combo.itemData(idx)
            if isinstance(item_data, str) and item_data == value_str:
                combo.setCurrentIndex(idx)
                return
        combo.setCurrentText(value_str)

    def add_object(self) -> None:
        """Prompt for a new object name and create the corresponding object entry."""
        name, ok = QInputDialog.getText(self, "Add Object", "Object name:")
        if not ok:
            return
        name = name.strip()
        if not name:
            QMessageBox.warning(self, "Invalid name", "Object name cannot be empty.")
            return
        self._create_object_entry(name)

    def _create_object_entry(self, name: str) -> ObjectInfo:
        """Create a new object model, row widget, and selection state."""
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
        """Ask SAM to generate object proposals for the current frame from a text prompt."""
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
        self._sam_task_contexts[task_id] = SamTaskContext(kind=TASK_KIND_TEXT_PROMPT)
        self._set_status("Generating text prompt proposals...", progress=0, total=1)
        self._wait_for_sam_task(task_id)

    def accept_selected_text_prompt_proposals(self) -> None:
        """Promote checked text-prompt proposals into real objects, boxes, and masks."""
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
        """Clear the current text-prompt proposal state and its overlay/list UI."""
        self._text_prompt_proposals_frame_idx = None
        self._text_prompt_last_prompt = ""
        self._text_prompt_proposals = []
        if hasattr(self, "text_prompt_list"):
            self.text_prompt_list.clear()
        self._render_current_frame()

    def _clear_text_prompt_proposals_if_needed(self, frame_idx: Optional[int] = None) -> None:
        """Discard proposals when leaving the frame that produced them or when forced."""
        if not self._text_prompt_proposals:
            return
        if frame_idx is None or self._text_prompt_proposals_frame_idx != frame_idx:
            self._text_prompt_proposals_frame_idx = None
            self._text_prompt_last_prompt = ""
            self._text_prompt_proposals = []
            if hasattr(self, "text_prompt_list"):
                self.text_prompt_list.clear()

    def _selected_text_prompt_proposal_indices(self) -> List[int]:
        """Return the proposal indices whose checklist entries are currently checked."""
        selected: List[int] = []
        for idx in range(self.text_prompt_list.count()):
            item = self.text_prompt_list.item(idx)
            if item is None:
                continue
            if item.checkState() == Qt.Checked:
                selected.append(idx)
        return selected

    def _refresh_text_prompt_list(self) -> None:
        """Rebuild the proposal checklist from the current in-memory proposal list."""
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
        """Insert one accepted text-prompt proposal into prompts, boxes, and outputs."""
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
        """Remove the active object and all of its annotations across every frame."""
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
        """Move to the previous frame if one exists."""
        if not self.frame_paths:
            return
        self._set_current_frame_idx(max(0, self.current_frame_idx - 1))

    def go_next_frame(self) -> None:
        """Move to the next frame if one exists."""
        if not self.frame_paths:
            return
        self._set_current_frame_idx(min(len(self.frame_paths) - 1, self.current_frame_idx + 1))

    def go_next_frame_shortcut(self) -> None:
        """Handle rightward navigation, optionally reusing auto-propagated next-frame state."""
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
        """Populate the immediate next frame using the current propagation configuration."""
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
            self._sam_task_contexts[task_id] = SamTaskContext(
                kind=TASK_KIND_PREFETCH,
                target_frame_idx=next_idx,
                seed_frame_idx=self.current_frame_idx,
                cache_generation=self._prefetch_prompt_version,
            )
            self._prefetch_busy = True
            self._prefetch_seed_frame_idx = self.current_frame_idx
            self._prefetch_target_frame_idx = next_idx
            self._prefetch_active_version = self._prefetch_prompt_version
            self._set_status("Prefetching next frame...", progress=0, total=1)
        if self._prefetch_busy:
            task_id = f"prefetch-wait:{self.current_frame_idx}->{next_idx}:{self._prefetch_prompt_version}"
            self._sam_task_contexts[task_id] = SamTaskContext(kind=TASK_KIND_PREFETCH_WAIT)
            self._sam_waiting[task_id] = QEventLoop()
            self._sam_waiting[task_id].exec()
            self._sam_waiting.pop(task_id, None)
            self._sam_task_contexts.pop(task_id, None)
        return self._has_valid_prefetch(next_idx)

    def _set_current_frame_idx(self, frame_idx: int) -> None:
        """Clamp, apply, and render a newly selected current frame index."""
        if not self.frame_paths:
            return
        self.current_frame_idx = max(0, min(len(self.frame_paths) - 1, frame_idx))
        self._clear_text_prompt_proposals_if_needed(self.current_frame_idx)
        self._sync_frame_navigation_controls()
        self._render_current_frame()

    def _sync_frame_navigation_controls(self) -> None:
        """Keep slider, spinbox, and target-frame controls aligned with current frame bounds."""
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
        """Navigate to the frame chosen in the slider widget."""
        if not self.frame_paths:
            return
        self._set_current_frame_idx(value - 1)

    def _on_frame_jump_changed(self, value: int) -> None:
        """Navigate to the frame chosen in the numeric jump control."""
        if not self.frame_paths:
            return
        target_idx = value - 1
        if self.auto_propagate_next and target_idx == self.current_frame_idx + 1:
            self.go_next_frame_shortcut()
            return
        self._set_current_frame_idx(target_idx)

    def fit_current_frame_to_view(self) -> None:
        """Reset zoom/pan so the current frame fits the canvas again."""
        self._zoom_multiplier = 1.0
        self._pan_offset_ui = (0.0, 0.0)
        self._pan_drag_last_ui_xy = None
        self._render_current_frame()

    def on_pan_press(self, ui_x: float, ui_y: float) -> None:
        """Start a canvas pan gesture from the current pointer location."""
        if not self.frame_paths:
            return
        self._pan_drag_last_ui_xy = (ui_x, ui_y)

    def on_pan_drag(self, ui_x: float, ui_y: float) -> None:
        """Update pan offsets during a right-drag gesture and rerender the canvas."""
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
        """Finish the active canvas pan gesture."""
        self._pan_drag_last_ui_xy = None

    def on_image_wheel(self, ui_x: float, ui_y: float, delta_y: int) -> None:
        """Zoom the canvas around the pointer (or center fallback) and preserve the anchor."""
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
        """Handle point-click annotation or begin a box interaction on the canvas."""
        if not self.frame_paths:
            return
        if self.active_object_id is None:
            QMessageBox.information(self, "No active object", "Select an object before annotating.")
            return

        mapped = self._map_ui_to_image_xy(ui_x, ui_y)
        if mapped is None:
            return
        x_px, y_px = mapped
        if self._research_mode_enabled:
            target_name = "canvas_box_press" if self._is_box_mode() else "canvas_prompt_press"
            self._research_controller.queue_canvas_click(
                target_name=target_name,
                x_px=x_px,
                y_px=y_px,
                gesture_kind="single_click",
                current_frame_idx=self.current_frame_idx,
                has_frames=bool(self.frame_paths),
                image_label=self.image_label,
                prompt_mode_index=int(self.prompt_mode_combo.currentIndex()),
                active_object_id=self.active_object_id,
            )

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
        """Select the smallest visible box under the cursor and enter box-edit mode."""
        if not self.frame_paths or not self.show_boxes:
            return
        mapped = self._map_ui_to_image_xy(ui_x, ui_y)
        if mapped is None:
            return
        if self._research_mode_enabled:
            self._research_controller.cancel_pending_click("canvas")
            self._research_controller.record_canvas_event(
                target_name="canvas_double_click",
                x_px=mapped[0],
                y_px=mapped[1],
                gesture_kind="double_click",
                current_frame_idx=self.current_frame_idx,
                has_frames=bool(self.frame_paths),
                image_label=self.image_label,
                prompt_mode_index=int(self.prompt_mode_combo.currentIndex()),
                active_object_id=self.active_object_id,
            )
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
        """Update the temporary box rubber-band during an active box drag."""
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
        """Commit the dragged box, optionally resegmenting if the box is unlocked."""
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
        """Return the mutable point list for the active object on the current frame."""
        frame_map = self.prompts_by_frame_obj.setdefault(self.current_frame_idx, {})
        return frame_map.setdefault(self.active_object_id, [])

    def refresh_point_list(self) -> None:
        """Rebuild the prompt list widget for the active object on the current frame."""
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
        """Delete the selected point or box prompt from the active object."""
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
        """Remove all prompts for the active object on the current frame."""
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
        """Create the SAM worker thread lazily and block until it reports readiness."""
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
        """Refine all prompted objects on the current frame."""
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
        """Segment the requested current-frame objects to refresh their boxes and masks."""
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
        """Build a segmentation payload, queue the task, and wait for the result."""
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
        """Start or continue a manual multi-frame propagation run using current controls."""
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
                    self.chunk_size_spin.value(),
                ) or 0
                if total_chunks <= 0:
                    QMessageBox.information(self, "Invalid target", "Target frame is too close for the current N frames setting.")
                    return
            self._pending_propagation = PendingPropagationState(
                remaining_chunks=total_chunks,
                next_seed_frame_idx=self.current_frame_idx,
                n_frames=self.chunk_size_spin.value(),
                total_chunks=total_chunks,
                target_frame_idx=target_frame_idx,
                run_start_frame_idx=self.current_frame_idx,
                active_chunk_n_frames=self.chunk_size_spin.value(),
            )
            self._propagation_stop_requested = False
            self._propagation_smart_restart_requested = False
            self._smart_propagation_triggered_loss_keys.clear()
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
        """Build normalized tracker seed payloads from prompts, boxes, and prior masks."""
        del use_carryover_sampling
        frame_boxes = self.box_prompts_by_frame_obj.get(seed_frame_idx, {})
        frame_prompts = self.prompts_by_frame_obj.get(seed_frame_idx, {})

        frame_size = self._get_frame_size(seed_frame_idx)
        if frame_size is None:
            QMessageBox.critical(self, "Propagation setup failed", "Failed to load prompt source frame.")
            return None
        w, h = frame_size
        payload_result = build_propagation_seed_payload(
            frame_prompts=frame_prompts,
            frame_boxes=frame_boxes,
            seed_output=self.outputs_by_frame.get(seed_frame_idx),
            enabled_obj_ids=enabled_obj_ids,
            frame_size=(w, h),
            use_point_prompts=bool(self.use_point_prompts_for_propagation),
            sampled_boxes=sample_boxes_from_output_masks(
                self.outputs_by_frame.get(seed_frame_idx),
                h,
                w,
            ),
        )

        if payload_result.missing_obj_ids:
            missing_names = [
                self._find_object(obj_id).name if self._find_object(obj_id) else str(obj_id)
                for obj_id in payload_result.missing_obj_ids
            ]
            QMessageBox.warning(
                self,
                "Propagation setup failed",
                "These enabled objects do not have any usable seed-frame prompts or masks: "
                + ", ".join(missing_names),
            )
            return None

        if not enabled_obj_ids:
            QMessageBox.information(
                self,
                "No prompts",
                "No valid tracker seed inputs are available for this chunk seed.",
            )
            return None

        if not payload_result.payload_by_obj_id:
            QMessageBox.information(
                self,
                "No prompts",
                "No valid prompts available for this chunk seed.",
            )
            return None

        return prompt_payload_inputs_to_task_payload(payload_result.payload_by_obj_id)

    def _build_segment_payload(
        self,
        frame_idx: int,
        obj_ids: set[int],
        show_no_prompts: bool,
    ) -> Optional[Dict[int, Dict[str, object]]]:
        """Build the per-object prompt payload needed for single-frame segmentation."""
        frame_size = self._get_frame_size(frame_idx)
        if frame_size is None:
            QMessageBox.critical(self, "Segmentation failed", "Failed to load current frame.")
            return None
        payload = build_segment_prompt_payload(
            frame_prompts=self.prompts_by_frame_obj.get(frame_idx, {}),
            frame_boxes=self.box_prompts_by_frame_obj.get(frame_idx, {}),
            obj_ids=obj_ids,
            frame_size=frame_size,
        )
        if not payload and show_no_prompts:
            QMessageBox.information(self, "No prompts", "Add point or box prompts for at least one object on this frame.")
        if not payload:
            return None
        return prompt_payload_inputs_to_task_payload(payload)

    def _copy_boxes_forward_once(
        self,
        src_frame_idx: int,
        dst_frame_idx: int,
        enabled_obj_ids: set[int],
    ) -> List[int]:
        """Copy canonical boxes one frame forward and invalidate replaced object outputs."""
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
        """Execute one copy-box propagation chunk synchronously across its destination range."""
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
        """Launch the next propagation chunk using either tracker or copy-box mode."""
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

        chunk_n_frames = int(state.active_chunk_n_frames or state.n_frames)
        self._propagation_busy = True
        self._propagation_active_chunk_idx = chunk_idx
        self._propagation_active_seed_frame_idx = seed_frame_idx
        self._set_propagation_ui_enabled(False)
        if state.pending_smart_restart_seed_frame_idx is not None:
            loss_frame_idx = state.pending_smart_restart_loss_frame_idx
            restart_note = f"Smart restart: rewound to frame {seed_frame_idx + 1}"
            if loss_frame_idx is not None:
                restart_note += f" after loss on frame {loss_frame_idx + 1}"
            self._set_status(restart_note, progress=state.completed_chunks, total=max(1, state.total_chunks))
            state.pending_smart_restart_seed_frame_idx = None
            state.pending_smart_restart_loss_frame_idx = None
        else:
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
                "n_frames": chunk_n_frames,
                "frame_paths": self.frame_paths,
                "prompt_payload": prompt_payload,
                "use_one_session": use_one_session,
                "rebuild_one_session": rebuild_one_session,
            },
        )
        self._sam_task_contexts[task_id] = SamTaskContext(
            kind=TASK_KIND_MANUAL_PROPAGATION,
            seed_frame_idx=seed_frame_idx,
            last_emitted_frame_idx=seed_frame_idx - 1,
            use_one_session=use_one_session,
            n_frames=chunk_n_frames,
        )
        self._propagation_task_id = task_id

    def _handle_propagation_frame(self, abs_frame_idx: int, output: SamFrameOutput, session_idx: int, total_frames: int) -> None:
        """Merge one emitted propagation frame into app state and refresh progress UI."""
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
        """Merge finished segmentation results into frame state and wake any waiter."""
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
        """Store text-prompt proposals, refresh proposal UI, and wake any waiter."""
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
        """Handle emitted frames for prefetch, auto-step, or manual propagation tasks."""
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
            self._maybe_request_smart_propagation_restart(
                abs_frame_idx=abs_frame_idx,
                output=output,
                enabled_obj_ids=self._propagation_enabled_obj_ids or set(),
                state=state,
                task_id=task_id,
            )
            if self._propagation_smart_restart_requested:
                return
            self._set_status(
                f"Chunk {state.completed_chunks + 1}/{state.total_chunks}: {session_idx + 1}/{total_frames} frames",
                progress=state.completed_chunks,
                total=state.total_chunks,
            )

    def _maybe_request_smart_propagation_restart(
        self,
        *,
        abs_frame_idx: int,
        output: SamFrameOutput,
        enabled_obj_ids: set[int],
        state: PendingPropagationState,
        task_id: str,
    ) -> None:
        """Detect object loss and convert it into a rewind-and-recover request."""
        if self._propagation_smart_restart_requested or not enabled_obj_ids:
            return
        if not self._is_smart_propagation_available_for_run():
            return
        cooldown_until_frame_idx = state.smart_restart_cooldown_until_frame_idx
        if cooldown_until_frame_idx is not None and abs_frame_idx <= cooldown_until_frame_idx:
            return

        has_mask_by_obj_id = {int(obj_id): False for obj_id in enabled_obj_ids}
        for idx, obj_id in enumerate(output.obj_ids):
            obj_id_int = int(obj_id)
            if obj_id_int not in has_mask_by_obj_id or idx >= len(output.masks):
                continue
            has_mask = bool(np.asarray(output.masks[idx]).any())
            has_mask_by_obj_id[obj_id_int] = has_mask
            if has_mask:
                self._propagation_seen_obj_ids.add(obj_id_int)
                self._smart_propagation_waiting_for_recovery_obj_ids.discard(obj_id_int)

        candidate_obj_ids = {
            int(obj_id)
            for obj_id in enabled_obj_ids
            if int(obj_id) not in self._smart_propagation_waiting_for_recovery_obj_ids
        }
        if not candidate_obj_ids:
            return

        decision = detect_smart_propagation_restart(
            enabled_obj_ids=candidate_obj_ids,
            has_mask_by_obj_id=has_mask_by_obj_id,
            previously_seen_obj_ids=self._propagation_seen_obj_ids,
            already_triggered_loss_keys=self._smart_propagation_triggered_loss_keys,
            loss_frame_idx=abs_frame_idx,
            run_start_frame_idx=state.run_start_frame_idx,
            rewind_frames=self.smart_propagation_rewind_frames,
        )
        if not decision.should_restart:
            return

        restart_seed_frame_idx = decision.restart_seed_frame_idx
        loss_frame_idx = decision.loss_frame_idx
        if restart_seed_frame_idx is None or loss_frame_idx is None:
            return
        if restart_seed_frame_idx >= abs_frame_idx:
            return

        self._propagation_smart_restart_requested = True
        state.pending_smart_restart_seed_frame_idx = restart_seed_frame_idx
        state.pending_smart_restart_loss_frame_idx = loss_frame_idx
        state.smart_restart_cooldown_until_frame_idx = loss_frame_idx
        state.next_seed_frame_idx = restart_seed_frame_idx
        state.active_chunk_n_frames = self.smart_propagation_recovery_chunk_size
        for obj_id in decision.lost_obj_ids:
            self._smart_propagation_triggered_loss_keys.add((int(obj_id), loss_frame_idx))
            self._propagation_lost_obj_ids.add(int(obj_id))
            self._smart_propagation_waiting_for_recovery_obj_ids.add(int(obj_id))
        context = self._sam_task_contexts.get(task_id)
        if context is not None:
            context["last_emitted_frame_idx"] = abs_frame_idx
        self._set_status(
            f"Smart propagation triggered at frame {abs_frame_idx + 1}; rewinding to frame {restart_seed_frame_idx + 1}.",
            progress=state.completed_chunks,
            total=max(1, state.total_chunks),
        )
        if self._sam_worker is not None:
            self._sam_worker.cancel_propagation()

    def _handle_smart_propagation_restart_after_stop(self, context: Dict[str, object]) -> None:
        """Clear replayed outputs and relaunch the recovery chunk after cancellation."""
        state = self._pending_propagation
        if state is None:
            self._clear_pending_propagation_state()
            return

        restart_seed_frame_idx = state.pending_smart_restart_seed_frame_idx
        if restart_seed_frame_idx is None:
            self._clear_pending_propagation_state()
            return

        last_emitted_frame_idx = int(context.get("last_emitted_frame_idx", restart_seed_frame_idx))
        enabled_obj_ids = self._propagation_enabled_obj_ids or set()
        self._clear_replayed_propagation_range(
            start_seed_frame_idx=restart_seed_frame_idx,
            end_frame_idx=last_emitted_frame_idx,
            obj_ids=enabled_obj_ids,
        )
        self._propagation_busy = False
        self._set_propagation_ui_enabled(True)
        self._propagation_smart_restart_requested = False
        self._start_propagation_chunk_async(enabled_obj_ids=enabled_obj_ids, state=state)

    def _on_sam_propagate_done(self, task_id: str, last_masked_frame_idx: int, chunk_last_frame_idx: int) -> None:
        """Advance, complete, or resume propagation after a worker run ends normally."""
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
        state.active_chunk_n_frames = state.n_frames
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
        """Handle cancellation for prefetch, auto-step, or manual propagation tasks."""
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
            if kind == "manual" and self._pending_propagation is not None and self._propagation_smart_restart_requested:
                self._handle_smart_propagation_restart_after_stop(context)
            else:
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
        """Surface worker failures, unwind runtime state, and wake any blocked waiters."""
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
        """Record success for fire-and-forget SAM tasks and wake any blocked waiter."""
        self._sam_task_results[task_id] = True
        loop = self._sam_waiting.get(task_id)
        if loop is not None:
            loop.quit()
        self._sam_task_contexts.pop(task_id, None)

    def _frame_output_has_masks(self, output: SamFrameOutput) -> bool:
        """Return whether any object in a frame output still has a non-empty mask."""
        for mask in output.masks:
            if np.asarray(mask).any():
                return True
        return False

    def _refresh_after_prompt_edit(self) -> None:
        """Recompute or clear current-frame output after prompt edits, then prefetch again."""
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
        """Return whether the active object has any point or box prompts on this frame."""
        if self.active_object_id is None:
            return False
        frame_map = self.prompts_by_frame_obj.get(self.current_frame_idx, {})
        if frame_map.get(self.active_object_id):
            return True
        frame_boxes = self.box_prompts_by_frame_obj.get(self.current_frame_idx, {})
        return self.active_object_id in frame_boxes

    def _is_current_box_locked(self, frame_idx: int, obj_id: Optional[int]) -> bool:
        """Return whether one object's box is locked against automatic replacement."""
        if obj_id is None:
            return False
        return self.box_locked_by_frame_obj.get(frame_idx, {}).get(obj_id, False)

    def _set_current_box_locked(self, frame_idx: int, obj_id: int, locked: bool) -> None:
        """Set or clear the stored lock flag for one frame/object pair."""
        frame_map = self.box_locked_by_frame_obj.setdefault(frame_idx, {})
        if locked:
            frame_map[obj_id] = True
            return
        frame_map.pop(obj_id, None)
        if not frame_map:
            self.box_locked_by_frame_obj.pop(frame_idx, None)

    def _clone_frame_output(self, output: Optional[SamFrameOutput]) -> Optional[SamFrameOutput]:
        """Deep-copy a frame output so undo snapshots are isolated from future edits."""
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
        """Capture prompt, box, lock, and output state needed to undo the next edit."""
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
        """Restore the last stashed prompt-edit snapshot for the active object/frame."""
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
        """Remove one object's output from a frame and prune empty frame outputs."""
        if obj_id is None:
            return
        updated_output = remove_object_from_output(self.outputs_by_frame.get(frame_idx), obj_id)
        if updated_output is None:
            if frame_idx not in self.outputs_by_frame:
                return
            self.outputs_by_frame.pop(frame_idx, None)
            self._output_version_by_frame.pop(frame_idx, None)
            return
        self.outputs_by_frame[frame_idx] = updated_output

    def _clear_replayed_propagation_range(
        self,
        *,
        start_seed_frame_idx: int,
        end_frame_idx: int,
        obj_ids: set[int],
    ) -> None:
        """Delete outputs and boxes that will be replayed after a smart rewind."""
        if not obj_ids or end_frame_idx <= start_seed_frame_idx:
            return
        start_frame_idx = max(0, start_seed_frame_idx + 1)
        final_frame_idx = min(end_frame_idx, len(self.frame_paths) - 1)
        for frame_idx in range(start_frame_idx, final_frame_idx + 1):
            frame_boxes = self.box_prompts_by_frame_obj.get(frame_idx)
            if frame_boxes is not None:
                for obj_id in obj_ids:
                    frame_boxes.pop(obj_id, None)
                    self._set_current_box_locked(frame_idx, obj_id, False)
                if not frame_boxes:
                    self.box_prompts_by_frame_obj.pop(frame_idx, None)
            for obj_id in obj_ids:
                self._remove_object_output_from_frame(frame_idx, obj_id)
        if self.current_frame_idx >= start_frame_idx and self.current_frame_idx <= final_frame_idx:
            self._sync_current_box_lock_check()
            self.refresh_point_list()
            self._render_current_frame()

    def _remove_object_annotations_from_frame(self, frame_idx: int, obj_id: Optional[int]) -> None:
        """Remove prompts, box, locks, and output for one object on one frame."""
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
        """Delegate frame-output merge rules to the pure propagation helper module."""
        return merge_frame_outputs(base_output, new_output)

    def _get_enabled_propagation_obj_ids(self) -> set[int]:
        """Return the object ids currently enabled for propagation on this frame."""
        return self._get_enabled_propagation_obj_ids_from_rows()

    def _get_enabled_propagation_obj_ids_from_rows(self) -> set[int]:
        """Resolve propagation enablement from the current object-row widgets."""
        enabled: set[int] = set()
        for obj_id in self._object_row_widgets.keys():
            if self._is_object_enabled_for_current_frame_propagation(int(obj_id)):
                enabled.add(int(obj_id))
        return enabled

    def _carry_prompts_forward(
        self,
        src_frame_idx: int,
        dst_frame_idx: int,
        obj_ids: List[int],
        *,
        translate: bool = True,
    ) -> None:
        """Copy or translate point prompts into the next frame alongside propagated boxes."""
        src_prompts = self.prompts_by_frame_obj.get(src_frame_idx, {})
        if not src_prompts:
            return
        dst_prompts = self.prompts_by_frame_obj.setdefault(dst_frame_idx, {})
        for obj_id in obj_ids:
            points = src_prompts.get(obj_id)
            if not points:
                continue
            if translate and self.translate_prompts_on_propagation:
                dst_prompts[obj_id] = translate_prompts_by_box_delta(
                    points=points,
                    src_box=self.box_prompts_by_frame_obj.get(src_frame_idx, {}).get(obj_id),
                    dst_box=self.box_prompts_by_frame_obj.get(dst_frame_idx, {}).get(obj_id),
                    frame_size=self._get_frame_size(dst_frame_idx),
                )
            else:
                dst_prompts[obj_id] = clone_point_prompts(points)

    def _sync_canonical_boxes_from_output(
        self,
        frame_idx: int,
        output: SamFrameOutput,
        preserve_locked: bool = True,
    ) -> None:
        """Update stored per-frame boxes from SAM output, respecting locks when requested."""
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
        """Convert a normalized xywh box into a clamped pixel-space ``BoxPrompt``."""
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
        """Reset propagation runtime state after a run completes, stops, or fails."""
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
        self._smart_propagation_waiting_for_recovery_obj_ids.clear()
        self._propagation_loss_notified = False
        self._propagation_smart_restart_requested = False
        self._smart_propagation_triggered_loss_keys.clear()
        if hasattr(self, "stop_propagate_btn"):
            self.stop_propagate_btn.setEnabled(False)
        self._schedule_prefetch_for_next_frame()

    def export_annotations(self) -> None:
        """Export the current annotations in either COCO or legacy Perk format."""
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
        """Prompt for a session directory and save the current annotator state there."""
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
        """Prompt for a session directory and load its ``session.json`` payload."""
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
        """Persist the current session snapshot and any research data to disk."""
        if self.image_dir is None:
            return
        SESSION_REPOSITORY.save_session(session_dir, self._build_session_payload())
        self._save_research_data(session_dir)
        self._set_status("Session saved.", progress=1, total=1)

    def _load_session(self, session_path: Path) -> None:
        """Restore a saved session, including objects, prompts, outputs, and research data."""
        session_dir = session_path.parent
        progress_dialog = QProgressDialog("Loading session...", None, 0, 100, self)
        progress_dialog.setWindowTitle("Loading Session")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setAutoClose(True)
        progress_dialog.setAutoReset(True)

        def update_progress(value: int, text: str) -> None:
            """Update the modal load dialog while keeping the Qt event loop responsive."""
            progress_dialog.setLabelText(text)
            progress_dialog.setValue(max(0, min(100, value)))
            QApplication.processEvents()

        progress_dialog.show()
        update_progress(5, "Reading session metadata...")
        payload = SESSION_REPOSITORY.load_session(
            session_path,
            on_output_loaded=lambda idx, total: update_progress(
                6 + int(18 * idx / max(1, total)),
                f"Reading saved outputs... {idx}/{total}",
            ),
        )
        frame_dir = Path(payload.frame_dir)
        if not frame_dir.exists():
            alt_dir = QFileDialog.getExistingDirectory(self, "Select Frame Directory for Session")
            if not alt_dir:
                progress_dialog.close()
                return
            frame_dir = Path(alt_dir)

        frame_files = payload.frame_files
        if not frame_files:
            progress_dialog.close()
            QMessageBox.warning(self, "Invalid session", "No frame list found in session.")
            return

        update_progress(28, "Preparing frame directory...")
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

        self._checkpoint_path = payload.checkpoint_path
        self._set_optional_combo_path(self.checkpoint_combo, self._checkpoint_path)
        self._apply_experimental_settings(payload.experimental_settings)
        update_progress(36, "Initializing SAM worker...")
        self._initialize_sam_worker(show_errors=False)
        self._apply_experimental_settings_live()

        update_progress(44, "Restoring objects...")
        self.objects.clear()
        self.object_list.clear()
        self._object_row_widgets.clear()
        for obj in payload.objects:
            obj = ObjectInfo(
                obj_id=int(obj.obj_id),
                name=obj.name,
                color_bgr=tuple(obj.color_bgr),
            )
            self.objects.append(obj)
            item = QListWidgetItem()
            item.setData(Qt.UserRole, obj.obj_id)
            item.setForeground(QBrush(QColor(obj.color_bgr[2], obj.color_bgr[1], obj.color_bgr[0])))
            self.object_list.addItem(item)
            self.object_list.setItemWidget(item, self._create_object_row_widget(obj.obj_id))

        self.hidden_obj_ids = {int(obj_id) for obj_id in payload.hidden_obj_ids}
        self.solo_object_id = int(payload.solo_object_id) if payload.solo_object_id is not None else None
        self.next_obj_id = 1 + max((obj.obj_id for obj in self.objects), default=0)
        valid_obj_ids = {obj.obj_id for obj in self.objects}
        self.hidden_obj_ids &= valid_obj_ids
        if self.solo_object_id not in valid_obj_ids:
            self.solo_object_id = None
        self.active_object_id = payload.active_object_id
        self.flagged_frame_indices = {
            int(frame_idx)
            for frame_idx in payload.flagged_frames
            if 0 <= int(frame_idx) < len(self.frame_paths)
        }

        update_progress(52, "Restoring prompts...")
        self.prompts_by_frame_obj = {
            int(frame_idx): {
                int(obj_id): list(point_list)
                for obj_id, point_list in per_obj.items()
            }
            for frame_idx, per_obj in payload.prompts_by_frame_obj.items()
        }

        update_progress(60, "Restoring boxes...")
        self.box_prompts_by_frame_obj = {
            int(frame_idx): {
                int(obj_id): BoxPrompt(
                    x1_px=int(box.x1_px),
                    y1_px=int(box.y1_px),
                    x2_px=int(box.x2_px),
                    y2_px=int(box.y2_px),
                )
                for obj_id, box in per_obj.items()
            }
            for frame_idx, per_obj in payload.boxes_by_frame_obj.items()
        }

        update_progress(68, "Restoring box locks...")
        self.box_locked_by_frame_obj = {
            int(frame_idx): {
                int(obj_id): bool(locked)
                for obj_id, locked in per_obj.items()
            }
            for frame_idx, per_obj in payload.box_locks_by_frame_obj.items()
        }

        self.manual_propagation_overrides_by_frame_obj = {
            int(frame_idx): {
                int(obj_id): bool(enabled)
                for obj_id, enabled in per_obj.items()
            }
            for frame_idx, per_obj in payload.propagation_overrides_by_frame_obj.items()
            if per_obj
        }

        update_progress(76, "Restoring masks and outputs...")
        self.outputs_by_frame = {}
        self._output_version_by_frame.clear()
        for frame_idx, output in payload.outputs_by_frame.items():
            frame_idx_int = int(frame_idx)
            self.outputs_by_frame[frame_idx_int] = output
            self._output_version_by_frame[frame_idx_int] = self._prefetch_prompt_version

        if self.outputs_by_frame:
            self.segment_mode = True
            self.mode_label.setText("Mode: Segment" if self._use_point_prompt_mode() else "Mode: Box Annotation")

        update_progress(84, "Restoring view settings...")
        self._apply_view_settings(payload.view_settings)

        update_progress(90, "Restoring propagation settings...")
        self._apply_propagation_settings(payload.propagation_settings)

        self.prompt_mode_combo.setCurrentIndex(int(payload.prompt_mode_index))

        update_progress(96, "Finalizing session...")
        self.current_frame_idx = int(payload.current_frame_idx)
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
        self._load_research_data(session_dir, frame_dir)
        update_progress(100, "Session loaded.")
        progress_dialog.close()

    def _update_research_status_widgets(self) -> None:
        """Refresh status-bar widgets owned by the research controller."""
        self._research_controller.update_status_widgets()

    def _start_new_research_experiment(self, frame_dir: Path, start_paused: bool) -> None:
        """Delegate new-experiment setup to the research controller."""
        self._research_controller.start_new(frame_dir, start_paused=start_paused)

    def _pause_research_experiment(self) -> None:
        """Delegate experiment pause handling to the research controller."""
        self._research_controller.pause()

    def _resume_research_experiment(self) -> None:
        """Delegate experiment resume handling to the research controller."""
        self._research_controller.resume()

    def _toggle_research_visibility(self) -> None:
        """Delegate research-widget visibility toggling to the controller."""
        self._research_controller.toggle_visibility()

    def _save_research_data(self, session_dir: Path) -> None:
        """Persist research telemetry alongside the current annotator session."""
        self._research_controller.save_session(session_dir, image_dir=self.image_dir)

    def _load_research_data(self, session_dir: Path, frame_dir: Path) -> None:
        """Restore research telemetry for a loaded annotator session."""
        self._research_controller.load_session(session_dir, frame_dir=frame_dir)

    def _render_current_frame(self) -> None:
        """Render the current frame plus overlays into the canvas and refresh frame UI."""
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
        """Blend visible segmentation masks from a frame output onto the frame image."""
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
        """Draw temporary masks and boxes for current text-prompt proposals."""
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
        """Draw boxes, labels, handles, and point prompts for the current frame."""
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
        """Return the selected point index if the prompt list selection is a point row."""
        row = self.point_list.currentRow()
        if row < 0 or row >= len(self._active_prompt_rows):
            return None
        prompt_type, idx = self._active_prompt_rows[row]
        if prompt_type != "point":
            return None
        return idx

    def _is_selected_box_prompt(self) -> bool:
        """Return whether the selected prompt-list row represents the object's box."""
        row = self.point_list.currentRow()
        if row < 0 or row >= len(self._active_prompt_rows):
            return False
        prompt_type, _idx = self._active_prompt_rows[row]
        return prompt_type == "box"

    def _is_box_mode(self) -> bool:
        """Return whether the UI is currently in box annotation mode."""
        return not self._use_point_prompt_mode()

    def is_box_mode_active(self) -> bool:
        """Public wrapper used by the canvas widget to query box-mode state."""
        return self._is_box_mode()

    def _begin_box_interaction(
        self,
        x_px: int,
        y_px: int,
    ) -> Tuple[Optional[BoxPrompt], bool]:
        """Resolve a canvas press into new-box, move, resize, or selection-only behavior."""
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
        """Return the nearest resize-handle name, or ``move`` when the point is inside."""
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
        """Return corner and edge handle anchor points for a box."""
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
        """Choose a handle radius scaled to the current box size."""
        box_w = max(1, box.x2_px - box.x1_px)
        box_h = max(1, box.y2_px - box.y1_px)
        min_dim = min(box_w, box_h)
        return max(2, min(6, min_dim // 6))

    def _point_in_box(self, box: BoxPrompt, x_px: int, y_px: int) -> bool:
        """Return whether a pixel lies within the inclusive bounds of a box."""
        return box.x1_px <= x_px <= box.x2_px and box.y1_px <= y_px <= box.y2_px

    def _find_box_at_point(self, x_px: int, y_px: int) -> Optional[Tuple[int, BoxPrompt]]:
        """Return the smallest visible box containing a point for intuitive selection."""
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
        """Build the preview/committed box implied by the current drag state."""
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
        """Resize a reference box according to the active drag handle."""
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
        """Move a box while clamping it inside the current frame bounds."""
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
        """Project a box into canvas coordinates and update the rubber-band preview."""
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
        """Draw a solid or dashed box outline into an image overlay."""
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
        """Render a dashed line segment directly into an image array."""
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
        """Return whether resize handles should be shown for the given object."""
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
        """Draw the resize handles around the selected editable box."""
        handle_radius = self._box_handle_radius_px(box)
        outline_color = (255, 255, 255)
        for cx, cy in self._get_box_handle_points(box).values():
            cv2.circle(img, (cx, cy), handle_radius + 1, outline_color, -1)
            cv2.circle(img, (cx, cy), handle_radius, color, -1)

    def _sync_selected_box_state(self) -> None:
        """Clear stale box selection when the active object/frame no longer matches."""
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
        """Return the in-memory object model for an id, if it exists."""
        for obj in self.objects:
            if obj.obj_id == obj_id:
                return obj
        return None

    def _map_ui_to_image_xy(self, ui_x: float, ui_y: float) -> Optional[Tuple[int, int]]:
        """Map canvas coordinates into image pixels when the pointer is inside the image."""
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
        """Map canvas coordinates into image pixels, clamping positions to image bounds."""
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
        """Project image-space coordinates back into canvas UI coordinates."""
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
        """Compute the displayed image origin after centering and pan offsets."""
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
        """Clamp pan offsets so the frame stays within sensible canvas bounds."""
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
        """Update pan offsets so zoom keeps the chosen image point under the cursor."""
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
        """Reclamp pan and rerender after the window or canvas changes size."""
        super().resizeEvent(event)
        if self.frame_paths:
            self._clamp_pan_offset()
            self._render_current_frame()

    def _get_current_frame_size(self) -> Optional[Tuple[int, int]]:
        """Return the current frame size as ``(width, height)`` pixels."""
        return self._get_frame_size(self.current_frame_idx)

    def _get_frame_size(self, frame_idx: int) -> Optional[Tuple[int, int]]:
        """Read width and height for one frame using OpenCV with a PIL fallback."""
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
        """Read one frame as a BGR array using OpenCV with a PIL fallback."""
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
        """Choose a repeatable display color for an object from the fixed palette."""
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
        """Stop timers and shut down the worker thread before the window closes."""
        try:
            self._autosave_timer.stop()
            if self._sam_thread is not None:
                self._sam_thread.quit()
                self._sam_thread.wait()
        except Exception:
            pass
        super().closeEvent(event)
