"""Reusable Qt widgets owned by the annotator application.

These widgets keep UI-specific interaction behavior out of the main window so
the window can focus on application orchestration.
"""

from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QWidget,
)

DEFAULT_CANVAS_SIZE = (960, 540)


class ClickableImageLabel(QLabel):
    """Canvas label that translates Qt mouse gestures into annotator callbacks."""

    def __init__(self, parent_window) -> None:
        """Initialize the interactive canvas label and its delayed click handling."""
        super().__init__()
        self.parent_window = parent_window
        self._target_size = DEFAULT_CANVAS_SIZE
        self._pending_left_click_pos: Optional[Tuple[float, float]] = None
        self._pending_left_click_timer = QTimer(self)
        self._pending_left_click_timer.setSingleShot(True)
        self._pending_left_click_timer.timeout.connect(self._commit_pending_left_click)
        self.setAlignment(Qt.AlignCenter)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(*self._target_size)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setScaledContents(False)
        self.setStyleSheet("background-color: #111; border: 1px solid #444; color: #ddd;")
        self.setMouseTracking(True)

    def set_target_size(self, width: int, height: int) -> None:
        """Keep the canvas layout stable while allowing dynamic fit-to-view sizing."""
        width = max(1, int(width))
        height = max(1, int(height))
        self._target_size = (width, height)
        self.setMinimumSize(width, height)
        self.updateGeometry()

    def sizeHint(self):
        """Report the desired canvas size to parent layouts."""
        return QSize(*self._target_size)

    def _cancel_pending_left_click(self) -> None:
        """Discard the deferred single-click gesture when a competing action wins."""
        self._pending_left_click_timer.stop()
        self._pending_left_click_pos = None

    def _commit_pending_left_click(self) -> None:
        """Forward a delayed single click once Qt has ruled out a double click."""
        if self._pending_left_click_pos is None:
            return
        ui_x, ui_y = self._pending_left_click_pos
        self._pending_left_click_pos = None
        self.parent_window.on_image_press(ui_x, ui_y)

    def mousePressEvent(self, event):
        """Start panning, box editing, or delayed point-click handling based on the button."""
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
        """Forward drag and hover updates to the owning window."""
        if event.buttons() & Qt.RightButton:
            self.parent_window.on_pan_drag(event.position().x(), event.position().y())
            return
        self.parent_window.on_image_hover(event.position().x(), event.position().y())
        self.parent_window.on_image_drag(event.position().x(), event.position().y())

    def mouseReleaseEvent(self, event):
        """Finish pan or box-drag gestures on mouse release."""
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
        """Promote the gesture to an explicit double-click action on the canvas."""
        if event.button() != Qt.LeftButton:
            event.ignore()
            return
        self._cancel_pending_left_click()
        self.setFocus(Qt.MouseFocusReason)
        self.parent_window.on_image_double_click(event.position().x(), event.position().y())
        event.accept()

    def leaveEvent(self, _event):
        """Clear pending hover/click state when the pointer leaves the canvas."""
        self._cancel_pending_left_click()
        self.parent_window.on_image_hover(None, None)

    def wheelEvent(self, event):
        """Delegate wheel-based zoom gestures to the owning window."""
        self.parent_window.on_image_wheel(
            event.position().x(),
            event.position().y(),
            event.angleDelta().y(),
        )
        event.accept()


class ArrowSpinBox(QWidget):
    """Compact spinbox with explicit previous/next arrow buttons."""

    valueChanged = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Compose the arrow buttons and wrapped spinbox into one reusable control."""
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
        """Forward minimum-value changes to the wrapped spinbox."""
        self._spin.setMinimum(value)

    def setMaximum(self, value: int) -> None:
        """Forward maximum-value changes to the wrapped spinbox."""
        self._spin.setMaximum(value)

    def setRange(self, minimum: int, maximum: int) -> None:
        """Forward range changes to the wrapped spinbox."""
        self._spin.setRange(minimum, maximum)

    def setValue(self, value: int) -> None:
        """Forward value updates to the wrapped spinbox."""
        self._spin.setValue(value)

    def value(self) -> int:
        """Return the wrapped spinbox value."""
        return self._spin.value()

    def blockSignals(self, block: bool) -> bool:
        """Keep wrapper and inner spinbox signal blocking behavior aligned."""
        self._spin.blockSignals(block)
        return super().blockSignals(block)

    def spinBox(self) -> QSpinBox:
        """Expose the inner spinbox for callers that need lower-level access."""
        return self._spin
