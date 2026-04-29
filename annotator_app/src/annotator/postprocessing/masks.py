"""Pure mask post-processing helpers for SAM frame outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from annotator.models import SamFrameOutput


@dataclass
class MaskPostProcessingSettings:
    """Configuration for optional binary-mask cleanup."""

    enabled: bool = False
    remove_small_components: bool = True
    min_component_area_px: int = 50
    fill_holes: bool = False
    max_hole_area_px: int = 100
    morph_open: bool = False
    morph_close: bool = False
    morph_kernel_size: int = 3
    simplify_contours: bool = False
    simplify_epsilon_fraction: float = 0.002


def _normalized_odd_kernel_size(value: int) -> int:
    """Clamp kernel size to a positive odd integer."""
    size = max(1, int(value))
    if size % 2 == 0:
        size += 1
    return size


def remove_small_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """Remove foreground connected components smaller than a pixel-area threshold."""
    mask_bool = np.asarray(mask).astype(bool)
    min_area = max(0, int(min_area_px))
    if min_area <= 1 or not mask_bool.any():
        return mask_bool.copy()

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask_bool.astype(np.uint8),
        connectivity=8,
    )
    cleaned = np.zeros(mask_bool.shape, dtype=bool)
    for label_idx in range(1, num_labels):
        if int(stats[label_idx, cv2.CC_STAT_AREA]) >= min_area:
            cleaned[labels == label_idx] = True
    return cleaned


def fill_small_holes(mask: np.ndarray, max_hole_area_px: int) -> np.ndarray:
    """Fill enclosed background components up to a pixel-area threshold."""
    mask_bool = np.asarray(mask).astype(bool)
    max_area = max(0, int(max_hole_area_px))
    if max_area <= 0 or not mask_bool.any():
        return mask_bool.copy()

    background = (~mask_bool).astype(np.uint8)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        background,
        connectivity=8,
    )
    cleaned = mask_bool.copy()
    height, width = mask_bool.shape[:2]
    for label_idx in range(1, num_labels):
        component = labels == label_idx
        touches_border = (
            component[0, :].any()
            or component[height - 1, :].any()
            or component[:, 0].any()
            or component[:, width - 1].any()
        )
        if touches_border:
            continue
        if int(stats[label_idx, cv2.CC_STAT_AREA]) <= max_area:
            cleaned[component] = True
    return cleaned


def apply_morphology(mask: np.ndarray, *, open_mask: bool, close_mask: bool, kernel_size: int) -> np.ndarray:
    """Apply optional morphological open and close operations to one mask."""
    mask_u8 = np.asarray(mask).astype(np.uint8)
    if not mask_u8.any() or not (open_mask or close_mask):
        return mask_u8.astype(bool)

    size = _normalized_odd_kernel_size(kernel_size)
    kernel = np.ones((size, size), dtype=np.uint8)
    processed = mask_u8
    if open_mask:
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
    if close_mask:
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    return processed.astype(bool)


def simplify_mask_contours(mask: np.ndarray, epsilon_fraction: float) -> np.ndarray:
    """Simplify external mask contours and rasterize them back into a binary mask."""
    mask_bool = np.asarray(mask).astype(bool)
    if not mask_bool.any():
        return mask_bool.copy()

    epsilon_scale = max(0.0, float(epsilon_fraction))
    if epsilon_scale <= 0.0:
        return mask_bool.copy()

    contours, _hierarchy = cv2.findContours(
        mask_bool.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    simplified = np.zeros(mask_bool.shape, dtype=np.uint8)
    for contour in contours:
        if contour.shape[0] < 3:
            cv2.drawContours(simplified, [contour], -1, 1, thickness=-1)
            continue
        perimeter = cv2.arcLength(contour, closed=True)
        epsilon = max(0.0, epsilon_scale * perimeter)
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        if approx.shape[0] < 3:
            cv2.drawContours(simplified, [contour], -1, 1, thickness=-1)
            continue
        cv2.drawContours(simplified, [approx], -1, 1, thickness=-1)
    return simplified.astype(bool)


def process_mask(mask: np.ndarray, settings: MaskPostProcessingSettings) -> np.ndarray:
    """Apply all enabled post-processing operations to one binary mask."""
    processed = np.asarray(mask).astype(bool)
    if not settings.enabled:
        return processed.copy()
    if settings.remove_small_components:
        processed = remove_small_components(processed, settings.min_component_area_px)
    if settings.fill_holes:
        processed = fill_small_holes(processed, settings.max_hole_area_px)
    processed = apply_morphology(
        processed,
        open_mask=settings.morph_open,
        close_mask=settings.morph_close,
        kernel_size=settings.morph_kernel_size,
    )
    if settings.simplify_contours:
        processed = simplify_mask_contours(processed, settings.simplify_epsilon_fraction)
    return processed.astype(bool)


def mask_to_box_xywh_norm(mask: np.ndarray) -> Tuple[float, float, float, float]:
    """Return a normalized xywh box enclosing the positive mask area."""
    mask_bool = np.asarray(mask).astype(bool)
    if mask_bool.ndim < 2 or not mask_bool.any():
        return (0.0, 0.0, 0.0, 0.0)
    height, width = mask_bool.shape[:2]
    ys, xs = np.where(mask_bool)
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())
    return (
        x1 / max(1, width),
        y1 / max(1, height),
        (x2 - x1 + 1) / max(1, width),
        (y2 - y1 + 1) / max(1, height),
    )


def process_frame_output(
    output: SamFrameOutput,
    settings: MaskPostProcessingSettings,
) -> SamFrameOutput:
    """Apply mask post-processing to every object in a frame output."""
    if not settings.enabled:
        return SamFrameOutput(
            obj_ids=list(output.obj_ids),
            masks=[np.asarray(mask).astype(bool).copy() for mask in output.masks],
            boxes_xywh_norm=[tuple(box) for box in output.boxes_xywh_norm],
            scores=list(output.scores),
            tracker_scores=list(output.tracker_scores),
        )

    processed_masks = [process_mask(mask, settings) for mask in output.masks]
    boxes = []
    for idx, mask in enumerate(processed_masks):
        if idx < len(output.boxes_xywh_norm):
            boxes.append(mask_to_box_xywh_norm(mask))
        else:
            boxes.append(mask_to_box_xywh_norm(mask))
    return SamFrameOutput(
        obj_ids=list(output.obj_ids),
        masks=processed_masks,
        boxes_xywh_norm=boxes,
        scores=list(output.scores),
        tracker_scores=list(output.tracker_scores),
    )
