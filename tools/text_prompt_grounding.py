from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class TextPromptProposal:
    proposal_idx: int
    score: float
    mask: np.ndarray
    box_xyxy_px: Tuple[int, int, int, int]


def build_text_prompt_proposals(
    *,
    masks: Sequence[np.ndarray],
    boxes_xywh_norm: Sequence[Tuple[float, float, float, float]],
    scores: Sequence[float],
    frame_size: Tuple[int, int],
) -> List[TextPromptProposal]:
    width, height = frame_size
    proposals: List[TextPromptProposal] = []
    for idx, mask in enumerate(masks):
        if idx >= len(boxes_xywh_norm):
            continue
        mask_bool = np.asarray(mask).astype(bool)
        if mask_bool.size == 0 or not mask_bool.any():
            continue
        x1, y1, x2, y2 = _box_xyxy_from_norm(boxes_xywh_norm[idx], width, height)
        if x2 <= x1 or y2 <= y1:
            continue
        score = float(scores[idx]) if idx < len(scores) else 0.0
        proposals.append(
            TextPromptProposal(
                proposal_idx=idx,
                score=score,
                mask=mask_bool,
                box_xyxy_px=(x1, y1, x2, y2),
            )
        )
    return proposals


def next_prompt_object_name(prompt: str, existing_names: Iterable[str]) -> str:
    stem = " ".join(prompt.strip().split())
    if not stem:
        stem = "Object"
    escaped = re.escape(stem)
    pattern = re.compile(rf"^{escaped}\s+(\d+)$", re.IGNORECASE)
    next_idx = 1
    for name in existing_names:
        stripped = " ".join(str(name).strip().split())
        if stripped.lower() == stem.lower():
            next_idx = max(next_idx, 2)
            continue
        match = pattern.match(stripped)
        if match:
            next_idx = max(next_idx, int(match.group(1)) + 1)
    return f"{stem} {next_idx}"


def _box_xyxy_from_norm(
    box_xywh_norm: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x_norm, y_norm, w_norm, h_norm = box_xywh_norm
    x1 = int(round(float(x_norm) * width))
    y1 = int(round(float(y_norm) * height))
    box_w = int(round(float(w_norm) * width))
    box_h = int(round(float(h_norm) * height))
    x2 = x1 + box_w
    y2 = y1 + box_h
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    return x1, y1, x2, y2
