"""Low-level JSON and mask image I/O helpers for session persistence."""

import json
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np


def write_session_json(path: Path, data: Dict[str, Any]) -> None:
    """Write the session metadata JSON, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_session_json(path: Path) -> Dict[str, Any]:
    """Read the raw session metadata JSON from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_mask_png(path: Path, mask: np.ndarray) -> None:
    """Persist a binary mask as an 8-bit PNG compatible with the session schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_u8 = (mask.astype(np.uint8) * 255) if mask.dtype != np.uint8 else mask
    cv2.imwrite(str(path), mask_u8)


def read_mask_png(path: Path) -> np.ndarray:
    """Load a persisted mask PNG back into the repository's boolean mask shape."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(path))
    return img > 0
