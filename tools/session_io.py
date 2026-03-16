import json
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np


def write_session_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_session_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_mask_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_u8 = (mask.astype(np.uint8) * 255) if mask.dtype != np.uint8 else mask
    cv2.imwrite(str(path), mask_u8)


def read_mask_png(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(path))
    return img > 0
