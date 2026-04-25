"""Helpers for locating and exposing the vendored SAM3 runtime package."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def annotator_workspace_root() -> Path:
    """Return the repository root expected by the annotator-side vendor adapter."""
    return Path(__file__).resolve().parents[4]


def vendor_sam3_root() -> Path:
    """Resolve the vendored SAM3 tree, honoring the optional environment override."""
    override = os.getenv("SAM3_VENDOR_ROOT")
    if override:
        return Path(override).resolve()
    return annotator_workspace_root() / "sam3v2"


def ensure_vendor_sam3_on_path() -> Path:
    """Validate the SAM3 checkout and prepend it to ``sys.path`` for imports."""
    vendor_root = vendor_sam3_root()
    if not vendor_root.exists():
        raise RuntimeError(
            f"Expected vendored SAM3 tree at {vendor_root}. "
            "Set SAM3_VENDOR_ROOT if the runtime lives somewhere else."
        )
    vendor_root_str = str(vendor_root)
    if vendor_root_str not in sys.path:
        sys.path.insert(0, vendor_root_str)
    return vendor_root
