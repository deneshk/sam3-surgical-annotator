"""Disk I/O boundary for annotator sessions.

The repository reads and writes session files while leaving schema mapping to
`session_mapper`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2

from annotator.persistence.session_io import (
    read_mask_png,
    read_session_json,
    write_mask_png,
    write_session_json,
)
from annotator.persistence.session_mapper import data_to_payload, payload_to_data
from annotator.persistence.session_models import SessionPayload


class SessionRepository:
    """Repository boundary for reading and writing annotator sessions on disk."""

    def save_session(self, session_dir: Path, payload: SessionPayload) -> None:
        """Write session JSON and any referenced mask assets to disk."""
        data, mask_assets = payload_to_data(payload)
        for asset in mask_assets:
            write_mask_png(session_dir / asset.relative_path, asset.mask)
        write_session_json(session_dir / "session.json", data)

    def load_session(
        self,
        session_path: Path,
        *,
        on_output_loaded: Optional[Callable[[int, int], None]] = None,
    ) -> SessionPayload:
        """Load session JSON from disk and resolve any relative mask paths."""
        data = read_session_json(session_path)
        session_dir = session_path.parent
        frame_dir = Path(str(data.get("frame_dir", "")))
        frame_files = [str(name) for name in data.get("frame_files", [])]
        frame_paths = [frame_dir / name for name in frame_files]

        def load_mask(mask_path: str):
            """Resolve a saved mask path relative to the session directory and load it."""
            path = Path(mask_path)
            if not path.is_absolute():
                path = session_dir / path
            return read_mask_png(path)

        def fallback_shape_for_frame(frame_idx: int) -> Tuple[int, int]:
            """Provide a best-effort frame shape when a saved mask asset is missing."""
            if 0 <= frame_idx < len(frame_paths):
                image = cv2.imread(str(frame_paths[frame_idx]), cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    return (int(image.shape[0]), int(image.shape[1]))
            return (1, 1)

        return data_to_payload(
            data,
            load_mask=load_mask,
            fallback_shape_for_frame=fallback_shape_for_frame,
            on_output_loaded=on_output_loaded,
        )
