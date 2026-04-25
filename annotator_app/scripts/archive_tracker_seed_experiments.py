#!/usr/bin/env python3
"""Archived reference for removed tracker-input and seed-frame experiments.

This file is intentionally not imported by the main app. It preserves the
experimental UI/state concepts that were temporarily added to
`surgical_annotator_qt.py`:

- tracker input modes:
  - "Box Only"
  - "Mask Only"
  - "Box + Mask"
  - "Points + Box"
  - "Points + Mask"
  - "Points + Box + Mask"
- seed-frame behaviors:
  - "Preserve Manual"
  - "Recompute"

The main app no longer exposes these controls. Propagation now uses one fixed
strategy instead:

- include manual seed-frame points when present
- include the current box, or an AABB fallback derived from the seed mask
- include the stored seed mask as extra guidance when available
- preserve the accepted seed frame instead of overwriting it during propagation

The snippets below are kept for future reference only.
"""

TRACKER_INPUT_MODES = [
    "Box Only",
    "Mask Only",
    "Box + Mask",
    "Points + Box",
    "Points + Mask",
    "Points + Box + Mask",
]

SEED_FRAME_BEHAVIORS = ["Preserve Manual", "Recompute"]


ARCHIVE_NOTES = """
Former main-app behavior summary:

1. The propagation UI exposed a tracker-input dropdown and a seed-frame
   overwrite dropdown.
2. `_build_seed_prompts(...)` switched validation and prompt construction based
   on the selected tracker mode.
3. `Mask Only` used a dedicated predictor path (`add_mask`) rather than the
   normal prompt path.
4. The manual propagation writeback path used a per-task `preserve_seed_frame`
   flag to decide whether to ignore the propagated seed-frame result.

This file exists so the experimental combinations can be revived later without
having to reconstruct the mode list and intent from git history.
"""

