# HANDOVER

## Project
SAM3-assisted surgical video annotation tool for:
- Per-object prompts (positive/negative points + box prompts)
- Segmentation masks + bounding boxes
- Frame-to-frame propagation/tracking (chunked)
- COCO-style export with mask PNGs

Repository root for active work: `sam3v2/`

## Current Status
Implemented:
- Plan document: `tools/SURGICAL_ANNOTATOR_PLAN.md`
- PySide6 app: `surgical_annotator_qt.py` (entrypoint at repo root)
- Export module: `tools/exporters/coco_export.py`
- App usage doc: `tools/README_surgical_annotator.md`
- App deps: `requirements-annotator.txt`

## Implemented Behavior
- Load directory of images as frame sequence.
- Object list with stable object IDs and per-object propagate checkbox.
- Active object is visually highlighted in the object list and mirrored in an `Active:` label.
- Prompt modes:
  - Positive point (`label=1`)
  - Negative point (`label=0`)
  - Box prompt via drag; internally sent as point prompts with labels `(2,3)`.
- Prompt editing:
  - Remove selected prompt
  - Clear active object prompts
  - Segment mode auto-refreshes on prompt edits
  - If the last prompt/box for an object is removed on the current frame, that object's stored segmentation/derived box is removed from that frame.
  - Prompt boxes cannot be edited while `Show Prompts` is off.
- Prompt box editing:
  - Click box in `Box (drag)` mode to reveal handles
  - Drag corner handle to resize both axes
  - Drag edge handle to resize one axis
  - Drag inside selected box to move it
- Segment:
  - Runs SAM3 on current frame only (fresh 1-frame PIL session each click).
  - Per-object segmentation results are merged into existing frame outputs instead of replacing other objects.
- Propagate (chunked):
  - `N frames` per chunk
  - `Chunks` count
  - `Pause Between Chunks` toggle
  - `Carryover` mode:
    - `Sample Points` (user-controlled count, 1..8)
    - `Mask AABB Box` (mask->axis-aligned box, injected as labels `(2,3)`)
  - Chunk 1 seed: current frame.
  - Next chunk seed: last frame with valid masks from previous chunk.
  - Per-object prompt injection behavior:
    - First chunk:
      - If object has manual prompts/box on seed frame, use those.
      - Otherwise, if object has an existing stored mask on seed frame, convert that mask to an AABB box prompt and use that.
      - If any enabled object has neither prompts nor an existing mask on the seed frame, propagation aborts with a warning.
    - Later chunks:
      - If object has manual prompts on seed frame, use those.
      - Otherwise, use carryover prompt generation for that object.
  - If a chunk has no valid masks, propagation stops for manual correction.
  - Only checked objects are included in propagation.
  - Newly propagated object outputs are merged per-frame with existing outputs instead of clearing other objects on overlapping frames.
  - Viewer stays on the current frame during propagation; propagation no longer jumps to the last masked frame.
- Overlay rendering:
  - Masks + model boxes + prompt visuals.
  - Independent toggles for `Show Prompts`, `Show Segmentations`, and `Show Boxes`.
  - Adjustable mask opacity and predicted-box line thickness.
  - Prompt boxes render as dashed outlines; predicted boxes render as solid outlines.
  - Box drag preview uses `QRubberBand` (no drag-time rerender zoom).
- Navigation / viewing:
  - Frame slider for scrubbing
  - `Go to` spinbox for direct frame jump
  - Hotkeys: Left/Right arrows for frame nav, `P` for propagate
  - Mouse wheel zoom
  - Right-click drag pan
  - `Fit to Screen` resets zoom/pan
- Export:
  - COCO-style JSON + mask PNG files.

## Memory / GPU Loading Behaviour (important)
SAM3 is lazily loaded; model weights are not put on GPU at directory load time.

| Action | What happens on GPU |
|---|---|
| Load frame directory | Nothing; only frame file paths stored, checkpoint/BPE paths captured for later |
| Click **Segment** | If needed, loads weights and constructs `Sam3Adapter`. Creates a fresh 1-frame PIL session for current frame, injects prompts at session frame 0, returns mask output |
| Click **Propagate** | If needed, loads weights first. For each chunk, creates a fresh N-frame PIL session from seed frame, injects per-object prompts at session frame 0 (manual prompts first; on chunk 1, stored seed masks may be converted to AABB box prompts), runs forward propagation, remaps session indices to absolute frame indices |
| Change directory | Resets `sam_adapter` to `None`; weights reload on next Segment/Propagate |

Notes:
- No explicit memory cap is set; PyTorch default CUDA allocator is used.
- Negative points use SAM convention `1` foreground / `0` background.
- Box prompts are routed through point prompt API with labels `(2,3)`.

## How To Run
From `sam3v2/`:

```bash
python3 -m pip install -r requirements-annotator.txt
python3 -m pip install -e .
python3 surgical_annotator_qt.py
```

Syntax check used:

```bash
python3 -m py_compile surgical_annotator_qt.py tools/exporters/coco_export.py
```

## Environment Notes
- SAM3 runtime deps (Torch/CUDA, model access) must be available in active env.
- If checkpoint/BPE not provided in UI, SAM3 defaults are used (may require HF access).
- Pillow is required (used for single/N-frame PIL sessions).

## Key Files To Read First
1. `surgical_annotator_qt.py`
2. `tools/exporters/coco_export.py`
3. `tools/README_surgical_annotator.md`
4. `tools/SURGICAL_ANNOTATOR_PLAN.md`
5. `sam3/model/sam3_video_predictor.py`
6. `sam3/model/sam3_video_inference.py`
7. `sam3/model/io_utils.py`

## Key Classes / Methods in `surgical_annotator_qt.py`
| Symbol | Purpose |
|---|---|
| `Sam3Adapter` | Thin wrapper around `Sam3VideoPredictor`; owns session lifecycle |
| `Sam3Adapter.start_session(resource_path)` | Accepts path string or list of `PIL.Image` |
| `Sam3Adapter.propagate_n_frames(start, max_frames)` | Forward propagation stream; returns session-relative frame outputs |
| `AnnotatorMainWindow.segment_current_frame()` | Single-frame segmentation path |
| `AnnotatorMainWindow.propagate_next_frame()` | Chunk controller (pause/resume, carryover mode, per-object filtering) |
| `AnnotatorMainWindow._apply_prompts_for_seed()` | Per-object prompt selection; chunk 1 can fall back to stored seed masks -> AABB boxes |
| `AnnotatorMainWindow._merge_frame_outputs()` | Merges per-object outputs into an existing frame result |
| `AnnotatorMainWindow._draw_box_outline()` | Shared solid/dashed box renderer for predicted vs prompt boxes |
| `AnnotatorMainWindow.fit_current_frame_to_view()` | Resets zoom/pan viewport state |
| `AnnotatorMainWindow._sample_prompts_from_seed_masks()` | Carryover point sampling |
| `AnnotatorMainWindow._sample_boxes_from_seed_masks()` | Carryover mask->AABB box sampling |

## Known Gaps / Next Improvements
- Move SAM calls off UI thread (worker/QThread) to avoid UI freezes.
- Add test coverage (unit + integration with mocked SAM adapter).
- Improve exporter robustness:
  - ensure mask resize to frame size before writing/area
  - optional polygon/RLE segmentation for downstream tools.
- Consider a dedicated pan/annotate mode or modifier-key panning if right-click pan is not preferred.
- Add optional undo/redo.
- Add explicit object removal sync with SAM `remove_object` if needed.
- Consider `offload_video_to_cpu=True` option for large N.
- Consider cache management (`torch.cuda.empty_cache`) if VRAM fragmentation appears.

## Suggested New-Chat Initialization Prompt
```text
You are working in D:\SAM3Annotator\sam3v2.
Read HANDOVER.md first, then read:
- surgical_annotator_qt.py
- tools/exporters/coco_export.py
- tools/README_surgical_annotator.md

Constraints:
- Prefer adding new files.
- Do not modify unrelated existing tracked files unless explicitly required.

Before edits, summarize current behavior and your exact implementation plan.
```

## Changelog
| Date | Change |
|---|---|
| 2026-03-06 | App entry point moved to `sam3v2/surgical_annotator_qt.py` |
| 2026-03-06 | SAM3 lazy loading deferred to first Segment/Propagate |
| 2026-03-06 | Segment switched to 1-frame PIL session |
| 2026-03-06 | Initial N-frame propagation flow added |
| 2026-03-07 | Added chunked propagation controls (`N`, `Chunks`, pause/resume) |
| 2026-03-07 | Added carryover modes: sampled points and mask->AABB box |
| 2026-03-07 | Added box-draw prompt mode; internally mapped to point labels `(2,3)` |
| 2026-03-07 | Added per-object propagation checkbox filtering |
| 2026-03-07 | Fixed drag-time zoom by using `QRubberBand` preview |
| 2026-03-07 | Updated image label sizing behavior to stabilize fit in splitter |
| 2026-03-07 | Fixed multi-object carryover prompt injection (per-object fallback) |
| 2026-03-12 | Added independent prompt/segmentation/box visibility toggles plus mask opacity and box thickness controls |
| 2026-03-12 | Added prompt-box editing (select, handle-based resize, move) and removed stale per-object output when last prompt is deleted |
| 2026-03-12 | Differentiated prompt vs predicted boxes with dashed vs solid rendering |
| 2026-03-12 | Added zoom, right-click pan, fit-to-screen reset, frame slider, frame jump, and hotkeys |
| 2026-03-12 | Propagation now preserves current view frame, merges per-object outputs, and allows chunk-1 fallback from stored seed masks to AABB box prompts |
