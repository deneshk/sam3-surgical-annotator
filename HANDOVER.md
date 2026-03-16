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
  - Selecting a point row in the annotation list highlights that point on the image with a white outline.
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
  - Carried point prompts are translated forward by the propagated box motion (box-center delta) when both source and destination boxes are available; otherwise they are copied unchanged.
- Control panel:
  - Tabbed layout: `Prompting` and `Segment/Propagate`.
  - Status bar includes a progress bar with status text (directory load + propagation) and a live cursor coordinate readout over the image.
- Layout:
  - Image canvas centered in a landscape-format main area.
  - Left column reserved as a placeholder (currently blank).
- Overlay rendering:
  - Masks + model boxes + prompt visuals.
  - Independent toggles for `Show Prompts`, `Show Segmentations`, and `Show Boxes`.
  - Adjustable mask opacity and predicted-box line thickness.
  - Prompt boxes render as dashed outlines; predicted boxes render as solid outlines.
  - The currently selected point prompt is outlined in white for easier visual identification.
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
SAM3 is constructed on the first successful directory load, not at app startup. After that, later directory loads reuse the same adapter/model instance and only close any active session. All SAM3 calls run on a dedicated `SamWorker` in a single QThread; the UI sends queued tasks and waits for completion only when it must block.

| Action | What happens on GPU |
|---|---|
| First successful **Load frame directory** | Captures checkpoint/BPE paths if provided, constructs `Sam3Adapter`, and loads the SAM3 model onto GPU |
| Later **Load frame directory** actions | Closes any active session, reuses the existing adapter/model, and resets app state for the new frame directory |
| Click **Segment** | Creates a fresh 1-frame PIL session for current frame, injects prompts at session frame 0, returns mask output |
| Click **Propagate** | For each chunk, creates a fresh N-frame PIL session from seed frame, injects per-object prompts at session frame 0 (manual prompts first; on chunk 1, stored seed masks may be converted to AABB box prompts), runs forward propagation, remaps session indices to absolute frame indices |
| Change directory | Keeps the existing adapter/model alive and closes the current session before resetting annotator state |

Notes:
- No explicit memory cap is set; PyTorch default CUDA allocator is used.
- Negative points use SAM convention `1` foreground / `0` background.
- Box prompts are routed through point prompt API with labels `(2,3)`.
- A single adapter instance is shared by all segment/propagate/prefetch tasks; no parallel SAM3 model instances are created.

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
- Checkpoint/BPE changes only affect the first adapter construction in the current app session; after SAM3 is loaded once, later directory loads reuse the existing model instance.
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
| `SamWorker` | Owns the single `Sam3Adapter` instance and runs SAM3 tasks in a QThread |
| `AnnotatorMainWindow._initialize_sam_worker()` | Creates the SAM worker thread and waits for adapter init |
| `AnnotatorMainWindow._enqueue_sam_task()` | Sends queued SAM tasks to the worker |
| `AnnotatorMainWindow._wait_for_sam_task()` | UI-thread wait for specific task completion |
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
| `AnnotatorMainWindow._schedule_prefetch_for_next_frame()` | One-frame lookahead propagation prefetch |
| `AnnotatorMainWindow._auto_propagate_next_frame()` | Right-arrow path that waits for prefetch if needed |

## Known Gaps / Next Improvements
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
| 2026-03-15 | Deferred SAM3 construction until first successful directory load and reused the same adapter/model across later directory changes |
| 2026-03-15 | Added white overlay highlight for the currently selected point prompt in the annotation list |
| 2026-03-15 | Changed prompt carryover during propagation to translate point prompts by propagated box motion instead of keeping them static |
| 2026-03-16 | Moved all SAM3 calls to a single `SamWorker` (QThread) and reused one adapter for segment/propagate/prefetch |
| 2026-03-16 | Added 1-frame prefetch caching for right-arrow navigation and prompt-change-triggered prefetch |
| 2026-03-16 | Tabbed control panel, centered landscape image layout, status bar progress text + cursor coordinates |
