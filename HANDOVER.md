# HANDOVER

## Project
SAM3-assisted surgical video annotation tool for:
- Per-object prompts (positive/negative points + box prompts)
- Segmentation masks + bounding boxes
- Frame-to-frame propagation/tracking (chunked)
- Annotation export in COCO and Perk formats

Repository root for active work: `sam3v2/`

## Current Status
Implemented:
- Plan document: `tools/SURGICAL_ANNOTATOR_PLAN.md`
- PySide6 app: `surgical_annotator_qt.py` (entrypoint at repo root)
- Export module: `tools/exporters/coco_export.py`
- Export module: `tools/exporters/perk_export.py`
- Text-prompt proposal helper: `tools/text_prompt_grounding.py`
- App usage doc: `tools/README_surgical_annotator.md`
- App deps: `requirements-annotator.txt`

## Implemented Behavior
- Load directory of images as frame sequence.
- Object list with stable object IDs and per-object propagate checkbox.
- Active object is visually highlighted in the object list and mirrored in an `Active:` label.
- Selected object rows use white text on a blue highlight for readability.
- `Prop` is resolved per frame: objects with usable current-frame seed data default to checked, unlabeled frames show unchecked/disabled, and manual unchecks are remembered for that specific frame.
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
  - Prompt boxes cannot be edited while `Show Boxes` is off.
  - Undo last prompt edit with `Ctrl+Z` (single-step).
- Prompt box editing:
  - Click box in `Box (drag)` mode to reveal handles
  - Drag corner handle to resize both axes
  - Drag edge handle to resize one axis
  - Drag inside selected box to move it
  - Double-click a visible box outside box mode to activate that object, switch to box mode, and select the box for editing.
- Box locking:
  - Per-object inline `Lock` checkbox in the object row
  - Bulk `Lock All Boxes` control in the prompt panel
- Segment:
  - Runs SAM3 on current frame only (fresh 1-frame PIL session each click).
  - Per-object segmentation results are merged into existing frame outputs instead of replacing other objects.
- Text prompt proposals:
  - A frame-level text prompt field can be used to ask SAM3 for object proposals on the current frame only.
  - `Generate Objects` runs a fresh 1-frame SAM3 session with a text prompt and returns temporary proposals.
  - Proposals are reviewed in a dedicated list and overlaid on the current frame.
  - `Accept Selected` converts chosen proposals into normal annotator objects with:
    - new persistent `obj_id`s
    - names derived from the prompt, e.g. `dog 1`, `dog 2`
    - stored current-frame mask output
    - stored canonical current-frame box derived from the proposal
  - Accepted text-prompt objects then use the normal per-object propagation workflow.
  - Text prompts are not stored as tracked prompts and are not propagated directly.
  - Temporary proposals are not saved unless accepted.
- Propagate (chunked):
  - `N frames` per chunk
  - `Chunks` count
  - `Stop` supports cooperative mid-chunk cancellation between yielded frames.
  - `Pause` control removed.
  - `Propagation Mode`:
    - `Tracker`
    - `Copy Boxes`
  - Chunk 1 seed: current frame.
  - Next chunk seed: last frame with valid masks from previous chunk.
  - Per-object prompt injection behavior:
    - Manual seed-frame points are included when present.
    - Current seed-frame box is included when present.
    - If no current box exists but a stored seed mask exists, propagation falls back to mask-derived AABB carryover.
    - Stored seed mask is passed as extra guidance when available.
    - If an enabled object has no usable current-frame points, box, or stored seed mask, propagation aborts with a warning.
  - If a chunk has no valid masks, propagation stops for manual correction.
  - Only checked objects are included in propagation.
  - Newly propagated object outputs are merged per-frame with existing outputs instead of clearing other objects on overlapping frames.
  - UI advances through propagated frames as outputs arrive.
  - Seed frame is preserved during propagation; the propagated frame-0 result is not written back over the accepted current frame.
  - Carried point prompts are translated forward by box-center displacement only (translation without box-size scaling).
  - Point translation is applied only on propagation, not during same-frame re-segmentation.
  - `Use Point Prompts for Propagation` can disable current-frame point prompts as tracker seed inputs while still allowing boxes and seed masks to drive propagation.
  - Object-row `Prop` state is frame-local: unlabeled frames show unchecked/disabled, labeled frames default back to checked, and a manual uncheck is remembered only for that same frame.
  - Prompt edits invalidate stale next-frame prefetch state for affected objects.
- Control panel:
  - Tabbed layout: `Prompting`, `Segment/Propagate`, and `Experimental Features`.
  - Checkpoint selector moved to `Segment/Propagate`; BPE dropdown removed.
  - `Experimental Features` exposes live SAM3 periodic re-prompt controls for:
    - `Re-prompt Every N Frames`
    - `High Confidence Threshold`
    - `High IoU Threshold`
    - `Use One Session for Chunked Propagation`
  - Auto-save defaults on at 5 minutes once a session folder exists.
  - Help menu includes `Hotkeys List` and `Instructions / Walkthrough`.
  - Status bar includes a progress bar with status text (directory load + propagation) and a live cursor coordinate readout over the image.
- Layout:
  - Image canvas centered in a landscape-format main area.
  - Left review panel contains flagged-frame controls (`Flag / Unflag`, `Prev Flag`, `Next Flag`) and a flagged-frame list.
  - Right control panel has a larger minimum width.
- Overlay rendering:
  - Masks + model boxes + prompt visuals.
  - Independent toggles for `Show Prompts`, `Show Segmentations`, and `Show Boxes`.
  - Adjustable mask opacity and predicted-box line thickness.
  - Prompt boxes render as solid outlines.
  - The currently selected point prompt is outlined in white (1px outside the prompt), and selected boxes get a 1px white outline outside the box perimeter.
  - Point prompts use a smaller radius; negative prompts render as an `X` only (no circle).
  - Box drag preview uses `QRubberBand` (no drag-time rerender zoom).
- Navigation / viewing:
  - Frame slider for scrubbing
  - Custom frame jump control with left/right arrows embedded (no up/down buttons)
  - Hotkeys:
    - Left/Right arrows or `A`/`D` for frame nav
    - `P` for propagate
    - `F` to flag/unflag current frame
    - `Shift+Left` / `Shift+Right` to jump flagged frames
    - `Delete` to remove selected box / point prompt
    - `N` to toggle auto-propagate
    - `M` to cycle propagation mode
  - Mouse wheel zoom
  - Right-click drag pan
  - `Fit to Screen` resets zoom/pan
  - Enter/Escape in text/spin widgets returns focus to the canvas.
- Export:
  - Combined export flow with format chooser.
  - COCO-style JSON + mask PNG files.
  - Perk CSV export with columns `Filename` and `Tool bounding box`, using canonical boxes and Python-literal list-of-dict serialization.
  - Session save/load: JSON + mask PNGs with menu actions; auto-save optional.
  - Tracker-side scores are stored in session outputs for inspection but are not included in export formats yet.

## Memory / GPU Loading Behaviour (important)
SAM3 is constructed on the first successful directory load, not at app startup. After that, later directory loads reuse the same adapter/model instance and only close any active session. All SAM3 calls run on a dedicated `SamWorker` in a single QThread; the UI sends queued tasks and waits for completion only when it must block.

| Action | What happens on GPU |
|---|---|
| First successful **Load frame directory** | Captures checkpoint/BPE paths if provided, constructs `Sam3Adapter`, and loads the SAM3 model onto GPU |
| Later **Load frame directory** actions | Closes any active session, reuses the existing adapter/model, and resets app state for the new frame directory |
| Click **Segment** | Creates a fresh 1-frame PIL session for current frame, injects prompts at session frame 0, returns mask output |
| Click **Generate Objects** | Creates a fresh 1-frame PIL session for current frame, injects a text prompt, returns temporary frame-local proposals |
| Click **Propagate** | By default, each chunk creates a fresh N-frame PIL session from the seed frame; with **Use One Session for Chunked Propagation**, chunked propagation reuses one full-video session and refines the current chunk seed frame inside that persistent session before continuing |
| Change **Experimental Features** values | Updates the already-loaded SAM3 model's periodic re-prompt settings live for future tracker operations |
| Toggle **Use Point Prompts for Propagation** | Includes or excludes current-frame point prompts from tracker seed payloads without changing boxes or seed masks |
| Toggle **Use One Session for Chunked Propagation** | Keeps chunking in the UI but runs tracker propagation inside one persistent full-video session for that propagation run |
| Navigate between labeled and unlabeled frames | `Prop` checkboxes are recomputed from current-frame seed data: unlabeled frames show unchecked/disabled, while labeled frames auto-check again unless that frame was manually unchecked earlier |
| Change directory | Keeps the existing adapter/model alive and closes the current session before resetting annotator state |

Notes:
- No explicit memory cap is set; PyTorch default CUDA allocator is used.
- Negative points use SAM convention `1` foreground / `0` background.
- Box prompts are routed through point prompt API with labels `(2,3)`.
- A single adapter instance is shared by all segment/propagate/prefetch tasks; no parallel SAM3 model instances are created.
- Text prompting reuses the same `SamWorker` / `Sam3Adapter` path as segment/propagate instead of constructing a second SAM3 model instance.
- Experimental periodic re-prompt settings are applied live to the loaded model and persist in session JSON.
- One-session chunked propagation is session-persistent only within a propagation run; a new manual propagation run rebuilds the full-video session from the current seed frame.
- Experimental tracker-input / seed-frame toggle work was removed from the main app and archived in `tools/archive_tracker_seed_experiments.py`.

## How To Run
From `sam3v2/`:

```bash
python3 -m pip install -r requirements-annotator.txt
python3 -m pip install -e .
python3 surgical_annotator_qt.py
```

Syntax check used:

```bash
python3 -m py_compile surgical_annotator_qt.py tools/exporters/coco_export.py tools/exporters/perk_export.py tools/text_prompt_grounding.py
```

## Environment Notes
- SAM3 runtime deps (Torch/CUDA, model access) must be available in active env.
- If checkpoint not provided in UI, SAM3 defaults are used (may require HF access).
- Checkpoint changes only affect the first adapter construction in the current app session; after SAM3 is loaded once, later directory loads reuse the existing model instance.
- Pillow is required (used for single/N-frame PIL sessions).
- Loading a session initializes SAM3 immediately (no need to wait for first segment/propagate).

## Key Files To Read First
1. `surgical_annotator_qt.py`
2. `tools/exporters/coco_export.py`
3. `tools/exporters/perk_export.py`
4. `tools/text_prompt_grounding.py`
5. `tools/README_surgical_annotator.md`
6. `tools/SURGICAL_ANNOTATOR_PLAN.md`
7. `sam3/model/sam3_video_predictor.py`
8. `sam3/model/sam3_video_inference.py`
9. `sam3/model/io_utils.py`

## Key Classes / Methods in `surgical_annotator_qt.py`
| Symbol | Purpose |
|---|---|
| `Sam3Adapter` | Thin wrapper around `Sam3VideoPredictor`; owns session lifecycle |
| `SamWorker` | Owns the single `Sam3Adapter` instance and runs SAM3 tasks in a QThread |
| `AnnotatorMainWindow._initialize_sam_worker()` | Creates the SAM worker thread and waits for adapter init |
| `AnnotatorMainWindow._enqueue_sam_task()` | Sends queued SAM tasks to the worker |
| `AnnotatorMainWindow._wait_for_sam_task()` | UI-thread wait for specific task completion |
| `Sam3Adapter.start_session(resource_path)` | Accepts path string or list of `PIL.Image` |
| `Sam3Adapter.add_text_prompt()` | Runs a frame-local SAM3 text prompt in a 1-frame session |
| `Sam3Adapter.update_experimental_settings()` | Applies periodic re-prompt runtime settings on the loaded SAM3 model |
| `Sam3Adapter.propagate_n_frames(start, max_frames)` | Forward propagation stream; returns session-relative frame outputs |
| `SamWorker._run_propagate()` | Chooses between fresh chunk-local sessions and the experimental one-session chunked propagation path |
| `AnnotatorMainWindow.segment_current_frame()` | Single-frame segmentation path |
| `AnnotatorMainWindow.generate_objects_from_text_prompt()` | Runs frame-local text proposal generation |
| `AnnotatorMainWindow.accept_selected_text_prompt_proposals()` | Converts selected proposals into normal tracked objects |
| `AnnotatorMainWindow._apply_experimental_settings_live()` | Pushes Experimental Features values into the worker/model without rebuilding SAM3 |
| `AnnotatorMainWindow._build_seed_prompts()` | Builds the per-object tracker seed payload; now optionally excludes current-frame point prompts when the propagation toggle is off |
| `AnnotatorMainWindow.propagate_next_frame()` | Chunk controller (stop, propagation mode, per-object filtering) |
| `AnnotatorMainWindow._build_seed_prompts()` | Per-object seed selection; current points/box first, with stored seed mask and AABB fallback when available |
| `AnnotatorMainWindow._merge_frame_outputs()` | Merges per-object outputs into an existing frame result |
| `AnnotatorMainWindow._get_current_frame_tracker_score()` | Returns the current frame's tracker-side score for an object, when available |
| `AnnotatorMainWindow._draw_box_outline()` | Shared solid box renderer for prompt/model boxes |
| `AnnotatorMainWindow.fit_current_frame_to_view()` | Resets zoom/pan viewport state |
| `AnnotatorMainWindow._sample_boxes_from_seed_masks()` | Carryover mask->AABB box sampling |
| `AnnotatorMainWindow._schedule_prefetch_for_next_frame()` | One-frame lookahead propagation prefetch |
| `AnnotatorMainWindow._auto_propagate_next_frame()` | Right-arrow path that waits for prefetch if needed |

## Known Gaps / Next Improvements
- Add test coverage (unit + integration with mocked SAM adapter).
- Improve exporter robustness:
  - ensure mask resize to frame size before writing/area
  - optional polygon/RLE segmentation for downstream tools.
- Consider a dedicated pan/annotate mode or modifier-key panning if right-click pan is not preferred.
- Add multi-step undo/redo.
- Add explicit object removal sync with SAM `remove_object` if needed.
- Consider `offload_video_to_cpu=True` option for large N.
- Consider cache management (`torch.cuda.empty_cache`) if VRAM fragmentation appears.
- Consider moving help text out of modal message boxes into richer docs or an in-app help panel.

## How To Revert Text Prompt Feature
If you want to remove this feature cleanly, revert these additive pieces only:

1. Delete `tools/text_prompt_grounding.py`.
2. Remove the text-prompt imports, worker signal/task branch, and `Sam3Adapter.add_text_prompt()` from `surgical_annotator_qt.py`.
3. Remove the text-prompt UI widgets and methods from `AnnotatorMainWindow`:
   - `text_prompt_input`
   - `generate_text_prompt_btn`
   - `accept_text_prompt_btn`
   - `clear_text_prompt_btn`
   - `text_prompt_list`
   - `_text_prompt_*` state fields and helper methods
   - `_on_sam_text_prompt_done()`
   - `generate_objects_from_text_prompt()`
   - `accept_selected_text_prompt_proposals()`
   - proposal overlay drawing
4. Remove the text-prompt notes from `tools/README_surgical_annotator.md`.
5. Re-run:

```bash
python3 -m py_compile surgical_annotator_qt.py tools/exporters/coco_export.py tools/exporters/perk_export.py
```

Revert scope is intentionally small because the feature does not change exporter schema, propagation API, or session file structure for unaccepted proposals.

## Suggested New-Chat Initialization Prompt
```text
You are working in D:\SAM3Annotator\sam3v2.
Read HANDOVER.md first, then read:
- surgical_annotator_qt.py
- tools/exporters/coco_export.py
- tools/exporters/perk_export.py
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
| 2026-03-06 | SAM3 lazy loading deferred to first successful directory load |
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
| 2026-03-16 | Added session save/load (JSON + masks) and auto-save controls |
| 2026-03-16 | Simplified navigation controls with embedded left/right frame jump arrows |
| 2026-03-16 | Removed advanced prompt toggle; prompt controls always visible |
| 2026-03-16 | Moved checkpoint selector to Segment/Propagate tab; removed BPE selector |
| 2026-03-16 | Updated prompt visuals: smaller points, negative X-only, white outlines for selected prompts/boxes |
| 2026-03-16 | Updated point translation to box-relative mapping (no clamping) and only during propagation |
| 2026-03-16 | Added single-step undo for prompt edits (Ctrl+Z) |
| 2026-03-16 | Added A/D hotkeys for frame navigation |
| 2026-03-16 | Auto-prop/prefetch re-run only forward from last edited frame and skip annotated frames unless prompts changed |
| 2026-03-23 | Removed pause/carryover/tracker-seed experiment UI from the main app; archived experimental tracker seed work in `tools/archive_tracker_seed_experiments.py` |
| 2026-03-23 | Propagation now preserves the accepted seed frame, uses current points/box with optional stored seed mask guidance, and supports cooperative mid-chunk stop |
| 2026-03-23 | Added inline box locking, `Lock All Boxes`, selected-row white text, and automatic disabling of `Prop` when no current-frame propagation seed exists |
| 2026-03-23 | Added double-click box activation outside box mode, focus return from text/spin widgets, and expanded hotkeys/help menu |
| 2026-03-23 | Replaced blank left column with flagged-frame review panel and widened the right control panel |
| 2026-03-24 | Added propagation mode cycling hotkey, default 5-minute auto-save, combined export flow, and Perk CSV export |
| 2026-04-04 | Added frame-local text prompt proposal generation with review/accept flow that converts accepted proposals into normal tracked objects via `tools/text_prompt_grounding.py` |
| 2026-04-04 | Added `Experimental Features` tab and exposed live-editable SAM3 periodic re-prompt settings for interval, confidence, and IoU; values now persist in session JSON |
| 2026-04-05 | Added `Use Point Prompts for Propagation` toggle so current-frame point prompts can be excluded from tracker seeding while boxes and masks still propagate normally |
| 2026-04-05 | Changed object-row `Prop` toggles to frame-local state: unlabeled frames show unchecked/disabled, labeled frames auto-check from current-frame seed data, and manual unchecks persist only on that frame |
| 2026-04-05 | Exposed tracker-side object score as a second per-frame score channel in app outputs, object-row UI, and session save/load without changing export score semantics |
| 2026-04-05 | Added experimental `Use One Session for Chunked Propagation` mode that reuses one full-video SAM3 session across chunks within a propagation run |
Note: the annotator-owned application code has moved to `D:\SAM3Annotator\annotator_app`. Historical paths below that point at `sam3v2/surgical_annotator_qt.py` or `sam3v2/tools/...` describe the old layout unless explicitly updated.
