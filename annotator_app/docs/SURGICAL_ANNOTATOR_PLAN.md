# SAM3 Surgical Video Annotator (PySide6) — Decision-Complete Implementation Plan

## Summary
Build a new standalone PySide6 desktop application on top of `sam3.model.sam3_video_predictor.Sam3VideoPredictor` for frame-directory surgical tool annotation with:
- Object-centric positive/negative point prompting (max 6 per active object).
- Per-frame segmentation preview (`Segment`) with mask + bbox overlays.
- Interactive refinement in segment mode (auto-update after add/remove point).
- One-step forward tracking (`Propagate`) to next frame with auto-advance.
- Export to COCO-style JSON plus per-object/per-frame mask PNGs.
- Bounding boxes derived from SAM mask outputs.

## Scope
- In scope:
  - New app UI and controller logic.
  - SAM3 session lifecycle (start/reset/close).
  - Prompt management and point removal.
  - Segmentation and one-step propagation workflow.
  - Export pipeline.
- Out of scope for v1:
  - Multi-user/collaborative editing.
  - Full-video batch propagation UI workflow.
  - Manual bbox drawing/editing.
  - Undo/redo history beyond prompt removal actions.

## Key Decisions Locked
- UI framework: **PySide6** (user preference).
- Object management: **manual object list with stable object IDs**.
- Export: **COCO-style JSON + mask PNG files**.
- Propagate behavior: **next frame only + auto-advance**.
- Segment mode behavior: **auto-segment on each point add/remove**.
- BBoxes: **derived from SAM mask outputs**.

## Proposed File Additions
- `annotator_app/src/annotator/main_window.py`
  - Main app entrypoint and all PySide6 UI/controller code.
- `annotator_app/src/annotator/exporters/coco_export.py`
  - COCO JSON + mask PNG export helpers.
- `annotator_app/docs/README_surgical_annotator.md`
  - Setup/run instructions, expected checkpoint/auth behavior, shortcuts.
- `sam3v2/requirements-annotator.txt`
  - App-specific deps (PySide6, numpy, pillow, opencv-python, pycocotools optional).

## Runtime/Dependency Plan
- App assumes local Python env with:
  - `PySide6`, `numpy`, `opencv-python`, `Pillow`.
  - `torch` + `sam3` package installed and importable.
- Model loading behavior:
  - UI accepts optional local checkpoint path + BPE path.
  - If omitted, SAM3 default HF download path is used (requires prior HF auth/access).
- Startup validation:
  - Explicit dependency import checks with actionable error dialogs.
  - GPU availability warning if CUDA unavailable (continue if SAM3 supports fallback; otherwise hard-fail with message).

## Architecture
- `AnnotatorMainWindow (QMainWindow)`:
  - Left panel: frame viewer (custom clickable canvas), overlay rendering.
  - Right panel: object list, prompt mode controls, point list, action buttons.
  - Top toolbar: load directory, model settings, frame navigation.
  - Bottom status bar: session/model/frame state.
- `AppState` (in-memory model):
  - `image_dir`, `frame_paths`, `current_frame_idx`
  - `objects: list[ObjectInfo{id:int,name:str,color:tuple}]`
  - `active_object_id`
  - `prompts_by_frame_obj: dict[int, dict[int, list[PointPrompt]]]`
  - `outputs_by_frame: dict[int, SamFrameOutput]`
  - `segment_mode: bool`
  - `session_id`, `predictor_initialized: bool`
- `Sam3Adapter`:
  - Wraps SAM3 request API:
    - `start_session(image_dir)`
    - `add_object_points(frame_idx,obj_id,points,labels)`
    - `propagate_next(frame_idx)`
    - `reset_session()`, `close_session()`
  - Handles coordinate normalization/typing and response parsing.
- `OverlayRenderer`:
  - Draws masks (alpha), bboxes, object labels, points (+/-).
  - Uses object color map for consistent cross-frame identity.

## Data Models / Interfaces
- `PointPrompt`:
  - `x_px: int`, `y_px: int`, `is_positive: bool`
- `SamFrameOutput` (normalized internal format):
  - `obj_ids: list[int]`
  - `masks_bool_hwc: list[np.ndarray[bool]]`
  - `boxes_xywh_norm: list[tuple[float,float,float,float]]`
  - `scores: list[float]`
- Export schema:
  - COCO `images`, `annotations`, `categories`.
  - `segmentation`: compressed RLE or external mask reference (final choice: external PNG path + bbox/area in COCO annotation metadata extension).
  - `bbox`: `[x, y, w, h]` in absolute pixels.
  - `category_id`: from object ID mapping created in session.

## UX/Workflow Spec
1. User loads frame directory.
2. App creates SAM3 session against directory.
3. User adds named objects in right panel; first object auto-active.
4. User selects prompt polarity (positive/negative), clicks image to add point for active object.
5. Max 6 points enforced per active object per frame.
6. User can remove:
   - Selected point from point list, or
   - Last point, or
   - Clear all points for active object on current frame.
7. `Segment` button:
   - Runs `add_prompt` for each object with points on current frame.
   - Stores/refreshes `outputs_by_frame[current_frame]`.
   - Enables `segment_mode = true`.
8. In segment mode, each point add/remove triggers immediate re-segmentation for current frame.
9. `Propagate` button:
   - Calls forward propagate with `start_frame=current`, bounded to next frame.
   - Stores returned outputs (current/next if present).
   - Auto-advances to next frame and displays predicted masks/boxes.
10. User repeats refine -> propagate across video.
11. Export writes annotations + masks.

## SAM3 API Usage Contract
- Initialize predictor once per app session.
- Start session request:
  - `type="start_session", resource_path=<image_dir>`.
- Segmentation/refinement per object:
  - `type="add_prompt", frame_index=<idx>, points=<rel coords>, point_labels=<0/1>, obj_id=<stable id>`.
- Propagation:
  - stream request `type="propagate_in_video", propagation_direction="forward", start_frame_index=<current>, max_frame_num_to_track=1`.
- Parse response outputs:
  - `out_obj_ids`, `out_binary_masks`, `out_boxes_xywh`, `out_probs`.

## Edge Cases / Failure Modes
- Empty/invalid directory or unsupported file extension.
- No objects created before clicking image.
- No active object selected.
- Attempting >6 points for active object on current frame.
- Segment clicked with no points on any object for current frame.
- Propagate on final frame (no next frame available).
- SAM response missing expected object IDs/masks (graceful warning + keep prior view).
- Session expiry/errors (attempt auto-reset, otherwise prompt reload).
- Export with sparse annotations (allow; only export annotated frames/objects).

## Performance/Responsiveness Plan
- SAM calls executed in worker thread (`QThread`/`QRunnable`) to keep UI responsive.
- UI shows busy indicator and disables conflicting controls during in-flight requests.
- Rendering uses cached frame image + overlay layers to avoid full recompute when possible.

## Testing Plan
- Unit tests (pure Python components):
  - Coordinate conversions (px <-> relative).
  - Point cap enforcement and removal logic.
  - Output parsing and bbox conversion.
  - Export schema validation (required COCO keys present).
- Integration tests (mocked `Sam3Adapter`):
  - Load dir -> add object -> add points -> segment -> propagate -> auto-advance.
  - Multi-object segmentation same frame.
  - Segment mode auto-refresh on add/remove.
  - End-of-video propagate behavior.
- Manual QA scenarios:
  - Typical 20–50 frame sequence with 2–4 tools.
  - Dense overlaps to verify color/object consistency.
  - Recovery after model error and reloading session.
  - Export inspected with downstream visualization script.

## Acceptance Criteria
- User can complete full cycle: load frames, create objects, add/remove +/- points, segment, refine, propagate next frame, repeat.
- Max-6 prompt rule enforced exactly per active object per frame.
- Masks and derived boxes render per object on current frame.
- Propagate auto-advances and displays next-frame predictions.
- Export creates valid COCO-style JSON and corresponding mask PNG files.
- No UI freeze during SAM inference actions.

## Assumptions
- User will install required runtime deps (`PySide6`, `torch`, `sam3`, etc.) before running.
- SAM3 checkpoint access is available via local path or authenticated HF account.
- Frame directory uses image files in sort-able temporal order (lexicographic filename order).
- v1 category semantics map one object instance to one category entry (object-name-as-category).
