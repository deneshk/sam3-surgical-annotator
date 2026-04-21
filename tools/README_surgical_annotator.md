# SAM3 Surgical Annotator (PySide6)

## What it does
- Loads a directory of image frames as a video sequence.
- Lets you create objects and set one active object at a time.
- Each object has a checkbox in the object list to enable/disable propagation for that object.
- Each object row includes `Prop`, `Rename`, `Solo`, and `Hide` controls.
- Object rows show the current frame's tracker-side score when available as `trk=...`.
- Frame-level text prompts can propose multiple instances on the current frame; accepted proposals are converted into normal tracked objects.
- Uses a single canonical box per object/frame in the main workflow; drawn, propagated, and edited boxes are the same UI annotation.
- Newly drawn boxes and unlocked box edits are refined on the current frame with SAM3; the visible box is replaced by the box derived from the segmentation.
- `Lock Current Box` preserves a manual box on the current frame and prevents SAM3 from overwriting it there.
- The current frame annotation list stays visible in the main UI, including the box plus any point prompts for the active object.
- View controls now live in the `Prompting` tab, including prompt/segmentation/box visibility, box titles, mask opacity, and box thickness.
- An `Experimental Features` tab exposes SAM3 periodic re-prompt controls for frame interval, confidence threshold, IoU threshold, an experimental one-session chunked propagation toggle, and smart propagation rewind controls.
- Clicks add positive/negative point prompts (up to 6) for active object on current frame.
- Box prompts can be drawn by selecting `Box (drag)` and dragging on the image.
- Existing box prompts can be edited while `Box (drag)` mode is active:
  - click the box to reveal resize handles on corners and edge midpoints
  - drag a corner handle to resize both axes
  - drag an edge handle to resize one axis
  - drag inside the already-selected box to move it
- In the default workflow, boxes are the primary visible annotation surface; masks remain background data.
- Use the mouse wheel to zoom the image view.
- Use right-click drag to pan the zoomed image view.
- Use `Fit to Screen` to reset zoom and pan to the default fitted view.
- Use the frame slider to scrub through the loaded frames.
- Use the frame jump control (left/right arrows + field) to jump directly to a frame number.
- Hotkeys: Left/Right arrows or `A`/`D` move frame-by-frame, `P` starts propagation.
- `Auto Propagate Next Frame` can propagate checked objects one frame ahead when the right arrow is used.
- `Use Point Prompts for Propagation` controls whether current-frame point prompts are included as tracker seed inputs; boxes and seed masks still propagate when it is off.
- `Use One Session for Chunked Propagation` keeps manual chunking in the UI while reusing one full-video SAM3 session across chunks for that propagation run.
- `Enable Smart Propagation` is available for target-frame tracker runs. When an enabled object disappears after being tracked earlier in the run, propagation rewinds a few frames and launches one larger recovery chunk before returning to the normal chunk size.
  - After a rewind, the same object will not trigger another smart rewind until it is seen with a non-empty mask again.
- `Prop` is frame-local: unlabeled frames show unchecked/disabled, labeled frames auto-check again, and manual unchecks are remembered on that frame.
- Internally, each box prompt is passed to SAM3 as two point prompts with labels `(2, 3)`.
- View controls let you independently show/hide prompts, segmentations, and derived boxes.
- Box titles can be shown/hidden independently from box outlines.
- Segmentation opacity is adjustable from `0.00` to `1.00`.
- Bounding box line thickness is adjustable from `1` to `10` pixels.
- `Segment` runs SAM3 segmentation for prompted objects on current frame.
- In segment mode, point add/remove auto-refreshes segmentation.
- `Propagate` runs SAM3 forward tracking and advances the UI through frames as masks arrive.
- `Propagate` supports chunked tracking with:
  - `N frames` per chunk
  - `Chunks` count for manual chunk mode
  - Optional `Use Target Frame` to choose the last frame; chunk count is computed using overlap (`N-1` stride)
  - `Sample pts` carryover control (1-8 positive points sampled per object from seed masks)
  - `Carryover` mode:
    - `Sample Points`
    - `Mask AABB Box` (converts last mask to axis-aligned box carryover)
  - optional `Pause Between Chunks` review mode
  - `Auto-pause if Class Lost` to pause and notify when a class loses its mask
  - `Pause` pauses after the current chunk
  - `Stop` cancels propagation immediately
- Exports COCO-style JSON and mask PNG files.
- Save/Load session files (JSON + mask PNGs) with optional auto-save.
- Optional `--research-mode` tracks live experiment time plus left-click interaction events and saves them to `research.json`.

## Install
From `sam3v2/`:

```bash
python3 -m pip install -r requirements-annotator.txt
python3 -m pip install -e .
```

You also need a working SAM3 environment with PyTorch/CUDA compatible with your setup.

## Run
From `sam3v2/`:

```bash
python3 surgical_annotator_qt.py
```

Research mode:

```bash
python3 surgical_annotator_qt.py --research-mode
```

## Usage
1. `Load Frame Directory` from menu.
2. Add objects on the right panel.
   - The currently active object is highlighted and shown below the object list.
   - Use the row-level `Prop` checkbox to include/exclude an object from propagation.
   - Use the row-level `Rename`, `Solo`, and `Hide` controls to manage the object directly from its row.
3. Draw a box for the active object on the current frame.
   - Click the existing box to reveal handles, then drag a handle to resize or drag inside the selected box to move it.
   - Unlocked boxes are immediately refined by SAM3 on the current frame.
   - Lock the current box when you want to keep a manual edit without SAM3 overwrite.
   - The annotation list in the main panel shows the active object's box and any point prompts on the current frame.
4. Optionally enable `Auto Propagate Next Frame`.
   - With the toggle on, pressing the right arrow attempts a one-frame propagation for checked objects before moving to the next frame.
5. Use the `View` controls to toggle segmentations/boxes and tune mask opacity or box thickness.
6. Set `Checkpoint` (optional), `N frames`, and either `Chunks` or `Use Target Frame` + `To frame`, plus `Sample pts`, `Carryover` mode, and optional `Pause Between Chunks` when using manual propagation.
7. Optionally adjust `Experimental Features` values.
   - These settings apply live to future tracker operations.
   - `Re-prompt Every N Frames = 0` disables periodic re-prompting.
   - `Use One Session for Chunked Propagation` changes how tracker chunk propagation reuses SAM3 session state across chunk boundaries.
   - `Enable Smart Propagation` only applies to tracker propagation when `Use Target Frame` is on, and it is disabled while one-session chunked propagation is enabled.
8. Click `Propagate`.
    - Only checked objects are included in propagation.
   - The canonical box on the seed frame is used as the primary propagation input.
   - Use `Pause` to stop after the current chunk or `Stop` to cancel immediately.
   - Enable `Auto-pause if Class Lost` to pause after a class loses its mask and show a warning.
9. If pause is enabled and chunks remain, review/correct and click `Continue Propagate`.
10. Use `Save Session` to persist your work (and enable auto-save if desired).
11. `Export COCO` from menu when done.
12. In `--research-mode`, use the status-bar timer controls to pause, resume, or hide the live experiment timer.

Text prompt workflow:
- Enter a frame text prompt such as `dog`.
- Click `Generate Objects` to produce temporary proposals on the current frame only.
- Review the proposal list and click `Accept Selected` to convert chosen proposals into regular objects named from the prompt, such as `dog 1`, `dog 2`.
- Accepted proposals store a mask and canonical box on that frame and then use the normal propagation workflow.
- Temporary proposals are not saved unless accepted.

Ctrl+Z undoes the last prompt edit (single-step).

## Notes
- If the checkpoint field is left at default, SAM3 will use its default loading path.
- If needed, type a custom checkpoint path in the editable combo field.
- Bounding boxes are derived from SAM3 masks.
- Locked boxes are preserved on their own frame but still used as seeds for future propagation.
- When propagation advances to a later frame, the propagated box is stored there and the prior frame's point prompts are carried forward for the same object.
- Point translation during propagation maps relative position within the source box to the destination box (scale + translate); out-of-box points are not clamped.
- Point translation is applied only during propagation, not when re-segmenting the same frame.
- Auto-propagate/prefetch re-run only forward from the last edited frame and skip annotated frames unless prompts changed.
- Manual prompts are only used for the chunk whose seed frame contains those prompts.
- For carryover chunks, if the seed frame has no manual prompts, prompts are sampled from the seed frame's last known masks.
- In `Mask AABB Box` carryover mode, last masks are converted to axis-aligned boxes and passed as point labels `(2,3)`.
- Box prompts are not sent via SAM3 box API; they are always translated to point labels `(2,3)`.
- Text prompts are not tracked prompts. They are only used to generate one-frame proposals that can be accepted into the existing per-object workflow.
- Experimental periodic re-prompt settings are saved and restored with sessions.
- The one-session chunked propagation toggle is saved and restored with sessions.
- Smart propagation settings are saved and restored with sessions.
- Tracker-side scores are saved and restored with sessions for inspection, but are not exported yet.
- In `--research-mode`, loading a frame directory starts a fresh experiment timer, and saving a session also writes `research.json` with elapsed time and tracked click events.
- In `--research-mode`, loading a saved session restores prior research data in a paused state until you resume it.
- If a chunk returns no valid masks, chunked propagation stops and asks for prompt refinement before retrying.
