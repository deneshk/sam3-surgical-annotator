# SAM3 Surgical Annotator (PySide6)

## What it does
- Loads a directory of image frames as a video sequence.
- Lets you create objects and set one active object at a time.
- Each object has a checkbox in the object list to enable/disable propagation for that object.
- Each object row includes `Prop`, `Rename`, `Solo`, and `Hide` controls.
- Uses a single canonical box per object/frame in the main workflow; drawn, propagated, and edited boxes are the same UI annotation.
- Newly drawn boxes and unlocked box edits are refined on the current frame with SAM3; the visible box is replaced by the box derived from the segmentation.
- `Lock Current Box` preserves a manual box on the current frame and prevents SAM3 from overwriting it there.
- The current frame annotation list stays visible in the main UI, including the box plus any point prompts for the active object.
- View controls now live in the `Prompting` tab, including prompt/segmentation/box visibility, box titles, mask opacity, and box thickness.
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
7. Click `Propagate`.
    - Only checked objects are included in propagation.
   - The canonical box on the seed frame is used as the primary propagation input.
   - Use `Pause` to stop after the current chunk or `Stop` to cancel immediately.
   - Enable `Auto-pause if Class Lost` to pause after a class loses its mask and show a warning.
8. If pause is enabled and chunks remain, review/correct and click `Continue Propagate`.
9. Use `Save Session` to persist your work (and enable auto-save if desired).
10. `Export COCO` from menu when done.

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
- If a chunk returns no valid masks, chunked propagation stops and asks for prompt refinement before retrying.
