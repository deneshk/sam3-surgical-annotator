# SAM3 Surgical Annotator (PySide6)

## What it does
- Loads a directory of image frames as a video sequence.
- Lets you create objects and set one active object at a time.
- Each object has a checkbox in the object list to enable/disable propagation for that object.
- Clicks add positive/negative point prompts (up to 6) for active object on current frame.
- Box prompts can be drawn by selecting `Box (drag)` and dragging on the image.
- Existing box prompts can be edited while `Box (drag)` mode is active:
  - click the box to reveal resize handles on corners and edge midpoints
  - drag a corner handle to resize both axes
  - drag an edge handle to resize one axis
  - drag inside the already-selected box to move it
- Prompt boxes are drawn as dashed outlines; predicted boxes are drawn as solid outlines.
- Use the mouse wheel to zoom the image view.
- Use right-click drag to pan the zoomed image view.
- Use `Fit to Screen` to reset zoom and pan to the default fitted view.
- Use the frame slider to scrub through the loaded frames.
- Use the `Go to` box to jump directly to a frame number.
- Hotkeys: Left/Right arrows move frame-by-frame, `P` starts propagation.
- Internally, each box prompt is passed to SAM3 as two point prompts with labels `(2, 3)`.
- View controls let you independently show/hide prompts, segmentations, and derived boxes.
- Segmentation opacity is adjustable from `0.00` to `1.00`.
- Bounding box line thickness is adjustable from `1` to `10` pixels.
- `Segment` runs SAM3 segmentation for prompted objects on current frame.
- In segment mode, point add/remove auto-refreshes segmentation.
- `Propagate` runs SAM3 forward tracking and auto-advances to the last frame with masks in each chunk.
- `Propagate` supports chunked tracking with:
  - `N frames` per chunk
  - `Chunks` count (total span is `N * Chunks`, clamped by video end)
  - `Sample pts` carryover control (1-8 positive points sampled per object from seed masks)
  - `Carryover` mode:
    - `Sample Points`
    - `Mask AABB Box` (converts last mask to axis-aligned box carryover)
  - optional `Pause Between Chunks` review mode
- Exports COCO-style JSON and mask PNG files.

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
3. Select prompt mode (`Positive`, `Negative`, or `Box (drag)`).
4. Add prompts:
   - Click image for point prompts.
   - Drag on image for box prompts.
    - In `Box (drag)` mode, click a box to reveal handles, then drag a handle to resize or drag inside the selected box to move it.
   - Prompt boxes cannot be edited while `Show Prompts` is off.
5. Click `Segment`.
6. Refine by adding/removing points.
7. Use the `View` controls to toggle prompts/segmentations/boxes and tune mask opacity or box thickness.
8. Set `N frames`, `Chunks`, `Sample pts`, `Carryover` mode, and optional `Pause Between Chunks`.
9. Click `Propagate`.
    - Only checked objects are included in propagation.
   - On the first propagation chunk, enabled objects must have either manual prompts on the seed frame or an existing stored mask on that frame; stored masks are converted to box prompts automatically.
10. If pause is enabled and chunks remain, review/correct and click `Continue Propagate`.
11. `Export COCO` from menu when done.

## Notes
- If checkpoint/BPE fields are left at default, SAM3 will use its default loading path.
- If needed, type custom checkpoint/BPE paths in the editable combo fields.
- Bounding boxes are derived from SAM3 masks.
- Manual prompts are only used for the chunk whose seed frame contains those prompts.
- For carryover chunks, if the seed frame has no manual prompts, prompts are sampled from the seed frame's last known masks.
- In `Mask AABB Box` carryover mode, last masks are converted to axis-aligned boxes and passed as point labels `(2,3)`.
- Box prompts are not sent via SAM3 box API; they are always translated to point labels `(2,3)`.
- If a chunk returns no valid masks, chunked propagation stops and asks for prompt refinement before retrying.
