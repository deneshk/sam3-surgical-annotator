# SAM3 Box Prompt Findings

## Quick summary
- There are two different "box" behaviors in this codebase:
  - Native semantic box prompt path (`bounding_boxes` / `box_labels`).
  - Tracker box-as-points path (two points with labels `2` and `3`).
- In the current annotator UI, box prompts are implemented as points (`2,3`), not native `bounding_boxes`.

## 1) Native semantic box vs box-as-points

### A) Native semantic box prompt path
- Entry path: `Sam3VideoPredictor.add_prompt(..., bounding_boxes=..., bounding_box_labels=...)`.
- Routed to SAM3 semantic box handling (`boxes_xywh`, `box_labels`), not tracker point refinement.
- Box labels are effectively binary in this config:
  - `1` = positive box
  - `0` = negative box

Example:
```python
resp = predictor.handle_request(
    request={
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": 0,
        "bounding_boxes": [[0.30, 0.20, 0.25, 0.35]],  # [x, y, w, h], normalized
        "bounding_box_labels": [1],
    }
)
```

### B) Tracker `box` argument path
- Tracker converts a box to two points internally:
  - top-left -> label `2`
  - bottom-right -> label `3`
- This is object-id anchored refinement behavior.

Example:
```python
frame_idx, obj_ids, low_res_masks, video_res_masks = tracker.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=12,
    obj_id=7,
    box=[0.30, 0.20, 0.55, 0.55],  # [x1, y1, x2, y2], normalized
    clear_old_points=True,
    rel_coordinates=True,
)
```

### C) Explicit box-as-points (`2,3`) path
- Equivalent representation to tracker `box=...` after conversion.
- Useful when you already have point-based prompt wiring.

Example:
```python
resp = predictor.handle_request(
    request={
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": 12,
        "obj_id": 7,
        "points": [[0.30, 0.20], [0.55, 0.55]],
        "point_labels": [2, 3],
    }
)
```

## 2) Is there segmentation/tracking difference?
- Yes, there can be:
  - Segmentation can differ because native semantic boxes and point-based box corners go through different prompt encoding paths.
  - Tracking behavior can differ because point path is explicitly tied to `obj_id` refinement flow.

## 3) Can native box path take `obj_id`?
- Not in the same practical way as tracker points.
- `obj_id` binding for interactive per-instance refinement is on the points/tracker branch (`points` + `point_labels` + `obj_id`).

## 4) What if you provide multiple native boxes at once?
- On initial visual-prompt state, SAM3 expects a single initial box.
- Multiple boxes in that initial case can raise a runtime error.
- In later contexts, box prompts are not explicitly bound per-box to tracker object IDs like point refinement is.

Example that can fail in initial visual-prompt case:
```python
resp = predictor.handle_request(
    request={
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": 0,
        "bounding_boxes": [
            [0.10, 0.10, 0.20, 0.20],
            [0.50, 0.20, 0.25, 0.30],
        ],
        "bounding_box_labels": [1, 1],
    }
)
```

## 5) What labels are valid?
- Native semantic box labels: `0` (negative), `1` (positive).
- Box-as-points corner labels: `2` (top-left), `3` (bottom-right).

## 6) Current annotator behavior
- `surgical_annotator_qt.py` uses box-as-points:
  - Converts dragged box to two points.
  - Sends labels `2` and `3`.
- `tools/README_surgical_annotator.md` already documents this behavior.
