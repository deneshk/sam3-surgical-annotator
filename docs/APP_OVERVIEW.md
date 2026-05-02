# SAM3 Surgical Annotator App Overview

## Purpose

The app is a desktop surgical video annotation tool built with PySide6. It loads a directory of image frames, lets users define and refine object annotations, uses the vendored SAM3 runtime for mask prediction and tracking, and saves or exports the resulting annotations.

The workspace has two important ownership boundaries:

- `annotator_app/`: app-owned UI, workflow, persistence, export, and test code.
- `sam3/`: vendored SAM3 runtime used through a narrow adapter layer.

Most app behavior is orchestrated by `annotator_app/src/annotator/main_window.py`. The SAM3 runtime is reached through `annotator_app/src/annotator/sam.py`.

## Main Workflow

1. The user loads a directory of image frames.
2. The app stores sorted frame paths and decodes frames lazily as needed.
3. The user creates one or more named objects.
4. For each object, the user can draw a canonical box and add positive or negative point prompts.
5. The user runs current-frame segmentation to generate SAM masks.
6. The user can propagate annotations forward using tracker propagation or copy-box propagation.
7. The app can save and load sessions, including prompts, boxes, masks, view settings, and propagation settings.
8. The user can export annotations as COCO-style mask data or Perk-format box data.

## Key Concepts

### Frames

Loaded frame directories are treated as video sequences. The app keeps `Path` entries in `self.frame_paths` and reads image data only when rendering, measuring frame size, segmenting, or propagating. This avoids blocking large directory loads on full image decoding.

### Objects

Objects are stored as `ObjectInfo` records with a stable `obj_id`, display name, and color. Object IDs connect prompts, boxes, SAM outputs, visibility state, exports, and persistence records.

### Point Prompts

Point prompts are stored per frame and object in:

```text
self.prompts_by_frame_obj[frame_idx][obj_id] -> list[PointPrompt]
```

Each point has pixel coordinates and a positive or negative label. Positive prompts guide SAM toward the object; negative prompts guide it away from unwanted regions.

### Canonical Boxes

Canonical boxes are stored separately from SAM outputs:

```text
self.box_prompts_by_frame_obj[frame_idx][obj_id] -> BoxPrompt
```

These boxes are the editable UI boxes and the source for Perk exports. Boxes are sent to SAM as two point labels, `2` and `3`, rather than through a separate box API.

### SAM Outputs

SAM-generated masks, normalized boxes, confidence scores, and tracker scores are stored in:

```text
self.outputs_by_frame[frame_idx] -> SamFrameOutput
```

This state powers mask overlays and COCO export. It is intentionally separate from canonical box state so manual boxes and model outputs can evolve independently.

### Box Locks

Per-frame object boxes can be locked:

```text
self.box_locked_by_frame_obj[frame_idx][obj_id] -> bool
```

Unlocked boxes may be updated from SAM-derived boxes after segmentation or propagation. Locked boxes preserve manual edits on that frame.

## Main App Components

### `main_window.py`

`AnnotatorMainWindow` is the orchestration hub. It owns the main Qt UI, high-level app state, frame navigation, prompt editing, segmentation requests, propagation requests, session save/load actions, rendering, shortcut setup, and worker coordination.

Important responsibilities include:

- Building the main window, tool panels, object list, frame controls, and canvas.
- Loading frame directories and tracking the current frame.
- Creating, selecting, hiding, soloing, and deleting objects.
- Handling point prompt and box prompt interactions on the canvas.
- Rendering frames, masks, boxes, prompts, titles, zoom, and pan.
- Starting current-frame segmentation.
- Starting, stopping, and continuing propagation runs.
- Managing prefetch and one-frame-ahead auto-propagation.
- Saving and loading session payloads.
- Calling exporters.
- Coordinating research telemetry when research mode is enabled.

Shortcut ownership is centralized in `_setup_shortcuts()`. Menu actions should remain clickable without registering duplicate global shortcuts.

### `sam.py`

`sam.py` is the app boundary around the vendored SAM3 runtime.

`Sam3Adapter` wraps `Sam3VideoPredictor` and normalizes SAM responses into `SamFrameOutput`.

`SamWorker` runs on a Qt worker thread and serializes SAM work through one queue. It handles:

- Lazy SAM initialization.
- Current-frame segmentation tasks.
- Tracker propagation tasks.
- Prefetch propagation tasks.
- Cancellation requests.
- Emitting Qt signals back to the main window.

The app uses one worker and one adapter, not multiple concurrent SAM model instances.

### `models.py`

`models.py` defines shared app records and defaults:

- `PointPrompt`
- `BoxPrompt`
- `ObjectInfo`
- `SamFrameOutput`
- `ViewSettings`
- `PropagationSettings`
- `PendingPropagationState`

It also defines propagation mode constants:

- `tracker`
- `copy_boxes`

### `propagation/`

The propagation package contains pure helpers that keep prompt and output logic testable outside Qt.

Key files:

- `prompt_payloads.py`: converts app prompts, boxes, masks, and frame sizes into SAM worker payloads.
- `frame_outputs.py`: merges frame outputs, removes object outputs, and samples boxes from masks.
- `runtime.py`: typed state records for propagation, prefetch, and SAM task metadata.

Tracker propagation can seed from:

- Current-frame boxes.
- Current-frame point prompts.
- Existing current-frame output masks.
- Sampled boxes derived from masks when no explicit box exists.

Copy-box propagation copies canonical boxes forward and segments destination frames one at a time.

### `persistence/`

The persistence package owns the session schema and disk I/O boundary.

Key files:

- `session_models.py`: typed session payload records and current schema version.
- `session_mapper.py`: converts between typed payloads and persisted JSON data.
- `session_repository.py`: reads and writes session folders.
- `session_io.py`: low-level JSON and mask PNG helpers.

The current session schema version is `6`.

A saved session contains:

- `session.json`
- mask PNG assets under `masks/`
- optional research files when research mode is enabled

### `exporters/`

Export logic is intentionally split by data source:

- COCO export uses SAM outputs from `outputs_by_frame`.
- Perk export uses canonical boxes from `box_prompts_by_frame_obj`.

This distinction matters because masks and canonical boxes are separate state layers.

### `research/`

Research mode records optional interaction telemetry. It can track UI events, canvas interactions, timing, and sampled mouse positions, then persist them beside a saved session.

Research mode is enabled through the app launcher with:

```powershell
python run_annotator.py --research-mode
```

## Key User-Facing Functions

### Load Frames

The app loads image files from a selected directory, sorts them, stores their paths, and displays the first frame. Supported formats include common image extensions such as `.jpg`, `.png`, `.bmp`, `.tif`, and `.webp`.

### Navigate Frames

Users can move through frames with the slider, jump control, keyboard shortcuts, and playback controls. The preview player advances frames at a configurable FPS.

### Manage Objects

Users can create named objects, choose an active object, hide objects, solo one object, delete objects, and control whether each object participates in propagation from the current frame.

### Edit Prompts

Users can add positive and negative point prompts for the active object. Prompts are frame-local and object-specific.

### Edit Boxes

Users can draw, select, move, and resize canonical boxes directly on the canvas. Boxes can be locked to prevent SAM-derived updates from replacing manual edits.

### Segment Current Frame

Current-frame segmentation builds a one-frame SAM session, sends point and box prompts for relevant objects, receives masks and scores, merges object outputs, updates overlays, and updates unlocked canonical boxes when appropriate.

### Propagate Forward

Propagation can run in two modes:

- `tracker`: seed SAM3 on the current frame and stream outputs forward.
- `copy_boxes`: copy canonical boxes forward and segment each destination frame.

Propagation can run for fixed chunks or toward a target frame. The app tracks chunk progress and can stop active propagation.

### Auto-Propagate And Prefetch

When enabled, the app can prefetch next-frame outputs from the current frame state and apply them during frame navigation. Prompt edits invalidate the prefetch cache so stale outputs are not reused.

### Save And Load Sessions

Sessions preserve frame references, current frame, objects, prompts, canonical boxes, box locks, propagation settings, view settings, flags, model outputs, and masks.

### Export Annotations

COCO export writes mask-based annotations from model outputs. Perk export writes box-based annotations from canonical boxes.

### Flag Review Frames

Users can flag frames for later review. Flag state is stored in sessions.

## Validation And Tests

The main app test suite lives under `annotator_app/tests/`.

The default validation command from the repo root is:

```powershell
python -m unittest discover annotator_app\tests
```

Current focused coverage includes:

- Propagation helper behavior.
- Session persistence round trips.
- Runtime and research state records.
- Shortcut ownership checks.
- Disk-backed frame output storage.

Some tests skip automatically when optional GUI dependencies such as PySide6 are unavailable.
