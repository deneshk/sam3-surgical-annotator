# HANDOVER

## Purpose

This workspace contains a desktop surgical video annotator built on top of a vendored SAM3 runtime.

- `annotator_app/` is the app-owned codebase.
- `sam3/` is the vendored SAM3 runtime tree plus upstream-style assets.

This document is the current handover for the whole workspace. Treat `sam3/HANDOVER.md` as historical context from the pre-split layout, not as the primary source of truth.

## Top-Level Layout

- `annotator_app/README.md`: short app-level overview and run notes.
- `annotator_app/run_annotator.py`: convenience launcher.
- `annotator_app/src/annotator/main_window.py`: main Qt window and orchestration layer. This is still the largest and most important app file.
- `annotator_app/src/annotator/main.py`: Qt entrypoint.
- `annotator_app/src/annotator/sam.py`: adapter and worker boundary to the vendored SAM3 predictor.
- `annotator_app/src/annotator/models.py`: shared app dataclasses, defaults, and propagation constants.
- `annotator_app/src/annotator/widgets.py`: canvas widget and frame jump control.
- `annotator_app/src/annotator/propagation/`: pure helpers for prompt payloads, frame-output merges, and runtime state records.
- `annotator_app/src/annotator/persistence/`: session schema, JSON mapping, mask I/O, and repository boundary.
- `annotator_app/src/annotator/exporters/`: COCO and Perk export flows.
- `annotator_app/src/annotator/research/`: optional research telemetry and experiment logging.
- `annotator_app/tests/`: app-owned tests and the best quick behavioral reference after the code itself.
- `sam3/sam3/model/sam3_video_predictor.py`: main vendor predictor used by the app.

## What The Project Does

The app loads a directory of image frames and treats it as a video sequence. Users create named objects, annotate each object with a canonical box plus optional positive and negative point prompts, refine current-frame masks with SAM3, and propagate annotations forward through the sequence.

Core user-facing workflows:

- Load a frame directory and browse frames with slider, jump control, hotkeys, play/pause preview, zoom, and pan.
- Create objects with stable `obj_id`s, per-object colors, inline row controls, and frame-local propagation enablement.
- Annotate boxes directly on the canvas, including move and resize via drag handles.
- Add positive or negative point prompts on the active object.
- Run current-frame segmentation with SAM3.
- Run forward propagation in either tracker mode or copy-boxes mode.
- Optionally auto-propagate one frame ahead and prefetch next-frame outputs.
- Flag review frames.
- Save and load sessions, including masks and research telemetry.
- Export annotations as COCO-style JSON plus mask PNGs, or as Perk-format CSV.
- Optionally run `--research-mode` to record timing and interaction telemetry into `research.json`.

## Architecture In Practice

The app is split, but not fully decomposed. `main_window.py` still owns most orchestration, while pure or lower-level concerns have been extracted into helper modules.

Important boundaries:

- UI/orchestration: `annotator_app/src/annotator/main_window.py`
- SAM runtime boundary: `annotator_app/src/annotator/sam.py`
- Persistence boundary: `annotator_app/src/annotator/persistence/session_repository.py`
- Pure propagation logic: `annotator_app/src/annotator/propagation/*`
- Research telemetry: `annotator_app/src/annotator/research/*`
- Vendor runtime: `sam3/`

Rule of thumb for future work:

- If the task is app behavior, start in `annotator_app/`.
- Only edit `sam3/` when the app/runtime integration truly requires it.

## Current UI And Loading Notes

Frame-directory loading currently stores a sorted list of image paths, not decoded images. Frames are decoded on demand for rendering or frame-size queries through `main_window.py` helpers.

Important recent history:

- Video preview playback is part of `main`: the navigation row has Play/Pause plus configurable FPS.
- PR #2 attempted eager parallel image metadata caching for faster directory loading, but it made large directory loads appear to hang and was reverted by PR #3. Avoid blocking directory load on all-frame metadata or all-image preload work.
- Shortcut ownership is centralized in `_setup_shortcuts()`. Menu actions should remain clickable but should not register duplicate global shortcuts. Left/Right and Shift+Left/Shift+Right are owned by the window-level shortcut setup.
- The frame slider should not accept keyboard focus, and Up/Down should not move frames through the frame-jump control.

## Core State Model

These in-memory maps drive most app behavior:

- `self.objects`: list of `ObjectInfo`
- `self.prompts_by_frame_obj[frame_idx][obj_id] -> list[PointPrompt]`
- `self.box_prompts_by_frame_obj[frame_idx][obj_id] -> BoxPrompt`
- `self.box_locked_by_frame_obj[frame_idx][obj_id] -> bool`
- `self.outputs_by_frame[frame_idx] -> SamFrameOutput`
- `self.manual_propagation_overrides_by_frame_obj[frame_idx][obj_id] -> bool`
- `self.flagged_frame_indices`: review flags
- `self.hidden_obj_ids` and `self.solo_object_id`: visibility filtering

Important conceptual distinction:

- Canonical boxes live in `box_prompts_by_frame_obj`. These are the visible and editable frame/object boxes used throughout the UI and by the Perk exporter.
- Model outputs live in `outputs_by_frame`. These hold masks, normalized boxes, confidence scores, and tracker scores, and they drive overlays plus the COCO export path.

Box behavior matters:

- Boxes are sent to SAM as two point labels `(2, 3)`, not through a separate SAM box API.
- Unlocked boxes can be overwritten by SAM-derived boxes after segmentation or propagation.
- Locked boxes preserve manual box edits on that frame.

## SAM3 Integration

The app imports the vendored runtime through `annotator.vendor.sam3_runtime.ensure_vendor_sam3_on_path()`.

- Default vendor root: `D:\SAM3Annotator\sam3`
- Override path: `SAM3_VENDOR_ROOT`

`Sam3Adapter` wraps `Sam3VideoPredictor`, and `SamWorker` serializes all SAM work through one Qt worker thread.

Current behavior:

- SAM is initialized lazily, usually on first frame-directory load or first task that needs it.
- The app uses one worker and one adapter, not multiple concurrent SAM model instances.
- Current-frame segmentation uses a fresh 1-frame session.
- Tracker propagation uses chunk-local sessions by default.
- Auto-propagate/prefetch also uses the same worker queue and can be canceled and restarted.

## Propagation Model

There are two propagation modes:

- `tracker`: seed SAM3 with current-frame boxes, optional point prompts, and optional mask inputs, then stream forward outputs.
- `copy_boxes`: copy canonical boxes forward one frame at a time and re-segment each destination frame.

Tracker seeding rules are important:

- A seed can come from current-frame point prompts, current-frame box, or existing current-frame output mask.
- If there is no explicit box but there is a stored seed mask, the app samples an AABB from the mask.
- If an enabled object has no usable seed data, propagation aborts with a warning.
- Point prompts can be excluded from tracker seeding via `Use Point Prompts for Propagation`.
- Propagation enablement is frame-local: objects without seed data are disabled on that frame.

## Persistence And Export

Session persistence is versioned through `annotator_app/src/annotator/persistence/session_models.py`.

- Current session schema version: `6`
- Session folder contents: `session.json` plus mask PNG assets under `masks/`
- Research mode also writes `research.json`

Exports:

- COCO exporter uses `outputs_by_frame` and writes mask PNGs plus `annotations_coco.json`
- Perk exporter uses canonical boxes from `box_prompts_by_frame_obj` and writes `annotations_perk.csv`

That exporter split is intentional and important when reviewing bugs.

## Tests And Validation

Useful command from the repo root:

```powershell
python -m unittest discover annotator_app\tests
```

Coverage is focused on:

- propagation helpers
- session persistence
- runtime state records
- research telemetry helpers
- shortcut ownership/static UI regressions

Some research/controller tests skip automatically when `PySide6` is unavailable.

## Git Workflow

Use `main` as the stable integration branch. Do not develop new features directly on historical or experimental branches.

Recommended feature workflow:

```powershell
git switch main
git pull origin main
git switch -c feature/<short-description>
```

For fixes or maintenance:

```powershell
git switch -c fix/<short-description>
git switch -c chore/<short-description>
```

Branch naming conventions:

- `feature/...`: user-facing capability or workflow improvement
- `fix/...`: bug fix or regression fix
- `chore/...`: maintenance, repo hygiene, or non-user-facing cleanup
- `experiment/...`: exploratory work that should not be merged wholesale without review
- `revert/...`: targeted revert branch when undoing a merged PR

Default collaboration flow:

1. Start from updated `main`.
2. Make focused changes in app-owned files first.
3. Run relevant validation, usually `python -m unittest discover annotator_app\tests`.
4. Push the feature branch to `origin`.
5. Open a GitHub PR into `main`.
6. Leave the PR open for manual review unless explicitly asked to merge.

Prefer reverting merged work with a revert commit or revert PR rather than rewriting `main` history.

## Source-Of-Truth Guidance

Use this priority order when facts disagree:

1. Current code in `annotator_app/src/annotator/*`
2. App tests in `annotator_app/tests/*`
3. This handover
4. `annotator_app/README.md`
5. Historical documents such as `sam3/HANDOVER.md`, `sam3/REFACTOR_REVIEW.md`, or older annotator docs

Known stale or partially stale references:

- `sam3/HANDOVER.md`: pre-split handover, useful historically but not layout-accurate
- `annotator_app/docs/README_surgical_annotator.md`: useful overview, but some feature notes lag current code

## Suggested Reading Order For A New Coding Agent

1. Read this file.
2. Read `annotator_app/README.md`.
3. Read `annotator_app/src/annotator/main_window.py`.
4. Read `annotator_app/src/annotator/sam.py`.
5. Read `annotator_app/src/annotator/models.py`.
6. Read `annotator_app/src/annotator/propagation/prompt_payloads.py` and `frame_outputs.py`.
7. Read `annotator_app/src/annotator/persistence/session_models.py`, `session_mapper.py`, and `session_repository.py`.
8. Read `annotator_app/src/annotator/research/controller.py` and `experiment.py` if the task touches research mode.
9. Read `annotator_app/tests/*` for behavioral expectations.
10. Only then descend into `sam3/` if the task truly touches model internals or vendor integration.

## Recommended Agent Initialization Prompt

```text
You are working in D:\SAM3Annotator.

Start with docs/HANDOVER.md and treat it as the current repo-level handover.

Repository ownership:
- annotator_app/ = app-owned code
- sam3/ = vendored SAM3 runtime

Default approach:
- Inspect and edit annotator_app first.
- Only touch sam3 if the task clearly requires vendor/runtime changes.
- Do not rely on sam3/HANDOVER.md as the primary source of truth; it is historical.

Read in this order:
1. docs/HANDOVER.md
2. annotator_app/README.md
3. annotator_app/src/annotator/main_window.py
4. annotator_app/src/annotator/sam.py
5. annotator_app/src/annotator/models.py
6. annotator_app/src/annotator/propagation/*
7. annotator_app/src/annotator/persistence/*
8. annotator_app/tests/*

Working assumptions:
- main_window.py is still the orchestration hub
- canonical boxes and SAM outputs are separate state layers
- session schema version is 6
- fresh directory loading stores image paths and decodes frames lazily
- shortcut bindings should have one owner in _setup_shortcuts()
- code and tests beat older docs when they disagree

Before editing:
- summarize the affected workflow
- identify the exact files you will touch
- state how you will validate the change

Git workflow:
- start new work from updated main
- create a focused feature/fix/chore branch
- push to origin and open a PR into main for manual review
- do not merge PRs unless explicitly asked

Validation defaults:
- run python -m unittest discover annotator_app\tests when relevant
- mention any skipped tests or environment limits explicitly
```
