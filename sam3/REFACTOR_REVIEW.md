# Refactoring Review Plan for `sam3v2` Annotator

## Summary
Primary review target: [surgical_annotator_qt.py](/D:/SAM3Annotator/sam3v2/surgical_annotator_qt.py:825), with supporting modules in `tools/`. The main maintainability problem is not algorithmic complexity inside `sam3`; it is that the Qt entrypoint has become the app, the state store, the workflow controller, the persistence layer, and parts of the domain model at once.

The dominant refactoring goal should be to preserve behavior while shrinking responsibility boundaries, making state transitions explicit, and extracting pure logic from UI orchestration. There is also a repo-structure problem: your annotator application currently lives inside the upstream-style `sam3v2` tree, which blurs ownership and makes future upstream syncs harder than they need to be. The highest-value work is in `surgical_annotator_qt.py`; the `sam3/` package should remain mostly untouched unless later refactors expose a narrow integration cleanup.

## Findings

### 1. `AnnotatorMainWindow` is a god object
- Problem: One class owns too many responsibilities, making the code hard to read, change, and test safely.
- Evidence from the code: `AnnotatorMainWindow` spans roughly 5,250 lines starting at [surgical_annotator_qt.py](/D:/SAM3Annotator/sam3v2/surgical_annotator_qt.py:825) and owns around 172 `self` attributes. It handles UI setup, annotation state, propagation, prefetching, persistence, rendering, undo state, export, and research telemetry.
- Recommended refactor: Split responsibilities into cohesive components:
  - `AnnotatorState` or `AnnotationStore` for domain state
  - `SessionService` for save/load mapping and disk I/O
  - `PropagationController` for propagation and smart-restart workflow
  - `PrefetchController` for next-frame caching logic
  - `RenderController` for frame overlay composition
  - `ResearchController` for telemetry/timer behavior
  Keep `AnnotatorMainWindow` as the composition root and signal wiring layer.
- Priority: High
- Effort: Large
- Risk: Medium

### 2. Construction and initialization are too stateful
- Problem: Object creation is difficult to understand because unrelated runtime state, UI state, workflow state, and feature flags are initialized together.
- Evidence from the code: `__init__` at [surgical_annotator_qt.py](/D:/SAM3Annotator/sam3v2/surgical_annotator_qt.py:827) initializes dozens of fields across unrelated concerns. `_setup_ui` at [surgical_annotator_qt.py](/D:/SAM3Annotator/sam3v2/surgical_annotator_qt.py:1025) is about 443 lines and mixes widget construction, layout, defaults, and event binding.
- Recommended refactor: Group state into typed dataclasses such as `ViewSettings`, `PropagationSettings`, `ExperimentalSettings`, `CanvasState`, and `ResearchUiState`. Break `_setup_ui` into focused builders:
  - `_build_menu`
  - `_build_left_panel`
  - `_build_center_panel`
  - `_build_prompt_tab`
  - `_build_processing_tab`
  - `_build_experimental_tab`
  - `_build_status_bar`
- Priority: High
- Effort: Large
- Risk: Medium

### 3. Session persistence mixes too many concerns
- Problem: Save/load code combines schema definition, disk access, UI progress reporting, state reset, migration logic, and widget mutation.
- Evidence from the code: `_save_session` at [surgical_annotator_qt.py](/D:/SAM3Annotator/sam3v2/surgical_annotator_qt.py:4725) and `_load_session` at [surgical_annotator_qt.py](/D:/SAM3Annotator/sam3v2/surgical_annotator_qt.py:4839) both serialize and deserialize nested dicts, perform filesystem work, show dialogs, mutate window state, and recover from invalid paths.
- Recommended refactor: Introduce:
  - `SessionRepository` for disk I/O only
  - `SessionMapper` for translating between in-memory state and persisted schema
  - Typed session payload models for objects, prompts, boxes, outputs, view settings, and propagation settings
  Leave progress dialog and user-facing error prompts in thin window methods.
- Priority: High
- Effort: Large
- Risk: Medium

### 4. Propagation is an implicit state machine encoded as flags
- Problem: Propagation behavior is spread across callbacks and controlled by boolean fields instead of explicit state transitions.
- Evidence from the code: propagation and prefetch state are held in many fields near [surgical_annotator_qt.py](/D:/SAM3Annotator/sam3v2/surgical_annotator_qt.py:866), then coordinated across `propagate_next_frame`, `_on_sam_propagate_frame`, `_on_sam_propagate_done`, `_maybe_request_smart_propagation_restart`, and multiple prefetch handlers.
- Recommended refactor: Replace flag clusters with explicit run-state objects:
  - `PropagationRun`
  - `PrefetchRun`
  - `RunStatus`
  - `StopReason`
  Centralize transitions in a controller so slots invoke named transitions instead of patching shared state in place.
- Priority: High
- Effort: Large
- Risk: High

### 5. Domain logic is buried inside UI methods
- Problem: Core annotation and propagation behavior is difficult to test because it lives inside Qt-heavy methods.
- Evidence from the code: `_build_seed_prompts` at [surgical_annotator_qt.py](/D:/SAM3Annotator/sam3v2/surgical_annotator_qt.py:3286), `_apply_prompts_for_seed` at [surgical_annotator_qt.py](/D:/SAM3Annotator/sam3v2/surgical_annotator_qt.py:4363), and related merge/replay logic are business rules embedded in the main window.
- Recommended refactor: Extract pure functions or service methods for:
  - seed prompt selection
  - point translation
  - replay clearing
  - output merge/filter rules
  - visibility decisions
  These functions should accept plain data structures and return values without touching widgets.
- Priority: High
- Effort: Medium
- Risk: Low

### 6. Research mode is cross-cutting and leaks into unrelated code paths
- Problem: Research telemetry behavior is spread across window lifecycle, input handling, session persistence, and status-bar updates.
- Evidence from the code: research mode is initialized in `__init__`, hooked into `eventFilter` at [surgical_annotator_qt.py](/D:/SAM3Annotator/sam3v2/surgical_annotator_qt.py:954), saved during `_save_session`, restored in `_load_session`, and managed through methods around [surgical_annotator_qt.py](/D:/SAM3Annotator/sam3v2/surgical_annotator_qt.py:5157).
- Recommended refactor: Wrap `ResearchExperimentTracker` in a `ResearchController` with explicit hooks:
  - `on_frame_dir_loaded`
  - `on_session_loaded`
  - `on_session_saved`
  - `handle_widget_click`
  - `update_status_widgets`
  This keeps telemetry optional and isolated.
- Priority: Medium
- Effort: Medium
- Risk: Low

### 7. Naming obscures intent
- Problem: Several names describe implementation leftovers rather than current behavior.
- Evidence from the code:
  - `n_propagate_spin` is actually chunk size
  - `_prefetch_prompt_version` behaves as a cache generation token
  - mode labels mix `Prompt`, `Segment`, and `Box Annotation` semantics
  - propagation mode uses raw strings like `"tracker"` and `"copy_boxes"`
- Recommended refactor: Rename for intent and introduce enums/constants:
  - `chunk_size_spin`
  - `cache_generation`
  - `annotation_mode`
  - `PROPAGATION_MODE_TRACKER`
  - `PROPAGATION_MODE_COPY_BOXES`
- Priority: Medium
- Effort: Small
- Risk: Low

### 8. Raw dicts create hidden contracts and brittle code
- Problem: Internal behavior depends on string-keyed dicts with no typing or local schema enforcement.
- Evidence from the code: loose dict payloads are used for `_sam_task_contexts`, research event payloads, prefetch provenance, session payloads, and output serialization.
- Recommended refactor: Introduce dataclasses or small typed records for:
  - task context
  - prefetch provenance
  - research events
  - serialized frame output
  Keep raw dicts only at the final JSON boundary.
- Priority: Medium
- Effort: Medium
- Risk: Low

### 9. Settings synchronization is duplicated
- Problem: The same settings are represented in widget defaults, local state, persistence, and sync methods, which invites drift.
- Evidence from the code: view, experimental, and propagation settings are defined in UI setup, mirrored in `_sync_*` methods, saved in `_save_session`, and restored in `_load_session` using repeated keys and defaults.
- Recommended refactor: Define typed settings objects with one source of truth for defaults and helper methods for serialization and widget binding.
- Priority: Medium
- Effort: Medium
- Risk: Low

### 10. Test coverage does not match the risk surface
- Problem: The most brittle behavior is largely untested.
- Evidence from the code: there is a focused test for smart propagation in [tools/test_smart_propagation.py](/D:/SAM3Annotator/sam3v2/tools/test_smart_propagation.py:1), but there are no clear characterization tests for session round-trip, propagation transitions, prompt construction, or research-event handling.
- Recommended refactor: Add tests before large structural changes:
  - session save/load round-trip fixtures
  - propagation state transition tests
  - prompt builder tests
  - frame output merge tests
  - research event de-duplication and single/double click tests
- Priority: High
- Effort: Medium
- Risk: Low

### 11. Upstream `sam3` code and annotator application code are not cleanly separated
- Problem: Your application code is embedded inside the `sam3v2` repository layout, which makes ownership unclear and increases the cost of upstream merges, rebases, and reviews.
- Evidence from the code: the main app entrypoint [surgical_annotator_qt.py](/D:/SAM3Annotator/sam3v2/surgical_annotator_qt.py:1), annotator-specific helpers under `tools/`, and annotator docs all live alongside the upstream `sam3/` package and its original repo files. From a maintenance perspective, this makes it difficult to answer “what is upstream?” versus “what is ours?”.
- Recommended refactor: Split the repository structure into two clear layers:
  - keep upstream-derived code in a dedicated dependency or vendor area
  - move your annotator application into its own top-level app package outside `sam3v2`
  Suggested target shape:
  - `D:\SAM3Annotator\annotator_app\`
  - `D:\SAM3Annotator\annotator_app\src\annotator\`
  - `D:\SAM3Annotator\annotator_app\tests\`
  - `D:\SAM3Annotator\sam3v2\sam3\` and other upstream assets retained as the vendor/upstream tree
  Concretely:
  - move `surgical_annotator_qt.py` into the annotator app package
  - move annotator-only `tools/` modules into annotator-owned packages with clearer names like `annotator.persistence`, `annotator.propagation`, `annotator.research`, and `annotator.export`
  - keep the integration boundary narrow through a dedicated adapter layer that imports `sam3`
  - make the annotator runnable from its own package entrypoint instead of from inside `sam3v2`
- Priority: High
- Effort: Large
- Risk: Medium

## Phased Plan of Attack

### Phase 1: Characterize current behavior and lock the integration boundary
- Add tests around session save/load, smart restart behavior, prompt building, and propagation transitions.
- Capture a representative saved session fixture for round-trip assertions.
- Focus first on pure logic and state transitions rather than Qt-heavy integration tests.
- Identify the minimum stable surface the annotator needs from `sam3` and document it as the intended adapter boundary.

### Phase 2: Separate repository ownership
- Create a top-level annotator-owned application area outside `sam3v2`.
- Move the main annotator entrypoint, annotator-only modules, annotator tests, and annotator docs into that new area.
- Leave upstream-style `sam3v2` contents in place as a vendor or upstream-syncable tree.
- Introduce a dedicated adapter module so the annotator imports `sam3` through one narrow boundary.

### Phase 3: Stabilize vocabulary and types
- Introduce enums/constants for propagation modes and other magic string values.
- Add typed dataclasses for settings, task context, provenance, and research events.
- Rename misleading fields and methods while behavior is still unchanged.

### Phase 4: Extract persistence boundaries
- Move session schema mapping and filesystem logic into `SessionMapper` and `SessionRepository`.
- Keep dialog/progress behavior in thin window wrappers.
- Ensure save/load tests pass before any larger controller work.

### Phase 5: Extract workflow controllers
- Move propagation and prefetch logic into dedicated controllers with explicit run state.
- Convert boolean clusters into typed state transitions.
- Keep Qt signal connections in the window, but delegate decisions and state mutation.

### Phase 6: Modularize UI composition
- Split `_setup_ui` into focused builder methods or panel modules.
- Move widget-to-settings synchronization into dedicated binder helpers.
- Leave visual behavior unchanged.

### Phase 7: Isolate cross-cutting features
- Move research mode into a dedicated controller.
- Extract rendering helpers if still coupled to unrelated state.
- Remove remaining glue logic from `AnnotatorMainWindow` until it acts mainly as a shell.

## Best Quick Wins
- Add a thin adapter package boundary between annotator code and `sam3` imports.
- Move annotator-only docs and tests out of the upstream-looking tree first, even before full code extraction.
- Rename misleading fields such as `n_propagate_spin` and `_prefetch_prompt_version`.
- Replace raw propagation mode strings with constants or an enum.
- Introduce typed settings objects for view, propagation, and experimental settings.
- Extract session JSON encode/decode helpers before moving full save/load logic.
- Extract pure helper functions for prompt-building and output merge behavior.

## Refactors That Should Happen Before Others
1. Add characterization tests.
2. Separate annotator-owned files from the upstream-style tree and define the `sam3` adapter boundary.
3. Introduce types and naming cleanup.
4. Extract persistence mapping.
5. Extract propagation and prefetch controllers.
6. Split UI construction.
7. Isolate research mode and remaining cross-cutting helpers.

## Areas That Should Not Be Touched Yet
- `sam3/` model internals unless a narrow adapter cleanup becomes necessary.
- Rendering or propagation performance tuning without evidence of a concrete problem.
- Exporter behavior and file formats unless schema extraction makes a minimal cleanup unavoidable.
- Threading primitives themselves until propagation state is made explicit and well-tested.
- Upstream file layout inside `sam3v2` beyond what is needed to isolate your annotator ownership boundary.

## Missing Tests and Risky Areas
- Session round-trip preservation for objects, prompts, boxes, outputs, view settings, and research data.
- Propagation state transitions:
  - normal completion
  - stop request
  - smart restart
  - target-frame run behavior
  - prefetch invalidation
- Prompt building:
  - point-only seeds
  - box-only seeds
  - mixed prompts
  - empty-prompt validation
- Research event handling:
  - single vs double click
  - deduplication
  - pause/resume state
  - restore-from-session behavior

## Comments and Documentation Guidance
- Replace explanatory comments that describe what the code is doing with cleaner structure and better function names.
- Keep comments only where they explain a non-obvious invariant, lifecycle constraint, threading assumption, or serialization compatibility rule.
- Add short module-level docs for extracted controllers and repositories so their ownership boundaries are obvious.
- Document persisted session schema in one place after extraction rather than scattering field meaning across save/load code.

## Coding Standards to Enforce Going Forward
- Keep UI event handlers thin: validate input, delegate to a service/controller, then update widgets from returned state.
- Avoid methods longer than about 60 to 80 lines unless there is a clear reason.
- Do not use raw dicts for internal contracts; use dataclasses or typed objects.
- Maintain one source of truth for defaults and serialization keys.
- Prefer explicit state objects and enums over clusters of booleans.
- Extract pure business logic into testable helpers.
- Use comments only for non-obvious invariants, compatibility notes, or lifecycle constraints.

## Practical Implementation Notes
- Preserve behavior first; do not redesign the product while refactoring.
- Sequence work so tests and typed boundaries arrive before controller extraction.
- Treat `sam3v2` as upstream/vendor-owned as much as possible, and keep annotator-owned code in a separate top-level app area.
- Refactor in slices that can be reviewed independently:
  - repo separation
  - persistence
  - propagation
  - UI composition
  - research mode
- Keep public session and export behavior stable unless a compatibility migration is explicitly planned.
