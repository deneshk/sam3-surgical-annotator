# SAM3 Surgical Annotator App

This directory contains the annotator-owned application code. The SAM3 runtime remains vendored in the sibling [sam3](../sam3/README.md) tree, and the app imports it through a small vendor bootstrap layer.

## Layout

- `src/annotator/main_window.py`: main Qt window implementation
- `src/annotator/main.py`: app entrypoint
- `src/annotator/exporters/`: annotator-owned export logic
- `src/annotator/persistence/`: session JSON and mask I/O
- `src/annotator/propagation/`: propagation-side pure helpers
- `src/annotator/research/`: research telemetry helpers
- `tests/`: annotator-owned tests
- `docs/`: annotator-specific docs moved out of the vendor tree

## Run

From `D:\SAM3Annotator\annotator_app`:

```bash
python run_annotator.py
```

Research mode:

```bash
python run_annotator.py --research-mode
```

The app expects the vendored SAM3 runtime at `D:\SAM3Annotator\sam3` by default. Override this with `SAM3_VENDOR_ROOT` if needed.
