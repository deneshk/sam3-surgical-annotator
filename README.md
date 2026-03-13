# SAM3 Surgical Annotator

`sam3-surgical-annotator` is a SAM3-based desktop annotation tool for surgical video frame sequences. It provides per-object prompts, segmentation, frame-to-frame propagation, and COCO-style export with mask PNGs.

## Project Status

This repository is a derivative project built on top of Meta's SAM3 codebase. The surgical annotator application and related exporter/docs live in:

- `surgical_annotator_qt.py`
- `tools/exporters/coco_export.py`
- `tools/README_surgical_annotator.md`

The repository keeps the upstream SAM3 source tree because the annotator depends on the local SAM3 runtime.

## Key Features

- Load a directory of image frames as a sequence
- Create stable object IDs with per-object propagation enable/disable
- Add positive points, negative points, and draggable box prompts
- Run single-frame segmentation for prompted objects
- Run chunked forward propagation/tracking across frames
- Export COCO-style JSON with per-object mask PNG files

## Surgical Annotator

Install the annotator dependencies from the repository root:

```bash
python3 -m pip install -r requirements-annotator.txt
python3 -m pip install -e .
```

Run the desktop app:

```bash
python3 surgical_annotator_qt.py
```

Detailed app usage is documented in `tools/README_surgical_annotator.md`.

## Relationship to Upstream SAM3

This project includes code derived from Meta's SAM3 repository:

- Upstream project: `https://github.com/facebookresearch/sam3`
- License: see `LICENSE`

Any redistribution of the included SAM3 materials and derivative works remains subject to the SAM license shipped with this repository.

## Notes

- A working SAM3 environment is still required for inference.
- Checkpoint/BPE resolution follows the behavior implemented in the local SAM3 code and annotator UI.
- This repository is focused on surgical annotation workflows rather than being a clean fork of upstream documentation.
