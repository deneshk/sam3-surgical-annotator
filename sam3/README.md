# Vendored SAM3 Runtime

This directory now serves as the vendored SAM3 runtime tree used by the surgical annotator application. The annotator-owned app code has been moved out to the sibling `annotator_app` directory so this tree can stay closer to upstream ownership boundaries.

## Project Status

The surgical annotator application now lives in `../annotator_app`. This directory keeps the upstream-style SAM3 source tree because the annotator depends on the local SAM3 runtime.

## Annotator Integration

Install the SAM3 runtime dependencies from this directory:

```bash
python3 -m pip install -e .
```

Then install and run the annotator from `../annotator_app`:

```bash
cd ../annotator_app
python3 -m pip install -e .
python3 run_annotator.py
```

Detailed app usage is documented in `../annotator_app/docs/README_surgical_annotator.md`.

## Relationship to Upstream SAM3

This project includes code derived from Meta's SAM3 repository:

- Upstream project: `https://github.com/facebookresearch/sam3`
- License: see `LICENSE`

Any redistribution of the included SAM3 materials and derivative works remains subject to the SAM license shipped with this repository.

## Notes

- A working SAM3 environment is still required for inference.
- Checkpoint/BPE resolution follows the behavior implemented in the vendored SAM3 code and annotator UI.
- This directory is no longer the home of the annotator application itself.
