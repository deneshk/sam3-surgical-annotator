"""Package entrypoint for running the annotator with `python -m annotator`."""

from annotator.main import main


if __name__ == "__main__":
    raise SystemExit(main())
