"""Qt application entrypoint for constructing and running the annotator window."""

from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication

from annotator.main_window import AnnotatorMainWindow


def main(argv: list[str] | None = None) -> int:
    """Parse lightweight annotator flags, build the Qt app, and run the main window."""
    raw_args = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--research-mode", action="store_true")
    args, remaining = parser.parse_known_args(raw_args)
    app = QApplication([sys.argv[0], *remaining])
    win = AnnotatorMainWindow(research_mode_enabled=args.research_mode)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
