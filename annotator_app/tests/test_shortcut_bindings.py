"""Static checks for main-window shortcut ownership."""

import re
import unittest
from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parents[1]
MAIN_WINDOW_PATH = TESTS_ROOT / "src" / "annotator" / "main_window.py"


class ShortcutBindingTests(unittest.TestCase):
    def test_global_shortcuts_are_registered_once(self) -> None:
        source = MAIN_WINDOW_PATH.read_text(encoding="utf-8")
        expected_once = [
            r"QKeySequence\(Qt\.Key_Left\)",
            r"QKeySequence\(Qt\.Key_Right\)",
            r'QKeySequence\("Shift\+Left"\)',
            r'QKeySequence\("Shift\+Right"\)',
            r'QKeySequence\("P"\)',
            r'QKeySequence\("F"\)',
            r"QKeySequence\(Qt\.Key_Delete\)",
            r"QKeySequence\.Undo",
        ]

        for pattern in expected_once:
            with self.subTest(pattern=pattern):
                self.assertEqual(len(re.findall(pattern, source)), 1)

    def test_frame_slider_does_not_accept_keyboard_focus(self) -> None:
        source = MAIN_WINDOW_PATH.read_text(encoding="utf-8")

        self.assertIn("self.frame_slider.setFocusPolicy(Qt.NoFocus)", source)


if __name__ == "__main__":
    unittest.main()
