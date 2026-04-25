"""Decision-level tests for smart propagation restart detection."""

import sys
import unittest
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = TESTS_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from annotator.propagation.smart_propagation import detect_smart_propagation_restart


class SmartPropagationDecisionTests(unittest.TestCase):
    def test_triggers_when_seen_object_becomes_empty(self) -> None:
        decision = detect_smart_propagation_restart(
            enabled_obj_ids={1, 2},
            has_mask_by_obj_id={1: False, 2: True},
            previously_seen_obj_ids={1, 2},
            already_triggered_loss_keys=set(),
            loss_frame_idx=20,
            run_start_frame_idx=5,
            rewind_frames=3,
        )

        self.assertTrue(decision.should_restart)
        self.assertEqual(decision.restart_seed_frame_idx, 17)
        self.assertEqual(decision.loss_frame_idx, 20)
        self.assertEqual(decision.lost_obj_ids, (1,))

    def test_does_not_trigger_for_object_never_seen(self) -> None:
        decision = detect_smart_propagation_restart(
            enabled_obj_ids={1},
            has_mask_by_obj_id={1: False},
            previously_seen_obj_ids=set(),
            already_triggered_loss_keys=set(),
            loss_frame_idx=10,
            run_start_frame_idx=0,
            rewind_frames=3,
        )

        self.assertFalse(decision.should_restart)

    def test_rewind_clamps_to_run_start(self) -> None:
        decision = detect_smart_propagation_restart(
            enabled_obj_ids={1},
            has_mask_by_obj_id={1: False},
            previously_seen_obj_ids={1},
            already_triggered_loss_keys=set(),
            loss_frame_idx=6,
            run_start_frame_idx=5,
            rewind_frames=10,
        )

        self.assertTrue(decision.should_restart)
        self.assertEqual(decision.restart_seed_frame_idx, 5)

    def test_same_object_same_loss_frame_only_triggers_once(self) -> None:
        decision = detect_smart_propagation_restart(
            enabled_obj_ids={1},
            has_mask_by_obj_id={1: False},
            previously_seen_obj_ids={1},
            already_triggered_loss_keys={(1, 12)},
            loss_frame_idx=12,
            run_start_frame_idx=0,
            rewind_frames=3,
        )

        self.assertFalse(decision.should_restart)


if __name__ == "__main__":
    unittest.main()
