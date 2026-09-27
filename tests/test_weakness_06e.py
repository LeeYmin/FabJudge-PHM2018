import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fabjudge.weakness_06e import _window_features


class TestLongWindowCausality(unittest.TestCase):
    def test_future_measurements_do_not_change_endpoint_features(self):
        times = np.arange(0, 40001, 100)
        signals = np.column_stack((times / 1000, np.sin(times / 1000),
                                   np.cos(times / 1000)))
        endpoints = np.array([299, 300, 350])
        before, sufficient = _window_features(times, signals, endpoints)
        changed = signals.copy()
        changed[times > 30000] += 10000
        after, _ = _window_features(times, changed, endpoints)
        self.assertFalse(sufficient[0])
        self.assertTrue(sufficient[1])
        np.testing.assert_allclose(before[1], after[1])


if __name__ == "__main__":
    unittest.main()
