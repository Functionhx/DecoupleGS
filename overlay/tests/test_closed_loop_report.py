from __future__ import annotations

import unittest

from decouplegs.closed_loop_report import (
    bootstrap_mean_interval,
    summarize_binary,
    summarize_numeric,
    wilson_interval,
)


class ClosedLoopReportTest(unittest.TestCase):
    def test_bootstrap_is_deterministic_and_contains_mean(self) -> None:
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        first = bootstrap_mean_interval(values, samples=2_000, seed=17)
        second = bootstrap_mean_interval(values, samples=2_000, seed=17)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], 0.3)
        self.assertGreaterEqual(first[1], 0.3)

    def test_binary_summary_keeps_rate_and_wilson_interval(self) -> None:
        summary = summarize_binary([True] * 45 + [False] * 5)
        self.assertAlmostEqual(summary["mean"], 0.9)
        self.assertEqual(summary["successes"], 45)
        self.assertLess(summary["ci95_low"], 0.9)
        self.assertGreater(summary["ci95_high"], 0.9)
        self.assertEqual(
            (summary["ci95_low"], summary["ci95_high"]), wilson_interval(45, 50)
        )

    def test_numeric_summary_uses_population_std(self) -> None:
        summary = summarize_numeric([1.0, 3.0], samples=1_000, seed=3)
        self.assertAlmostEqual(summary["mean"], 2.0)
        self.assertAlmostEqual(summary["std"], 1.0)
        self.assertEqual(summary["count"], 2)


if __name__ == "__main__":
    unittest.main()
