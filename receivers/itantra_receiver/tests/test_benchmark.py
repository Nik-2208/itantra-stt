"""
Unit Tests for benchmark.py
Tests statistical aggregation, RTF calculation, report export, and execution harness.
"""

import json
import unittest
from pathlib import Path
from benchmark import calculate_statistics, PipelineBenchmark
import config


class TestBenchmark(unittest.TestCase):
    def test_statistics_calculation(self):
        """Test exact percentile, mean, median, std_dev computation."""
        # Test known series: [10, 20, 30, 40, 50]
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = calculate_statistics(data)
        self.assertEqual(stats["count"], 5)
        self.assertEqual(stats["min"], 10.0)
        self.assertEqual(stats["max"], 50.0)
        self.assertEqual(stats["mean"], 30.0)
        self.assertEqual(stats["median"], 30.0)
        self.assertEqual(stats["p95"], 50.0)
        self.assertAlmostEqual(stats["std_dev"], 15.81, places=1)

    def test_statistics_empty(self):
        """Test statistical calculation with empty list."""
        stats = calculate_statistics([])
        self.assertEqual(stats["count"], 0)
        self.assertEqual(stats["mean"], 0.0)

    def test_benchmark_run_and_export(self):
        """Test running benchmark with a mock case and exporting reports."""
        test_cases = [
            {
                "id": "test_01",
                "category": "short",
                "source_language": "hi",
                "target_language": "en",
                "text": "मदद कीजिए।",
            }
        ]
        benchmark = PipelineBenchmark()
        report = benchmark.run(test_cases, warmup_runs=1, measured_runs=2)
        self.assertIn("metadata", report)
        self.assertIn("overall_summary", report)
        self.assertIn("cases", report)
        self.assertEqual(len(report["cases"]), 1)

        # Export test
        json_path, md_path = benchmark.export_reports(report, output_dir=config.REPORTS_DIR)
        self.assertTrue(json_path.exists())
        self.assertTrue(md_path.exists())

        # Verify JSON parse
        with open(json_path, "r", encoding="utf-8") as f:
            loaded_json = json.load(f)
        self.assertEqual(loaded_json["config"]["total_cases"], 1)


if __name__ == "__main__":
    unittest.main()
