"""
Unit Tests for benchmark.py
Tests statistical calculations (p95, mean, stddev), Cold and Warm benchmark modes,
quality warning logging, and report exports.
"""

import json
import unittest
from pathlib import Path
from benchmark import calculate_statistics, PipelineBenchmark
import config


class TestBenchmark(unittest.TestCase):
    def test_statistics_calculation(self):
        """Test exact percentile, mean, median, std_dev computation."""
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

    def test_benchmark_warm_and_cold_runs(self):
        """Test running warm and cold benchmark runs and exporting reports."""
        test_cases = [
            {
                "id": "test_01",
                "category": "short",
                "source_language": "hi",
                "target_language": "en",
                "text": "मदद कीजिए।",
                "reference": "Please help.",
            }
        ]
        benchmark = PipelineBenchmark()
        
        # Warm run
        warm_report = benchmark.run(test_cases, warmup_runs=1, measured_runs=2, cold_start=False)
        self.assertIn("metadata", warm_report)
        self.assertIn("overall_summary", warm_report)
        self.assertIn("cases", warm_report)
        self.assertEqual(len(warm_report["cases"]), 1)
        self.assertIn("translation_validation", warm_report["cases"][0])

        # Cold run
        cold_report = benchmark.run(test_cases, warmup_runs=0, measured_runs=1, cold_start=True)
        self.assertTrue(cold_report["config"]["cold_start"])

        # Export test
        json_path, md_path = benchmark.export_reports(warm_report, output_dir=config.REPORTS_DIR)
        self.assertTrue(json_path.exists())
        self.assertTrue(md_path.exists())

        # Verify JSON content
        with open(json_path, "r", encoding="utf-8") as f:
            loaded_json = json.load(f)
        self.assertEqual(loaded_json["config"]["total_cases"], 1)


if __name__ == "__main__":
    unittest.main()
