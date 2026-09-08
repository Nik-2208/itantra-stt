"""
Unit & Integration Tests for app.py
Tests Gradio UI component creation, event handlers, and benchmark UI runner.
"""

import unittest
from pathlib import Path
from app import create_ui, process_translate_and_speak, run_benchmark_ui


class TestApp(unittest.TestCase):
    def test_ui_creation(self):
        """Ensure Gradio Blocks UI is constructed without errors."""
        demo = create_ui()
        self.assertIsNotNone(demo)

    def test_process_translate_and_speak_handler(self):
        """Test event handler for translate and speak."""
        trans_text, audio_path, duration_str, summary, json_str, latency_tbl, log_str = process_translate_and_speak(
            transcript="मदद कीजिए।",
            source_lang="hi",
            target_lang="en",
            beam_choice="Greedy / Beam 1",
            is_emergency=False,
            emergency_keywords_str="",
        )
        self.assertEqual(trans_text, "Please help.")
        self.assertTrue(Path(audio_path).exists())
        self.assertIn("Please help.", summary)
        self.assertIn("Translation", latency_tbl)
        self.assertIn("Pipeline", log_str)

    def test_benchmark_ui_handler(self):
        """Test event handler for benchmark execution from UI."""
        table_md, json_p, md_p = run_benchmark_ui(
            custom_text="मदद कीजिए।",
            source_lang="hi",
            target_lang="en",
            warmup_runs=1,
            measured_runs=2,
        )
        self.assertIn("Benchmark Aggregate Results", table_md)
        self.assertTrue(Path(json_p).exists())
        self.assertTrue(Path(md_p).exists())


if __name__ == "__main__":
    unittest.main()
