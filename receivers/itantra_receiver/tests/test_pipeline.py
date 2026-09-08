"""
Unit & Integration Tests for pipeline.py
Tests end-to-end execution, timing accuracy, payload contracts, and error propagation.
"""

import unittest
from pathlib import Path
import config
from formatter import ReceiverFormatter
from pipeline import ReceiverPipeline
from translation import OnDemandTranslator, MockTranslationAdapter, InvalidInputError, UnsupportedLanguageError
from tts import OfflineTTS, MockTTSAdapter


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.translator = OnDemandTranslator(adapter=MockTranslationAdapter())
        self.tts = OfflineTTS(adapter=MockTTSAdapter())
        self.formatter = ReceiverFormatter()
        self.pipeline = ReceiverPipeline(
            translator=self.translator,
            tts=self.tts,
            formatter=self.formatter,
            output_dir=config.TEMP_AUDIO_DIR,
        )

    def test_successful_execution_and_contract(self):
        """Test full pipeline run producing valid data contract and audio file."""
        result = self.pipeline.translate_and_speak(
            text="मदद कीजिए।",
            source_language="hi",
            target_language="en",
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(result["source_text"], "मदद कीजिए।")
        self.assertEqual(result["translated_text"], "Please help.")
        self.assertEqual(result["source_language"], "hi")
        self.assertEqual(result["target_language"], "en")

        # Audio checks
        self.assertIn("audio", result)
        self.assertTrue(Path(result["audio"]["path"]).exists())
        self.assertEqual(result["audio"]["sample_rate"], config.TTS_SAMPLE_RATE)
        self.assertGreater(result["audio"]["duration_ms"], 0.0)

        # Formatted output checks
        self.assertIn("formatted_output", result)
        self.assertEqual(result["formatted_output"]["display"]["status"], "NORMAL")

        # Timing checks
        timing = result["timing"]
        self.assertIn("translation_latency_ms", timing)
        self.assertIn("formatter_latency_ms", timing)
        self.assertIn("tts_latency_ms", timing)
        self.assertIn("audio_write_latency_ms", timing)
        self.assertIn("total_latency_ms", timing)
        self.assertGreater(timing["total_latency_ms"], 0.0)

        # Benchmark RTF check
        benchmark = result["benchmark"]
        self.assertIn("tts_rtf", benchmark)
        self.assertIn("end_to_end_rtf", benchmark)

    def test_emergency_pipeline_run(self):
        """Test pipeline run with emergency flags passed through."""
        emergency_meta = {
            "is_emergency": True,
            "priority": "P1",
            "matched_keywords": ["help", "doctor"],
        }
        result = self.pipeline.translate_and_speak(
            text="मदद कीजिए।",
            source_language="hi",
            target_language="en",
            emergency_result=emergency_meta,
        )
        self.assertTrue(result["formatted_output"]["emergency"]["is_emergency"])
        self.assertEqual(result["formatted_output"]["emergency"]["priority"], "P1")

    def test_empty_input_validation(self):
        """Test that empty string raises InvalidInputError."""
        with self.assertRaises(InvalidInputError):
            self.pipeline.translate_and_speak("", "hi", "en")

    def test_unsupported_language_error(self):
        """Test that invalid language code raises UnsupportedLanguageError."""
        with self.assertRaises(UnsupportedLanguageError):
            self.pipeline.translate_and_speak("Help", "es", "en")


if __name__ == "__main__":
    unittest.main()
