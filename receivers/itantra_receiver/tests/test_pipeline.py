"""
Unit & Integration Tests for pipeline.py
Tests Section 42 return contract, micro-stage timings, audio metrics, and error propagation.
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

    def test_section_42_contract_and_execution(self):
        """Test full pipeline run producing valid Section 42 data contract."""
        result = self.pipeline.translate_and_speak(
            text="मदद कीजिए।",
            source_language="hi",
            target_language="en",
        )
        self.assertIsInstance(result, dict)

        # 1. Input block
        self.assertIn("input", result)
        self.assertEqual(result["input"]["text"], "मदद कीजिए।")
        self.assertEqual(result["input"]["language"], "hi")

        # 2. Translation block
        self.assertIn("translation", result)
        trans = result["translation"]
        self.assertEqual(trans["text"], "Please help.")
        self.assertEqual(trans["language"], "en")
        self.assertIsInstance(trans["segments"], list)
        self.assertEqual(trans["beam_size"], 1)
        self.assertIn("validation", trans)
        self.assertTrue(trans["validation"]["valid"])

        # 3. TTS block
        self.assertIn("tts", result)
        tts = result["tts"]
        self.assertEqual(tts["language"], "en")
        self.assertIn("voice_id", tts)
        self.assertIn("accent_id", tts)
        self.assertIn("speaker_id", tts)
        self.assertTrue(Path(tts["audio_path"]).exists())
        self.assertEqual(tts["sample_rate"], config.TTS_SAMPLE_RATE)
        self.assertGreater(tts["duration_ms"], 0.0)
        self.assertIn("rtf", tts)
        self.assertIn("peak_amplitude", tts)
        self.assertIn("rms", tts)
        self.assertIn("clipping_samples", tts)

        # 4. Timing block
        self.assertIn("timing", result)
        timing = result["timing"]
        for key in [
            "normalization_ms",
            "segmentation_ms",
            "tokenization_ms",
            "translation_ms",
            "detokenization_ms",
            "validation_ms",
            "tts_preprocess_ms",
            "tts_inference_ms",
            "tts_postprocess_ms",
            "audio_write_ms",
            "formatting_ms",
            "total_ms",
        ]:
            self.assertIn(key, timing)
            self.assertGreaterEqual(timing[key], 0.0)

        self.assertGreater(timing["total_ms"], 0.0)

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
