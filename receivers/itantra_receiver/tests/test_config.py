"""
Unit Tests for config.py
Validates configuration integrity, types, values, and required constants.
"""

import unittest
from pathlib import Path
import config


class TestConfig(unittest.TestCase):
    def test_providers_strictly_cpu(self):
        """Ensure ONNX providers strictly use CPUExecutionProvider and exclude GPU."""
        self.assertEqual(config.ONNX_PROVIDERS, ["CPUExecutionProvider"])
        self.assertNotIn("CUDAExecutionProvider", config.ONNX_PROVIDERS)
        self.assertNotIn("TensorrtExecutionProvider", config.ONNX_PROVIDERS)
        self.assertNotIn("DirectMLExecutionProvider", config.ONNX_PROVIDERS)

    def test_translation_hyperparameters(self):
        """Validate translation constants and constraints."""
        self.assertIn(config.TRANSLATION_BEAM_SIZE, [1, 4, 8])
        self.assertIsInstance(config.TRANSLATION_MAX_INPUT_TOKENS, int)
        self.assertGreater(config.TRANSLATION_MAX_INPUT_TOKENS, 0)
        self.assertIsInstance(config.TRANSLATION_MAX_OUTPUT_TOKENS, int)
        self.assertGreater(config.TRANSLATION_MAX_OUTPUT_TOKENS, 0)
        self.assertIsInstance(config.TRANSLATION_NUM_THREADS, int)
        self.assertGreaterEqual(config.TRANSLATION_NUM_THREADS, 1)

    def test_tts_hyperparameters(self):
        """Validate TTS constants."""
        self.assertIsInstance(config.TTS_SAMPLE_RATE, int)
        self.assertGreater(config.TTS_SAMPLE_RATE, 0)
        self.assertGreater(config.TTS_SPEED, 0.0)
        self.assertGreater(config.TTS_VOLUME, 0.0)
        self.assertGreater(config.TTS_MAX_TEXT_LENGTH, 0)

    def test_audio_specifications(self):
        """Validate audio channel, rate, format constants."""
        self.assertEqual(config.AUDIO_CHANNELS, 1)
        self.assertEqual(config.AUDIO_DTYPE, "float32")
        self.assertEqual(config.OUTPUT_AUDIO_FORMAT, "wav")

    def test_benchmark_constants(self):
        """Validate benchmarking parameters."""
        self.assertGreaterEqual(config.BENCHMARK_WARMUP_RUNS, 0)
        self.assertGreaterEqual(config.BENCHMARK_MEASURED_RUNS, 1)

    def test_supported_languages(self):
        """Validate supported language list and mapping."""
        expected_langs = ["hi", "gu", "mr", "kn", "ml", "ta", "te", "or", "bn", "en"]
        self.assertEqual(config.SUPPORTED_LANGUAGES, expected_langs)
        for lang in expected_langs:
            self.assertIn(lang, config.LANGUAGE_NAMES)
            self.assertIn(lang, config.INDICTRANS2_LANG_TAGS)


if __name__ == "__main__":
    unittest.main()
