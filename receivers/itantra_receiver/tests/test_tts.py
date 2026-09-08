"""
Unit & Integration Tests for tts.py
Tests audio synthesis, sample rate, dtype, non-clipping, WAV export, and error handling.
"""

import unittest
from pathlib import Path
import numpy as np
import soundfile as sf
import config
from tts import (
    OfflineTTS,
    MockTTSAdapter,
    IndicTTSFastPitchVitsONNXAdapter,
    TTSError,
    ModelNotFoundError,
    UnsupportedLanguageError,
    InvalidInputError,
)


class TestTTS(unittest.TestCase):
    def setUp(self):
        self.tts = OfflineTTS(language_code="en", adapter=MockTTSAdapter())

    def test_synthesize_normal(self):
        """Test standard synthesis returns valid float32 mono array."""
        audio = self.tts.synthesize("Please help.")
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(len(audio.shape), 1)  # Mono
        self.assertTrue(np.isfinite(audio).all())
        # Amplitude bounded
        self.assertLessEqual(np.max(np.abs(audio)), 1.0)
        self.assertGreater(len(audio), 0)

    def test_sample_rate_exposed(self):
        """Ensure sample rate property is exposed and matches config."""
        self.assertEqual(self.tts.sample_rate, config.TTS_SAMPLE_RATE)

    def test_empty_input_error(self):
        """Ensure empty input raises InvalidInputError."""
        with self.assertRaises(InvalidInputError):
            self.tts.synthesize("")
        with self.assertRaises(InvalidInputError):
            self.tts.synthesize("   ")

    def test_unsupported_language_error(self):
        """Ensure invalid language code raises UnsupportedLanguageError."""
        with self.assertRaises(UnsupportedLanguageError):
            self.tts.set_language("fr")

    def test_long_text_synthesis(self):
        """Ensure long text is synthesized without error or clipping."""
        long_text = "This is a long emergency announcement. " * 10
        audio = self.tts.synthesize(long_text)
        self.assertTrue(np.isfinite(audio).all())
        self.assertLessEqual(np.max(np.abs(audio)), 1.0)

    def test_save_audio_wav(self):
        """Ensure synthesized audio can be saved to a valid WAV file."""
        audio = self.tts.synthesize("Test audio export.")
        out_file = config.TEMP_AUDIO_DIR / "test_unit_audio.wav"
        saved_path, write_ms = self.tts.save_audio(audio, out_file)
        self.assertTrue(Path(saved_path).exists())
        self.assertGreater(write_ms, 0.0)

        # Verify WAV file contents with soundfile
        read_audio, sr = sf.read(saved_path)
        self.assertEqual(sr, self.tts.sample_rate)
        self.assertEqual(len(read_audio), len(audio))

    def test_timing_breakdown(self):
        """Ensure latency breakdown is populated after synthesis."""
        self.tts.synthesize("Check timing.")
        timing = self.tts.get_last_timing()
        self.assertIn("inference_ms", timing)
        self.assertIn("postprocess_ms", timing)
        self.assertIn("total_ms", timing)
        self.assertGreaterEqual(timing["total_ms"], 0.0)

    def test_missing_model_error_detection(self):
        """Ensure non-existent ONNX model path raises ModelNotFoundError."""
        non_existent_path = Path("models/tts/non_existent_lang")
        with self.assertRaises(ModelNotFoundError):
            IndicTTSFastPitchVitsONNXAdapter(non_existent_path)


if __name__ == "__main__":
    unittest.main()
