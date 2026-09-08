"""
Unit & Integration Tests for tts.py
Tests PronunciationProcessor, real voice/accent configuration, audio quality validation,
anti-clipping, metrics calculation, and WAV export.
"""

import unittest
from pathlib import Path
import numpy as np
import soundfile as sf
import config
from tts import (
    OfflineTTS,
    MockTTSAdapter,
    PronunciationProcessor,
    IndicTTSFastPitchVitsONNXAdapter,
    TTSError,
    ModelNotFoundError,
    UnsupportedLanguageError,
    InvalidInputError,
    AudioQualityError,
)


class TestTTS(unittest.TestCase):
    def setUp(self):
        self.tts = OfflineTTS(language_code="en", adapter=MockTTSAdapter())
        self.processor = PronunciationProcessor()

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

    def test_pronunciation_processor_lexicon(self):
        """Test pronunciation processor expands abbreviations and units locally."""
        text = "Patient in ICU requires CPR immediately."
        preprocessed = self.processor.preprocess(text, "en")
        self.assertIn("I C U", preprocessed)
        self.assertIn("C P R", preprocessed)

    def test_real_voice_and_accent_mapping(self):
        """Test that selecting local voice ID updates voice_config and accent_id."""
        self.tts.set_voice("en", "en_in_heera")
        self.assertEqual(self.tts.voice_config["accent_id"], "indian_english")
        self.assertEqual(self.tts.voice_config["speaker_id"], "female_01")

        self.tts.set_voice("en", "en_us_david")
        self.assertEqual(self.tts.voice_config["accent_id"], "us_english")

    def test_audio_quality_metrics_recording(self):
        """Test that synthesis computes RMS, peak amplitude, duration, and RTF."""
        self.tts.synthesize("Emergency medical response.")
        metrics = self.tts.get_last_metrics()
        self.assertIn("duration_ms", metrics)
        self.assertIn("peak_amplitude", metrics)
        self.assertIn("rms", metrics)
        self.assertIn("clipping_samples", metrics)
        self.assertIn("rtf", metrics)
        self.assertGreater(metrics["duration_ms"], 0.0)
        self.assertLessEqual(metrics["peak_amplitude"], 1.0)
        self.assertGreaterEqual(metrics["rms"], 0.0)

    def test_sample_rate_exposed(self):
        """Ensure sample rate property matches config."""
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

    def test_save_audio_wav(self):
        """Ensure synthesized audio can be saved to a valid WAV file."""
        audio = self.tts.synthesize("Test audio export.")
        out_file = config.TEMP_AUDIO_DIR / "test_unit_audio.wav"
        saved_path, write_ms = self.tts.save_audio(audio, out_file)
        self.assertTrue(Path(saved_path).exists())
        self.assertGreater(write_ms, 0.0)

        read_audio, sr = sf.read(saved_path)
        self.assertEqual(sr, self.tts.sample_rate)
        self.assertEqual(len(read_audio), len(audio))

    def test_missing_model_error_detection(self):
        """Ensure non-existent ONNX model path raises ModelNotFoundError."""
        non_existent_path = Path("models/tts/non_existent_lang")
        with self.assertRaises(ModelNotFoundError):
            IndicTTSFastPitchVitsONNXAdapter(non_existent_path)


if __name__ == "__main__":
    unittest.main()
