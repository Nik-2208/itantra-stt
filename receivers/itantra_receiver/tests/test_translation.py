"""
Unit & Integration Tests for translation.py
Tests input validation, chunking, language routing, error handling, and adapter contracts.
"""

import unittest
from pathlib import Path
import config
from translation import (
    OnDemandTranslator,
    MockTranslationAdapter,
    IndicTrans2ONNXAdapter,
    TranslationError,
    ModelNotFoundError,
    UnsupportedLanguageError,
    InvalidInputError,
)


class TestTranslation(unittest.TestCase):
    def setUp(self):
        self.translator = OnDemandTranslator(
            source_language="hi",
            target_language="en",
            beam_size=1,
            adapter=MockTranslationAdapter(),
        )

    def test_normal_translation(self):
        """Test standard translation execution and return structure."""
        result = self.translator.translate("मदद कीजिए।")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["source_language"], "hi")
        self.assertEqual(result["target_language"], "en")
        self.assertEqual(result["input_text"], "मदद कीजिए।")
        self.assertEqual(result["output_text"], "Please help.")
        self.assertGreaterEqual(result["latency_ms"], 0.0)
        self.assertEqual(result["beam_size"], 1)

    def test_empty_input_error(self):
        """Test that empty or whitespace-only inputs raise InvalidInputError."""
        with self.assertRaises(InvalidInputError):
            self.translator.translate("")
        with self.assertRaises(InvalidInputError):
            self.translator.translate("   ")

    def test_unsupported_language_error(self):
        """Test that unsupported language codes raise UnsupportedLanguageError."""
        with self.assertRaises(UnsupportedLanguageError):
            self.translator.set_languages("fr", "en")
        with self.assertRaises(UnsupportedLanguageError):
            self.translator.set_languages("hi", "de")

    def test_same_language_pass_through(self):
        """Test same source and target language pass-through behavior."""
        self.translator.set_languages("en", "en")
        result = self.translator.translate("This is an alert.")
        self.assertEqual(result["output_text"], "This is an alert.")
        self.assertEqual(result["latency_ms"], 0.0)

    def test_long_input_chunking(self):
        """Test deterministic sentence chunking on long text."""
        long_text = "पहला वाक्य है। दूसरा वाक्य भी बहुत लंबा है! तीसरा वाक्य यहाँ समाप्त होता है।"
        chunks = self.translator._chunk_text(long_text)
        self.assertIsInstance(chunks, list)
        self.assertGreaterEqual(len(chunks), 1)
        # Translation across chunks should combine seamlessly
        result = self.translator.translate(long_text)
        self.assertIsInstance(result["output_text"], str)

    def test_beam_sizes(self):
        """Test valid beam sizes."""
        for beam in [1, 4, 8]:
            self.translator.set_beam_size(beam)
            self.assertEqual(self.translator.beam_size, beam)
        with self.assertRaises(InvalidInputError):
            self.translator.set_beam_size(16)

    def test_missing_model_error_detection(self):
        """Test that requesting a missing ONNX model path raises ModelNotFoundError."""
        non_existent_path = Path("models/translation/non_existent_pair")
        with self.assertRaises(ModelNotFoundError):
            IndicTrans2ONNXAdapter(non_existent_path)


if __name__ == "__main__":
    unittest.main()
