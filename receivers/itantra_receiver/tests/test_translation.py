"""
Unit & Integration Tests for translation.py
Tests TokenizerAdapter, TranslationValidator, TextSegmenter, context translation,
partial text policy, and sequence translation contracts.
"""

import unittest
from pathlib import Path
import config
from translation import (
    OnDemandTranslator,
    TokenizerAdapter,
    TranslationValidator,
    TextSegmenter,
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
        self.tokenizer = TokenizerAdapter()
        self.validator = TranslationValidator()

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
        self.assertIn("validation", result)
        self.assertTrue(result["validation"]["valid"])

    def test_tokenizer_adapter_encode_decode(self):
        """Test TokenizerAdapter encoding, decoding, token counting, and special tokens."""
        text = "Hello world, emergency response."
        tokens = self.tokenizer.encode(text)
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

        # BOS and EOS tokens present
        self.assertEqual(tokens[0], self.tokenizer.bos_id)
        self.assertEqual(tokens[-1], self.tokenizer.eos_id)

        count = self.tokenizer.count_tokens(text)
        self.assertGreater(count, 0)

    def test_detokenization_artifact_cleansing(self):
        """Test that detokenization strips SentencePiece \u2581, ##, @@, <unk>, and ▁."""
        dirty_output = "\u2581Ple@@ ase\u2581help## ful\u2581now <unk> ▁."
        cleaned = TokenizerAdapter.clean_detokenized_text(dirty_output)
        self.assertNotIn("\u2581", cleaned)
        self.assertNotIn("@@", cleaned)
        self.assertNotIn("##", cleaned)
        self.assertNotIn("<unk>", cleaned)
        self.assertNotIn("▁", cleaned)
        self.assertEqual(cleaned, "Please helpful now.")

    def test_partial_stt_text_policy(self):
        """Section 3: Partial STT text hypotheses must NOT trigger translation."""
        partial_text = "मद"
        res = self.translator.translate(partial_text, is_final=False)
        self.assertFalse(res.get("is_final", True))
        self.assertEqual(res["output_text"], partial_text)
        self.assertEqual(res["latency_ms"], 0.0)

    def test_word_safe_segmentation(self):
        """Section 4: Segmentation must never cut inside lexical words."""
        long_sentence = "This is a sentence with multiple important keywords that should be segmented safely."
        chunks = TextSegmenter.segment_sentences(long_sentence, max_chars=30)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 35)
            # Ensure words are complete
            for word in chunk.split():
                self.assertIn(word, long_sentence)

    def test_translation_validator_number_mismatch(self):
        """Section 10 & 11: Detects missing numerical sequences."""
        source = "Room 108 is on fire."
        bad_trans = "The room is on fire."
        res = self.validator.validate(source, bad_trans, "en", "en")
        self.assertFalse(res["valid"])
        self.assertTrue(res["checks"]["number_mismatch"])

    def test_translation_validator_token_artifacts(self):
        """Section 11: Detects token artifacts in translation output."""
        source = "मदद कीजिए।"
        bad_trans = "Please help <unk> ▁."
        res = self.validator.validate(source, bad_trans, "hi", "en")
        self.assertFalse(res["valid"])
        self.assertTrue(res["checks"]["token_artifact"])

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

    def test_beam_sizes(self):
        """Test valid beam sizes."""
        for beam in [1, 4, 8]:
            self.translator.set_beam_size(beam)
            self.assertEqual(self.translator.beam_size, beam)
        with self.assertRaises(InvalidInputError):
            self.translator.set_beam_size(16)


if __name__ == "__main__":
    unittest.main()
