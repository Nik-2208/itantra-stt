"""
Regression Tests for Source Leakage (Section 28 Requirements)
=============================================================
Tests:
1. Exact Hindi-to-English leakage detection (FAIL with SOURCE_LEAKAGE, TTS skipped)
2. Valid translated output (PASS)
3. Same language passthrough (PASS with intentional bypass)
4. Legitimate proper nouns (iTantra -> iTantra)
5. Numbers (112 -> 112)
6. English-to-Hindi translation
7. Exception handling (translated_text == None, NEVER fallback to source)
8. Empty decoder output (FAIL with EMPTY_OUTPUT, TTS skipped)
9. Wrong language configuration (FAIL with UNSUPPORTED_LANGUAGE_PAIR, no source fallback)
"""

import unittest
from pathlib import Path
import config
from translation import (
    OnDemandTranslator,
    TranslationValidator,
    MockTranslationAdapter,
    TranslationModelAdapter,
    TokenizerAdapter,
    TranslationError,
    UnsupportedLanguageError,
    InvalidInputError,
)
from pipeline import ReceiverPipeline
from tts import OfflineTTS, MockTTSAdapter
from formatter import ReceiverFormatter


class TestSourceLeakageRegression(unittest.TestCase):
    def setUp(self):
        self.validator = TranslationValidator()
        self.translator = OnDemandTranslator(adapter=MockTranslationAdapter())
        self.pipeline = ReceiverPipeline(
            translator=self.translator,
            tts=OfflineTTS(adapter=MockTTSAdapter()),
            formatter=ReceiverFormatter(),
        )

    def test_01_exact_hindi_to_english_leakage(self):
        """Test 1: Exact Hindi-to-English leakage MUST FAIL with SOURCE_LEAKAGE and TTS must NOT run."""
        source_text = "मुझे मदद चाहिए।"
        # Simulate leaky translation returning Hindi when target is en
        leak_res = self.validator.detect_source_leakage(
            source_text=source_text,
            translated_text=source_text,
            source_language="hi",
            target_language="en",
        )
        self.assertTrue(leak_res["is_leakage"])
        self.assertGreaterEqual(leak_res["confidence"], 0.90)

        # In pipeline: if mock or model leaks Hindi to English, pipeline must catch it
        class LeakyAdapter(TranslationModelAdapter):
            def __init__(self):
                self.tokenizer = TokenizerAdapter()
            def is_ready(self) -> bool:
                return True
            def encode(self, text, src, tgt):
                return {"text": text}
            def infer(self, inputs, beam_size=1):
                return {"translated_text": inputs["text"]}
            def decode(self, outputs, tgt):
                return outputs["translated_text"]
            def get_token_counts(self):
                return 5, 5

        leaky_pipeline = ReceiverPipeline(
            translator=OnDemandTranslator(adapter=LeakyAdapter()),
            tts=OfflineTTS(adapter=MockTTSAdapter()),
            formatter=ReceiverFormatter(),
        )
        res = leaky_pipeline.translate_and_speak(
            text="मुझे मदद चाहिए।",
            source_language="hi",
            target_language="en",
        )
        self.assertEqual(res["status"], "TRANSLATION_FAILED")
        self.assertFalse(res["translation"]["success"])
        self.assertEqual(res["translation"]["error"], "SOURCE_LEAKAGE")
        self.assertIsNone(res["translation"]["translated_text"])
        self.assertIsNone(res["tts"])
        self.assertIsNone(res["audio"])

    def test_02_valid_translated_output(self):
        """Test 2: Valid translated output MUST PASS."""
        res = self.pipeline.translate_and_speak(
            text="मदद कीजिए।",
            source_language="hi",
            target_language="en",
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertTrue(res["translation"]["success"])
        self.assertEqual(res["translation"]["translated_text"], "Please help.")
        self.assertIsNotNone(res["tts"])
        self.assertIsNotNone(res["audio"])

    def test_03_same_language_passthrough(self):
        """Test 3: hi -> hi must not be treated as leakage; translation is skipped intentionally."""
        res = self.pipeline.translate_and_speak(
            text="मुझे मदद चाहिए।",
            source_language="hi",
            target_language="hi",
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertTrue(res["translation"]["success"])
        self.assertEqual(res["translation"]["translated_text"], "मुझे मदद चाहिए।")
        self.assertIn("SOURCE_AND_TARGET_LANGUAGE_IDENTICAL", res["translation"]["warnings"])
        self.assertIsNotNone(res["tts"])
        self.assertEqual(res["tts"]["language"], "hi")

    def test_04_proper_noun_preservation(self):
        """Test 4: Legitimate proper nouns (e.g. 'iTantra', 'COVID', 'SOS') must NOT automatically fail."""
        res = self.validator.detect_source_leakage(
            source_text="iTantra",
            translated_text="iTantra",
            source_language="hi",
            target_language="en",
        )
        self.assertFalse(res["is_leakage"])

        res_sos = self.validator.detect_source_leakage(
            source_text="SOS",
            translated_text="SOS",
            source_language="hi",
            target_language="en",
        )
        self.assertFalse(res_sos["is_leakage"])

    def test_05_numbers_preservation(self):
        """Test 5: Shared numbers (e.g. '112') must NOT automatically fail."""
        res = self.validator.detect_source_leakage(
            source_text="112",
            translated_text="112",
            source_language="hi",
            target_language="en",
        )
        self.assertFalse(res["is_leakage"])

    def test_06_english_to_hindi_translation(self):
        """Test 6: English-to-Hindi valid translation passes."""
        res = self.validator.detect_source_leakage(
            source_text="I need help.",
            translated_text="मुझे मदद चाहिए।",
            source_language="en",
            target_language="hi",
        )
        self.assertFalse(res["is_leakage"])

    def test_07_failure_fallback_never_returns_source_text(self):
        """Test 7: Exception during translation must return translated_text == None, NEVER source text."""
        class CrashingAdapter(TranslationModelAdapter):
            def is_ready(self) -> bool:
                return True
            def encode(self, text, src, tgt):
                raise RuntimeError("ONNX decoding crashed")
            def infer(self, inputs, beam_size=1):
                pass
            def decode(self, outputs, tgt):
                pass
            def get_token_counts(self):
                return None, None

        crashing_translator = OnDemandTranslator(adapter=CrashingAdapter())
        crashing_translator.set_languages("hi", "en")
        res = crashing_translator.translate("मदद कीजिए।")
        self.assertFalse(res["success"])
        self.assertIsNone(res["translated_text"])
        self.assertIsNotNone(res["error"])

    def test_08_empty_decoder_output(self):
        """Test 8: If decoder returns empty string, MUST FAIL with EMPTY_OUTPUT and TTS skipped."""
        class EmptyAdapter(TranslationModelAdapter):
            def is_ready(self) -> bool:
                return True
            def encode(self, text, src, tgt):
                return {"text": text}
            def infer(self, inputs, beam_size=1):
                return {"translated_text": "   "}
            def decode(self, outputs, tgt):
                return "   "
            def get_token_counts(self):
                return 0, 0

        empty_pipeline = ReceiverPipeline(
            translator=OnDemandTranslator(adapter=EmptyAdapter()),
            tts=OfflineTTS(adapter=MockTTSAdapter()),
            formatter=ReceiverFormatter(),
        )
        res = empty_pipeline.translate_and_speak(
            text="मदद कीजिए।",
            source_language="hi",
            target_language="en",
        )
        self.assertEqual(res["status"], "TRANSLATION_FAILED")
        self.assertFalse(res["translation"]["success"])
        self.assertEqual(res["translation"]["error"], "EMPTY_OUTPUT")
        self.assertIsNone(res["translation"]["translated_text"])
        self.assertIsNone(res["tts"])

    def test_09_wrong_language_configuration(self):
        """Test 9: Unsupported language pair must FAIL with UNSUPPORTED_LANGUAGE_PAIR and no source fallback."""
        with self.assertRaises(UnsupportedLanguageError):
            self.translator.set_languages("en", "fr")


if __name__ == "__main__":
    unittest.main()
