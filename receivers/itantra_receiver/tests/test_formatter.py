"""
Unit Tests for formatter.py
Tests formatting contract, emergency metadata handling, and language display names.
"""

import unittest
from formatter import ReceiverFormatter


class TestFormatter(unittest.TestCase):
    def setUp(self):
        self.formatter = ReceiverFormatter()

    def test_normal_formatting(self):
        """Test formatting of standard non-emergency translation output."""
        output = self.formatter.format_output(
            source_text="मदद कीजिए",
            translated_text="Please help.",
            source_language="hi",
            target_language="en",
        )
        self.assertIsInstance(output, dict)
        self.assertEqual(output["source"]["language"], "hi")
        self.assertEqual(output["source"]["language_name"], "Hindi")
        self.assertEqual(output["source"]["text"], "मदद कीजिए")

        self.assertEqual(output["translation"]["language"], "en")
        self.assertEqual(output["translation"]["language_name"], "English")
        self.assertEqual(output["translation"]["text"], "Please help.")

        self.assertFalse(output["emergency"]["is_emergency"])
        self.assertEqual(output["emergency"]["priority"], "NORMAL")
        self.assertEqual(output["display"]["status"], "NORMAL")
        self.assertIn("timestamp_iso", output)

    def test_emergency_formatting(self):
        """Test formatting with emergency flags and keywords."""
        emergency_meta = {
            "is_emergency": True,
            "priority": "P1",
            "matched_keywords": ["police", "fire", "help"],
        }
        output = self.formatter.format_output(
            source_text="आपातकाल! तुरंत पुलिस को बुलाएं।",
            translated_text="Emergency! Call the police immediately.",
            source_language="hi",
            target_language="en",
            emergency_result=emergency_meta,
        )
        self.assertTrue(output["emergency"]["is_emergency"])
        self.assertEqual(output["emergency"]["priority"], "P1")
        self.assertEqual(output["emergency"]["matched_keywords"], ["police", "fire", "help"])
        self.assertEqual(output["display"]["status"], "EMERGENCY")
        self.assertIn("EMERGENCY", output["display"]["formatted_summary"])

    def test_empty_translation(self):
        """Test formatting when translated text is empty."""
        output = self.formatter.format_output(
            source_text="नमस्ते",
            translated_text="",
            source_language="hi",
            target_language="en",
        )
        self.assertEqual(output["source"]["text"], "नमस्ते")
        self.assertEqual(output["translation"]["text"], "")
        self.assertEqual(output["display"]["body"], "[Translation Unavailable]")


if __name__ == "__main__":
    unittest.main()
