"""
Standalone test for emergency_classifier.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from emergency_classifier import EmergencyClassifier

def test_emergency_classifier():
    print("--- Running Test: EmergencyClassifier ---")
    classifier = EmergencyClassifier()

    # Test English emergency detection
    classifier.load_language("en")
    res_en = classifier.classify("Help! There is a fire on the third floor!")
    print("English Test Result:", res_en)
    assert res_en["is_emergency"] is True
    assert res_en["priority"] == "P0"
    assert "help" in res_en["matched_keywords"]
    assert "fire" in res_en["matched_keywords"]

    # Test Hindi emergency detection
    classifier.load_language("hi")
    res_hi = classifier.classify("मदद करो तीसरी मंजिल पर आग लगी है")
    print("Hindi Test Result:", res_hi)
    assert res_hi["is_emergency"] is True
    assert res_hi["priority"] == "P0"
    assert "मदद" in res_hi["matched_keywords"]
    assert "आग" in res_hi["matched_keywords"]

    # Test Non-emergency statement
    res_normal = classifier.classify("गुड मॉर्निंग आप कैसे हैं")
    print("Normal Test Result:", res_normal)
    assert res_normal["is_emergency"] is False
    assert res_normal["priority"] == "P2"

    print("--- Test EmergencyClassifier PASSED ---\n")

if __name__ == "__main__":
    test_emergency_classifier()
