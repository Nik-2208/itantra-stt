"""
Module 3: Emergency Classifier (EmergencyClassifier)
=====================================================
Rule-based, zero-ML keyword classifier matching STT hypothesis text against
per-language keyword lists loaded from JSON files.

Flags P0 (Emergency) vs P2 (Normal) instantly on both partial and final transcripts.
"""

import json
import logging
from pathlib import Path
import re

try:
    from .config import KEYWORDS_DIR, DEFAULT_LANGUAGE, PRIORITY_EMERGENCY, PRIORITY_NORMAL
except ImportError:
    from config import KEYWORDS_DIR, DEFAULT_LANGUAGE, PRIORITY_EMERGENCY, PRIORITY_NORMAL

logger = logging.getLogger("EmergencyClassifier")


class EmergencyClassifier:
    """
    Rule-based emergency keyword matching engine operating on per-language JSON specifications.
    """

    def __init__(self, keywords_dir: str = None):
        self.keywords_dir = Path(keywords_dir) if keywords_dir else Path(KEYWORDS_DIR)
        self.active_language = None
        self.emergency_terms = []
        self.location_terms = []
        self.numbers = []
        self.needs_review = False

        self.load_language(DEFAULT_LANGUAGE)

    def load_language(self, language_code: str):
        """Loads per-language keyword dictionary from JSON file."""
        self.active_language = language_code
        json_path = self.keywords_dir / f"{language_code}.json"

        if not json_path.exists():
            logger.warning(f"Keyword file {json_path} not found. Initializing empty rules.")
            self.emergency_terms = []
            self.location_terms = []
            self.numbers = []
            self.needs_review = True
            return

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.emergency_terms = [str(term).strip().lower() for term in data.get("emergency_terms", []) if term]
            self.location_terms = [str(term).strip().lower() for term in data.get("location_terms", []) if term]
            self.numbers = [str(term).strip().lower() for term in data.get("numbers", []) if term]
            self.needs_review = data.get("NEEDS_NATIVE_SPEAKER_REVIEW", False)

            logger.info(
                f"Loaded language '{language_code}': {len(self.emergency_terms)} emergency terms, "
                f"{len(self.location_terms)} location terms. (Needs Native Review: {self.needs_review})"
            )
        except Exception as e:
            logger.error(f"Error reading {json_path}: {e}")
            self.emergency_terms = []
            self.location_terms = []
            self.numbers = []
            self.needs_review = True

    def classify(self, text: str) -> dict:
        """
        Classifies input text string (partial hypothesis or final transcript).
        Returns dict with:
          - is_emergency: bool
          - matched_keywords: list of matched terms
          - priority: "P0" (Emergency) or "P2" (Normal)
        """
        if not text:
            return {
                "is_emergency": False,
                "matched_keywords": [],
                "priority": PRIORITY_NORMAL,
                "needs_review": self.needs_review,
            }

        # Case-insensitive normalization
        normalized_text = text.lower()
        matched_emergency = []
        matched_location = []
        matched_numbers = []

        # Extract individual tokens (including Devanagari and Latin script words)
        tokens = set(re.findall(r"[\w\u0900-\u097F]+", normalized_text))

        def matches_term(term: str) -> bool:
            if not term:
                return False
            if " " in term:
                # Multi-word phrase: match with boundary/whitespace
                pattern = r"(?:\b|\s|^)" + re.escape(term) + r"(?:\b|\s|$|[।,?!])"
                return bool(re.search(pattern, normalized_text))
            else:
                # Single word: match exact token (prevents 'आग' matching inside 'आगरा')
                return term in tokens or term in normalized_text.split()

        # Check emergency terms
        for term in self.emergency_terms:
            if matches_term(term):
                matched_emergency.append(term)

        # Check location terms
        for term in self.location_terms:
            if matches_term(term):
                matched_location.append(term)

        # Check numbers
        for term in self.numbers:
            if matches_term(term):
                matched_numbers.append(term)

        all_matched = list(dict.fromkeys(matched_emergency + matched_location + matched_numbers))
        is_emergency = len(matched_emergency) > 0
        priority = PRIORITY_EMERGENCY if is_emergency else PRIORITY_NORMAL

        return {
            "is_emergency": is_emergency,
            "matched_keywords": all_matched,
            "matched_emergency": matched_emergency,
            "matched_location": matched_location,
            "matched_numbers": matched_numbers,
            "priority": priority,
            "needs_review": self.needs_review,
        }
