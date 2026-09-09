"""
iTantra Receiver Web App - Two-Pass Emergency Detector (web_app/pipeline/emergency_detector.py)
=============================================================================================
Deterministic, auditable emergency detector supporting 10 Indian languages.
Performs two passes:
1. First pass on received source text.
2. Second pass on translated target text.
High-priority elevation to ALERT or SOS with category and matched term isolation.
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple
from web_app.config.settings import LEXICONS_DIR, LanguageCode, MessageType
from web_app.schemas.message import EmergencyResult

class EmergencyDetector:
    """
    Deterministic, dictionary & regex-based emergency detector.
    Zero neural hallucination, ultra-low latency (< 2 ms).
    """

    def __init__(self, lexicons_dir: Path = LEXICONS_DIR):
        self.lexicons_dir = Path(lexicons_dir)
        self.lexicons: Dict[str, Dict[str, List[str]]] = {}
        self._load_all_lexicons()

    def _load_all_lexicons(self):
        for lang in LanguageCode:
            lang_key = lang.value.lower()
            file_path = self.lexicons_dir / f"{lang_key}.json"
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        self.lexicons[lang_key] = json.load(f)
                except Exception as e:
                    print(f"[EmergencyDetector] Warning: could not load {file_path}: {e}")
            else:
                self.lexicons[lang_key] = {}

    def normalize_text(self, text: str) -> str:
        if not text:
            return ""
        # Unicode normalization (NFKC)
        normalized = unicodedata.normalize("NFKC", text)
        # Lowercase
        lowered = normalized.lower()
        # Clean extra punctuation but preserve Indian script words
        cleaned = re.sub(r"[!?,;:\-\.\(\)\[\]\"]+", " ", lowered)
        return re.sub(r"\s+", " ", cleaned).strip()

    def detect_single_pass(self, text: str, lang_code: str) -> Tuple[bool, MessageType, List[str], List[str], float]:
        """
        Runs single pass over text for given language.
        Returns: (is_emergency, priority, matched_terms, categories, confidence)
        """
        norm_text = self.normalize_text(text)
        if not norm_text:
            return False, MessageType.NORMAL, [], [], 0.0

        lang_key = lang_code.lower()
        lexicon = self.lexicons.get(lang_key, {})
        en_lexicon = self.lexicons.get("en", {})

        matched_terms: Set[str] = set()
        matched_categories: Set[str] = set()

        # Check language-specific dictionary
        for category, terms in lexicon.items():
            for term in terms:
                norm_term = self.normalize_text(term)
                if norm_term and (norm_term in norm_text or re.search(rf"\b{re.escape(norm_term)}\b", norm_text)):
                    matched_terms.add(term)
                    matched_categories.add(category)

        # Also check English SOS/HELP terms as global fallback (often used across all languages)
        if lang_key != "en":
            for term in ["sos", "help", "emergency", "danger"]:
                if term in norm_text or re.search(rf"\b{term}\b", norm_text):
                    matched_terms.add(term)
                    matched_categories.add("SOS" if term == "sos" else "HELP")

        # Check coordinate patterns (GPS coordinates: lat, long)
        coord_pattern = r"(\b-?\d{1,3}\.\d{3,},\s*-?\d{1,3}\.\d{3,}\b)"
        if re.search(coord_pattern, text):
            matched_terms.add("GPS_COORDINATES")
            matched_categories.add("COORDINATES")

        if not matched_terms:
            return False, MessageType.NORMAL, [], [], 0.0

        # Prioritization
        priority = MessageType.ALERT
        if "SOS" in matched_categories or "FIRE" in matched_categories or "TRAPPED" in matched_categories or "UNCONSCIOUS" in matched_categories:
            priority = MessageType.SOS

        confidence = min(1.0, 0.5 + (len(matched_terms) * 0.25))
        return True, priority, list(matched_terms), list(matched_categories), confidence

    def detect_two_pass(
        self,
        source_text: str,
        source_lang: str,
        translated_text: str = "",
        target_lang: str = "",
    ) -> EmergencyResult:
        """
        Two-pass detection on both original and translated text.
        """
        # Pass 1: Source
        is_em_src, prio_src, terms_src, cats_src, conf_src = self.detect_single_pass(source_text, source_lang)

        # Pass 2: Target
        is_em_tgt, prio_tgt, terms_tgt, cats_tgt, conf_tgt = False, MessageType.NORMAL, [], [], 0.0
        if translated_text and target_lang and target_lang.lower() != source_lang.lower():
            is_em_tgt, prio_tgt, terms_tgt, cats_tgt, conf_tgt = self.detect_single_pass(translated_text, target_lang)

        if not is_em_src and not is_em_tgt:
            return EmergencyResult(
                is_emergency=False,
                priority=MessageType.NORMAL,
                matched_terms=[],
                categories=[],
                confidence=0.0,
                detected_pass="none",
            )

        # Combine results
        combined_terms = list(set(terms_src + terms_tgt))
        combined_cats = list(set(cats_src + cats_tgt))
        combined_conf = max(conf_src, conf_tgt)

        priority = MessageType.ALERT
        if prio_src == MessageType.SOS or prio_tgt == MessageType.SOS:
            priority = MessageType.SOS

        detected_pass = "both" if (is_em_src and is_em_tgt) else ("source" if is_em_src else "translated")

        return EmergencyResult(
            is_emergency=True,
            priority=priority,
            matched_terms=combined_terms,
            categories=combined_cats,
            confidence=combined_conf,
            detected_pass=detected_pass,
        )
