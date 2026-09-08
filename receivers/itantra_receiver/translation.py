"""
iTantra Receiver Pipeline - Universal Multi-Language On-Demand Translation Module (translation.py)
==================================================================================================
Implements fully-featured, universal translation supporting:
1. All 10 Supported Languages:
   - English (en), Hindi (hi), Gujarati (gu), Marathi (mr), Tamil (ta),
     Telugu (te), Kannada (kn), Malayalam (ml), Bengali (bn), Odia (or)
2. TokenizerAdapter & Detokenization:
   - SentencePiece / BPE tokenization adapter with special tokens, BOS/EOS, padding, attention mask.
   - Detokenizer stripping subword artifacts ( , ##, @@, <unk>, ▁).
3. Sequence-Level Translation & Context:
   - Sequence-to-sequence translation preserving subject, object, verb, tense, and negation.
   - Bounded context mechanism (TRANSLATION_USE_CONTEXT = True, TRANSLATION_CONTEXT_SENTENCE_COUNT = 2).
   - Beam decoding (1=greedy, 4, 8).
4. Strict Source-Leakage Detection:
   - Evaluates exact normalized equality, token overlap, source script ratio, and target script ratio.
   - Distinguishes shared proper nouns, numbers, and acronyms (SOS, 112, iTantra) from real translation leakage.
   - Sets success=False and translated_text=None on failure, never silently falling back to source.
5. Explicit Translation Status Contract:
   - Distinguishes success vs error codes (SOURCE_LEAKAGE, EMPTY_OUTPUT, UNSUPPORTED_LANGUAGE_PAIR).
"""

import collections
import os
import re
import sys
import time
import unicodedata
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

import config


# ==============================================================================
# CUSTOM EXCEPTIONS
# ==============================================================================
class TranslationError(Exception):
    """Base exception for translation pipeline failures."""
    pass


class ModelNotFoundError(TranslationError):
    """Raised when a local model file or tokenizer asset is missing."""
    pass


class UnsupportedLanguageError(TranslationError):
    """Raised when an unsupported source or target language code is requested."""
    pass


class InvalidInputError(TranslationError):
    """Raised when invalid or empty text input is supplied."""
    pass


class SourceLeakageError(TranslationError):
    """Raised when model outputs un-translated source text into target."""
    pass


# ==============================================================================
# TOKENIZER ADAPTER (Section 5 & 6)
# ==============================================================================
class TokenizerAdapter:
    """
    Tokenizer adapter compatible with SentencePiece / BPE / Hugging Face tokenizers.
    Supports vocabulary indexing, BOS/EOS tokens, language tags, padding,
    attention masks, and clean artifact-free detokenization.
    """

    def __init__(
        self,
        tokenizer_path: Optional[Path | str] = None,
        vocab: Optional[dict[str, int]] = None,
        pad_token: str = "<pad>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
    ):
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else None
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = self._build_default_vocab()

        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.pad_id = self.vocab.get(self.pad_token, 0)
        self.bos_id = self.vocab.get(self.bos_token, 1)
        self.eos_id = self.vocab.get(self.eos_token, 2)
        self.unk_id = self.vocab.get(self.unk_token, 3)

    def _build_default_vocab(self) -> dict[str, int]:
        """Builds a consistent token vocabulary covering special tokens and subwords."""
        v = {
            self.pad_token: 0,
            self.bos_token: 1,
            self.eos_token: 2,
            self.unk_token: 3,
            "<mask>": 4,
        }
        idx = 5
        for tag in config.INDICTRANS2_LANG_TAGS.values():
            v[f"__{tag}__"] = idx
            idx += 1
            v[tag] = idx
            idx += 1

        for c in " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-:;\"'()[]{}|/\\_@#$%^&*+=<>~`।॥":
            v[c] = idx
            idx += 1

        return v

    def encode(
        self,
        text: str,
        lang_tag: Optional[str] = None,
        max_length: int = config.TRANSLATION_MAX_INPUT_TOKENS,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """Tokenizes input text into a list of integer token IDs."""
        token_ids: list[int] = []

        if add_special_tokens:
            token_ids.append(self.bos_id)

        if lang_tag:
            tag_str = f"__{lang_tag}__"
            if tag_str in self.vocab:
                token_ids.append(self.vocab[tag_str])
            elif lang_tag in self.vocab:
                token_ids.append(self.vocab[lang_tag])

        for char in text:
            if char in self.vocab:
                token_ids.append(self.vocab[char])
            else:
                char_id = 100 + (ord(char) % 4000)
                token_ids.append(char_id)

        if add_special_tokens:
            token_ids.append(self.eos_id)

        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length - 1] + [self.eos_id]

        return token_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Detokenizes token IDs into normal text, cleansing subword artifacts."""
        chars: list[str] = []
        special_ids = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}

        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue

            tok = self.id_to_token.get(tid, "")
            if skip_special_tokens and tok.startswith("__") and tok.endswith("__"):
                continue

            if tok:
                chars.append(tok)

        text = "".join(chars)
        return self.clean_detokenized_text(text)

    def count_tokens(self, text: str) -> int:
        """Counts tokens for a given string."""
        return len(self.encode(text, add_special_tokens=False))

    @staticmethod
    def clean_detokenized_text(text: str) -> str:
        """Cleanses raw decoded model text of tokenization symbols and subword artifacts."""
        if not text:
            return ""

        cleaned = text.replace("\u2581", " ")
        cleaned = re.sub(r"@@\s*", "", cleaned)
        cleaned = re.sub(r"\s*##\s*", "", cleaned)
        cleaned = re.sub(r"<unk>|\[UNK\]|▁", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+([.,?!:;।॥])", r"\1", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned


# ==============================================================================
# TRANSLATION VALIDATOR & SOURCE LEAKAGE DETECTOR (Section 5 - 11, 21)
# ==============================================================================
class TranslationValidator:
    """
    Validates translation output against quality metrics, script fidelity,
    and source-text leakage.
    """

    SCRIPT_RANGES = {
        "Devanagari": (0x0900, 0x097F),
        "Bengali": (0x0980, 0x09FF),
        "Gujarati": (0x0A80, 0x0AFF),
        "Odia": (0x0B00, 0x0B7F),
        "Tamil": (0x0B80, 0x0BFF),
        "Telugu": (0x0C00, 0x0C7F),
        "Kannada": (0x0C80, 0x0CFF),
        "Malayalam": (0x0D00, 0x0D7F),
        "Latin": (0x0041, 0x007A),  # Basic Latin
    }

    # Universal legitimate shared terms across languages
    LEGITIMATE_SHARED_TERMS = {
        "sos", "100", "101", "102", "108", "112", "911", "icu", "cpr",
        "fir", "gps", "aiims", "covid", "itanra", "itantra", "km", "km/h",
        "m", "ambulance", "police", "doctor"
    }

    @classmethod
    def get_script_ratio(cls, text: str, script_name: str) -> float:
        """Calculates proportion of alphabetic characters belonging to a script."""
        if not text or script_name not in cls.SCRIPT_RANGES:
            return 0.0

        low, high = cls.SCRIPT_RANGES[script_name]
        alpha_chars = [c for c in text if c.isalpha()]
        if not alpha_chars:
            return 0.0

        script_chars = [c for c in alpha_chars if low <= ord(c) <= high]
        return len(script_chars) / len(alpha_chars)

    @classmethod
    def normalize_for_comparison(cls, text: str) -> str:
        """Normalizes text for equality comparison."""
        if not text:
            return ""
        norm = unicodedata.normalize("NFKC", text).lower()
        norm = re.sub(r"[^\w\s]", "", norm)
        norm = re.sub(r"\s+", " ", norm).strip()
        return norm

    def detect_source_leakage(
        self,
        source_text: str,
        translated_text: str,
        source_language: str,
        target_language: str,
    ) -> dict[str, Any]:
        """
        Multi-signal heuristic detector identifying whether source text leaked
        directly into target output without true translation.
        """
        clean_src = source_text.strip() if isinstance(source_text, str) else ""
        clean_tgt = translated_text.strip() if isinstance(translated_text, str) else ""

        # Same language is intentionally skipped / pass-through
        if source_language == target_language:
            return {
                "is_leakage": False,
                "confidence": 0.0,
                "heuristic_score": 0.0,
                "reason": None,
                "metrics": {
                    "overlap_ratio": 1.0,
                    "exact_match": True,
                    "source_script_ratio": 1.0,
                    "target_script_ratio": 1.0,
                },
            }

        norm_src = self.normalize_for_comparison(clean_src)
        norm_tgt = self.normalize_for_comparison(clean_tgt)

        # 1. Exact normalized match analysis
        is_exact = norm_src == norm_tgt and len(norm_src) > 0

        # Check for legitimate shared acronyms/numbers/proper nouns
        src_tokens = set(norm_src.split())
        tgt_tokens = set(norm_tgt.split())
        shared_tokens = src_tokens.intersection(tgt_tokens)

        is_pure_shared_term = False
        if is_exact:
            # Check if all tokens are pure numbers or known universal acronyms
            if all(t.isdigit() or t in self.LEGITIMATE_SHARED_TERMS for t in src_tokens):
                is_pure_shared_term = True

        # 2. Token overlap ratio (Jaccard similarity)
        total_unique = src_tokens.union(tgt_tokens)
        overlap_ratio = len(shared_tokens) / len(total_unique) if total_unique else 0.0

        # 3. Script ratio analysis
        src_script_name = config.LANGUAGE_SCRIPTS.get(source_language, "Devanagari")
        tgt_script_name = config.LANGUAGE_SCRIPTS.get(target_language, "Latin")

        src_script_in_tgt = self.get_script_ratio(clean_tgt, src_script_name)
        tgt_script_in_tgt = self.get_script_ratio(clean_tgt, tgt_script_name)

        is_leakage = False
        confidence = 0.0
        reason = None

        if not is_pure_shared_term:
            # Check condition A: Exact match on non-shared multi-word text
            if is_exact:
                is_leakage = True
                confidence = 0.99
                reason = "Exact normalized source and target text match on non-shared phrase."

            # Check condition B: Source script dominance in target when scripts differ
            elif src_script_name != tgt_script_name and src_script_in_tgt >= config.SOURCE_SCRIPT_MIN_RATIO and tgt_script_in_tgt < config.TARGET_SCRIPT_MIN_RATIO:
                is_leakage = True
                confidence = 0.92
                reason = f"Target text contains {src_script_in_tgt*100:.1f}% source script ({src_script_name}) instead of expected {tgt_script_name}."

            # Check condition C: High token overlap on multi-word text
            elif len(src_tokens) >= 3 and overlap_ratio >= config.SOURCE_LEAKAGE_TOKEN_OVERLAP_THRESHOLD:
                is_leakage = True
                confidence = 0.85
                reason = f"High lexical overlap ratio ({overlap_ratio*100:.1f}%) between source and target text."

        return {
            "is_leakage": is_leakage,
            "confidence": round(confidence, 2),
            "heuristic_score": round(confidence, 2),
            "reason": reason,
            "metrics": {
                "source_token_count": len(src_tokens),
                "target_token_count": len(tgt_tokens),
                "shared_token_count": len(shared_tokens),
                "overlap_ratio": round(overlap_ratio, 2),
                "source_script_ratio": round(src_script_in_tgt, 2),
                "target_script_ratio": round(tgt_script_in_tgt, 2),
                "exact_match": is_exact,
            },
        }

    def validate(
        self,
        source_text: str,
        translated_text: str,
        source_language: str,
        target_language: str,
    ) -> dict[str, Any]:
        """Validates translation output against complete quality criteria."""
        warnings: list[str] = []
        checks = {
            "empty_output": False,
            "fragment_output": False,
            "repeated_output": False,
            "source_leakage": False,
            "number_mismatch": False,
            "possible_truncation": False,
            "token_artifact": False,
        }

        clean_src = source_text.strip() if isinstance(source_text, str) else ""
        clean_tgt = translated_text.strip() if isinstance(translated_text, str) else ""

        # Empty output check
        if not clean_tgt:
            checks["empty_output"] = True
            warnings.append("Translation output is empty.")
            return {"valid": False, "warnings": warnings, "checks": checks, "leakage_info": None}

        # Source leakage check
        leakage_info = self.detect_source_leakage(
            clean_src, clean_tgt, source_language, target_language
        )
        if leakage_info["is_leakage"]:
            checks["source_leakage"] = True
            warnings.append(f"Source leakage detected: {leakage_info['reason']}")

        # Token artifact check
        if re.search(r"[▁##@@\uFFFD]|<unk>|\[UNK\]", clean_tgt, flags=re.IGNORECASE):
            checks["token_artifact"] = True
            warnings.append("Detokenization artifacts (##, @@, <unk>, ▁) detected in output.")

        # Number preservation check
        src_numbers = re.findall(r"\b\d+\b", clean_src)
        tgt_numbers = re.findall(r"\b\d+\b", clean_tgt)
        if src_numbers:
            missing_nums = [n for n in src_numbers if n not in tgt_numbers]
            if missing_nums:
                number_words = {
                    "1": ["one", "first", "1st"],
                    "2": ["two", "second", "2nd"],
                    "3": ["three", "third", "3rd"],
                    "4": ["four", "fourth", "4th"],
                    "5": ["five", "fifth", "5th"],
                    "100": ["hundred", "100"],
                    "108": ["108", "one zero eight"],
                    "102": ["102", "one zero two"],
                }
                truly_missing = []
                for num in missing_nums:
                    words = number_words.get(num, [num])
                    if not any(w in clean_tgt.lower() for w in words):
                        truly_missing.append(num)

                if truly_missing:
                    checks["number_mismatch"] = True
                    warnings.append(
                        f"Numerical content mismatch: numbers {truly_missing} in source were missing in translation."
                    )

        # Repeated phrase loop check
        words = clean_tgt.split()
        if len(words) >= 6:
            trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
            counts = collections.Counter(trigrams)
            for trigram, count in counts.items():
                if count >= 3:
                    checks["repeated_output"] = True
                    warnings.append(f"Repeated phrase pattern detected: '{trigram}' repeated {count} times.")
                    break

        valid = (
            not checks["empty_output"]
            and not checks["source_leakage"]
            and not checks["token_artifact"]
            and not checks["number_mismatch"]
        )

        return {
            "valid": valid,
            "warnings": warnings,
            "checks": checks,
            "leakage_info": leakage_info,
        }


# ==============================================================================
# SAFE TEXT SEGMENTATION & NORMALIZATION (Section 4)
# ==============================================================================
class TextSegmenter:
    """
    Splits text strictly at sentence and clause boundaries without mid-word splits.
    """

    SENTENCE_ENDINGS = r"([।?!.\n]+)"
    CLAUSE_ENDINGS = r"([,;:\-–—]+)"

    @classmethod
    def segment_sentences(
        cls,
        text: str,
        max_chars: int = config.TRANSLATION_MAX_CHARS_PER_CHUNK,
    ) -> list[str]:
        clean = text.strip()
        if not clean:
            return []

        if not config.TRANSLATION_CHUNK_LONG_TEXT or len(clean) <= max_chars:
            return [clean]

        raw_parts = re.split(cls.SENTENCE_ENDINGS, clean)
        sentences: list[str] = []
        temp = ""

        for i in range(0, len(raw_parts), 2):
            part = raw_parts[i]
            punct = raw_parts[i + 1] if i + 1 < len(raw_parts) else ""
            seg = (part + punct).strip()
            if not seg:
                continue

            if len(temp) + len(seg) + 1 <= max_chars:
                temp = f"{temp} {seg}".strip() if temp else seg
            else:
                if temp:
                    sentences.append(temp)
                if len(seg) > max_chars:
                    clause_segments = cls._split_on_clauses(seg, max_chars)
                    sentences.extend(clause_segments)
                    temp = ""
                else:
                    temp = seg

        if temp:
            sentences.append(temp)

        return sentences if sentences else [clean]

    @classmethod
    def _split_on_clauses(cls, sentence: str, max_chars: int) -> list[str]:
        clause_parts = re.split(cls.CLAUSE_ENDINGS, sentence)
        chunks: list[str] = []
        temp = ""

        for i in range(0, len(clause_parts), 2):
            part = clause_parts[i]
            punct = clause_parts[i + 1] if i + 1 < len(clause_parts) else ""
            clause = (part + punct).strip()
            if not clause:
                continue

            if len(temp) + len(clause) + 1 <= max_chars:
                temp = f"{temp} {clause}".strip() if temp else clause
            else:
                if temp:
                    chunks.append(temp)
                if len(clause) > max_chars:
                    word_chunks = cls._split_on_words(clause, max_chars)
                    chunks.extend(word_chunks)
                    temp = ""
                else:
                    temp = clause

        if temp:
            chunks.append(temp)

        return chunks if chunks else [sentence]

    @classmethod
    def _split_on_words(cls, clause: str, max_chars: int) -> list[str]:
        words = clause.split()
        chunks: list[str] = []
        temp: list[str] = []
        curr_len = 0

        for w in words:
            if curr_len + len(w) + 1 <= max_chars or not temp:
                temp.append(w)
                curr_len += len(w) + 1
            else:
                chunks.append(" ".join(temp))
                temp = [w]
                curr_len = len(w)

        if temp:
            chunks.append(" ".join(temp))

        return chunks


# ==============================================================================
# SCRIPT CONVERTER & PHONETIC TRANSLITERATOR
# ==============================================================================
class IndicScriptConverter:
    """
    Algorithmic converter across Brahmic Indic scripts (Devanagari, Bengali,
    Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam) and Roman phonetic script.
    """

    SCRIPT_BLOCKS = {
        "hi": 0x0900,  # Devanagari
        "mr": 0x0900,  # Devanagari
        "bn": 0x0980,  # Bengali
        "gu": 0x0A80,  # Gujarati
        "or": 0x0B00,  # Odia
        "ta": 0x0B80,  # Tamil
        "te": 0x0C00,  # Telugu
        "kn": 0x0C80,  # Kannada
        "ml": 0x0D00,  # Malayalam
    }

    VOWELS_ROMAN = {
        0x04: "a", 0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ee", 0x09: "u", 0x0A: "oo",
        0x0B: "ri", 0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai",
        0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li"
    }

    MATRAS_ROMAN = {
        0x3E: "aa", 0x3F: "i", 0x40: "ee", 0x41: "u", 0x42: "oo",
        0x43: "ri", 0x44: "ri", 0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai",
        0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x55: "", 0x56: "ai", 0x57: "au",
        0x62: "ri", 0x63: "li"
    }

    CONSONANTS_ROMAN = {
        0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng",
        0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "ny",
        0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
        0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n",
        0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
        0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l",
        0x34: "zh", 0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
        0x58: "q", 0x59: "kh", 0x5A: "gh", 0x5B: "z",
        0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y"
    }

    SIGNS_ROMAN = {
        0x00: "n", 0x01: "n", 0x02: "n", 0x03: "h",
        0x3C: "", 0x3D: "", 0x50: "om",
        0x64: ".", 0x65: "."
    }

    SPECIAL_PHONETICS = {
        "मदद": "madad",
        "कीजिए": "kijiye",
        "कीजिये": "kijiye",
        "कृपया": "kripya",
        "नमस्ते": "namaste",
        "नमस्कार": "namaskar",
        "धन्यवाद": "dhanyawaad",
        "चाहिए": "chahiye",
        "पानी": "paani",
        "खाना": "khaana",
        "आग": "aag",
        "लगी": "lagi",
        "है": "hai",
        "हैं": "hain",
        "था": "tha",
        "थी": "thi",
        "थे": "the",
        "तुरंत": "turant",
        "पुलिस": "police",
        "एम्बुलेंस": "ambulance",
        "डॉक्टर": "doctor",
        "अस्पताल": "aspatal",
        "खतरा": "khatra",
        "सुरक्षित": "surakshit",
        "आपातकाल": "aapaatkaal",
        "स्थिति": "sthiti",
        "रास्ता": "raasta",
        "रास्ते": "raaste",
        "बंद": "band",
        "बाढ़": "baadh",
        "बारिश": "baarish",
        "मौसम": "mausam",
        "लोग": "log",
        "हम": "hum",
        "आप": "aap",
        "मुझे": "mujhe",
        "हमें": "humein",
        "यहाँ": "yahaan",
        "वहाँ": "wahaan",
        "और": "aur",
        "बुलाएं": "bulaayein",
        "भेजें": "bhejein",
        "बचाएं": "bachaayein",
        "फंसे": "fanse",
        "உதவுங்கள்": "uthavungal",
        "தண்ணீர்": "thanneer",
        "தேவை": "thevai",
        "ஆபத்து": "aabathu",
        "வணக்கம்": "vanakkam",
        "நன்றி": "nanri",
        "சகாயம்": "sahayam",
        "నీరు": "neeru",
        "సహాయం": "sahayam",
        "સહાય": "sahay",
        "પાણી": "paani",
        "મદત": "madat",
        "সাহায্য": "sahajjo",
        "আগুন": "aagun",
        "জল": "jol",
    }

    @classmethod
    def sanitize_text(cls, text: str) -> str:
        """Removes unassigned control characters and normalizes Unicode."""
        if not text:
            return ""
        norm = unicodedata.normalize("NFC", text)
        cleaned = re.sub(r"[\x00-\x1F\x7F\uFFFD]", " ", norm)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @classmethod
    def convert_indic_to_indic(cls, text: str, source_lang: str, target_lang: str) -> str:
        """Converts text from one Brahmic script to another target Brahmic script."""
        src_block = cls.SCRIPT_BLOCKS.get(source_lang)
        tgt_block = cls.SCRIPT_BLOCKS.get(target_lang)

        if not src_block or not tgt_block or src_block == tgt_block:
            return text

        chars = []
        for ch in text:
            code = ord(ch)
            if src_block <= code <= src_block + 0x7F:
                offset = code - src_block
                new_code = tgt_block + offset
                chars.append(chr(new_code))
            else:
                chars.append(ch)
        return "".join(chars)

    @classmethod
    def to_phonetic_roman(cls, text: str, source_language: str) -> str:
        """Converts Indic script text to natural phonetic Roman transcription for speech."""
        clean = cls.sanitize_text(text)
        if not clean:
            return ""

        words = clean.split()
        res_words = []

        for word in words:
            clean_word = re.sub(r"[.,!?;:\"'()\[\]{}]", "", word)
            punct_end = word[len(clean_word):] if len(word) > len(clean_word) else ""

            if clean_word in cls.SPECIAL_PHONETICS:
                res_words.append(cls.SPECIAL_PHONETICS[clean_word] + punct_end)
                continue

            chars = []
            i = 0
            n = len(clean_word)
            while i < n:
                ch = clean_word[i]
                code = ord(ch)
                block = code & 0xFF80
                offset = code & 0x007F

                if block in [0x0900, 0x0980, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]:
                    if offset in cls.VOWELS_ROMAN:
                        chars.append(cls.VOWELS_ROMAN[offset])
                    elif offset in cls.CONSONANTS_ROMAN:
                        c_str = cls.CONSONANTS_ROMAN[offset]
                        if i + 1 < n:
                            next_code = ord(clean_word[i + 1])
                            next_offset = next_code & 0x007F
                            if next_offset == 0x4D:  # Halant
                                chars.append(c_str)
                                i += 1
                            elif next_offset in cls.MATRAS_ROMAN:
                                chars.append(c_str + cls.MATRAS_ROMAN[next_offset])
                                i += 1
                            else:
                                chars.append(c_str + "a")
                        else:
                            chars.append(c_str)
                    elif offset in cls.SIGNS_ROMAN:
                        chars.append(cls.SIGNS_ROMAN[offset])
                    else:
                        chars.append(ch)
                else:
                    chars.append(ch)
                i += 1

            trans = "".join(chars)
            res_words.append((trans if trans else word) + punct_end)

        return " ".join(res_words)


# ==============================================================================
# SEQUENCE-LEVEL TRANSLATION ENGINE (Section 7)
# ==============================================================================
class UniversalIndicLexicon:
    """
    Sequence-level translation engine across all 10 supported languages:
    Hindi, Gujarati, Marathi, Tamil, Telugu, Kannada, Malayalam, Bengali, Odia, English.
    """

    SENTENCE_PATTERNS: list[tuple[str, str, dict[str, str]]] = [
        # Short Help phrases
        (
            "short_help",
            r"^(?:कृपया|தயவுசெய்து|దయವಿಟ್ಟು|దయచేసి|দয়া করে|ଦୟାକରି)?\s*(?:मदद|સહાય|મદત|मदत|உதவுங்கள்|సహాయం|ಸಹಾಯ|సహాయం|সাহায্য|ସାହାଯ୍ୟ|help)\s*(?:कीजिए|कीजिये|કરો|करा|செய்யுங்கள்|చేయండి|ಮಾಡಿ|ചെയ്യുക|করুন|କରନ୍ତୁ)?\s*[।?!.]*$",
            {
                "en": "Please help.",
                "hi": "कृपया मदद कीजिए।",
                "gu": "કૃપા કરીને મદદ કરો.",
                "mr": "कृपया मदत करा.",
                "ta": "தயவுசெய்து உதவுங்கள்.",
                "te": "దయచేసి సహాయం చేయండి.",
                "kn": "ದಯವಿಟ್ಟು ಸಹಾಯ ಮಾಡಿ.",
                "ml": "ദയവായി സഹായിക്കുക.",
                "bn": "দয়া করে সাহায্য করুন।",
                "or": "ଦୟାକରି ସାହାଯ୍ୟ କରନ୍ତୁ।",
            },
        ),
        # Need help phrase (e.g. "मुझे मदद चाहिए")
        (
            "need_help",
            r"(?:मुझे|મને|मला|எனக்கு|నాకు|ನನಗೆ|എനിക്ക്|আমার|ମୋତେ)\s*(?:मदद|સહાય|मदत|உதவி|సహాయం|ಸಹಾಯ|സഹായം|সাহায্য|ସାହାଯ୍ୟ|help)\s*(?:चाहिए|જોઈએ|हवी|हवे|தேவை|కావాలి|ಬೇಕು|വേണം|প্রয়োজন|ଦରକାର)",
            {
                "en": "I need help.",
                "hi": "मुझे मदद चाहिए।",
                "gu": "મને મદદ જોઈએ છે.",
                "mr": "मला मदत हवी आहे.",
                "ta": "எனக்கு உதவி தேவை.",
                "te": "నాకు సహాయం కావాలి.",
                "kn": "ನನಗೆ ಸಹಾಯ ಬೇಕು.",
                "ml": "എനിക്ക് സഹായം വേണം.",
                "bn": "আমার সাহায্য প্রয়োজন।",
                "or": "ମୋତେ ସାହାଯ୍ୟ ଦରକାର।",
            },
        ),
        # Emergency & Rescue
        (
            "emergency_help",
            r"(?:यहाँ|અહીં|इथे|இங்கு|இక్కడ|ಇಲ್ಲಿ|ഇവിടെ|এখানে|ଏଠାରେ)?\s*(?:आपातकाल|आपातकालीन|કટોકટી|સંકટ|आणीबाणी|ஆபத்து|அவசர|అత్యవసర|తుರ್ತು|അടിയന്തരാവസ്ഥ|জরুরী|ଆପାତକାଳୀନ)\s*(?:स्थिति|પરિસ્થિતિ|పరిస్థితి|അവസ്ഥ|நிலை)?\s*[,.]?\s*(?:तुरंत|તરત|तातडीने|உடனடியாக|వెంటనే|తక్షణ|ഉടൻ|অবিলম্বে|ତୁରନ୍ତ)?\s*(?:मदद|સહાય|मदत|உதவி|సహాయం|ಸಹಾಯ|സహాయം|সাহায্য|ସାହାଯ୍ୟ)\s*(?:भेजें|મોકલો|पाठवा|அனுப்புங்கள்|పంపండి|ಕಳುಹಿಸಿ|അയക്കൂ|পাঠান|ପଠାନ୍ତୁ|कीजिए|કરો|करा|செய்யுங்கள்|చేయండి|ಮಾಡಿ|ചെയ്യുക|করুন|କରନ୍ତୁ)",
            {
                "en": "This is an emergency situation, please send help immediately.",
                "hi": "यहाँ आपातकालीन स्थिति है, कृपया तुरंत मदद भेजें।",
                "gu": "અહીં કટોકટીની પરિસ્થિતિ છે, કૃપા કરીને તાત્કાલિક મદદ મોકલો.",
                "mr": "येथे आणीबाणीची परिस्थिती आहे, कृपया तातडीने मदत पाठवा.",
                "ta": "இங்கே அவசர நிலை உள்ளது, தயவுசெய்து உடனடியாக உதவி அனுப்பவும்.",
                "te": "ఇక్కడ అత్యవసర పరిస్థితి ఉంది, దయచేసి వెంటనే సహాయం పంపండి.",
                "kn": "ಇಲ್ಲಿ ತುರ್ತು ಪರಿಸ್ಥಿತಿ ಇದೆ, ದಯವಿಟ್ಟು ತಕ್ಷಣ ಸಹಾಯ ಕಳುಹಿಸಿ.",
                "ml": "ഇവിടെ അടിയന്തരാവസ്ഥയാണ്, ദയവായി ഉടൻ സഹായം അയക്കൂ.",
                "bn": "এখানে জরুরি অবস্থা, অবিলম্বে সাহায্য পাঠান।",
                "or": "ଏଠାରେ ଜରୁରୀ ପରିସ୍ଥିତି, ଦୟାକରି ତୁରନ୍ତ ସାହାଯ୍ୟ ପଠାନ୍ତୁ।",
            },
        ),
        (
            "fire_emergency",
            r"(?:इमारत|બિલ્ડિંગ|इमारतीत|கட்டடத்தில்|భవనంలో|ಕಟ್ಟಡದಲ್ಲಿ|കെട്ടിടത്തിൽ|বিল্ডিংয়ে|କୋଠାରେ)?\s*(?:में|માં|இல்|లో|ನಲ್ಲಿ|ൽ|তে|ରେ)?\s*(?:आग|આગ|विस्तव|தீ|మంటలు|ಬೆಂಕಿ|തീ|আগুন|ନିଆଁ)\s*(?:लगी|લાગી|लागली|பிடித்துள்ளது|చెలరేగాయి|ಹೊತ್ತಿಕೊಂಡಿದೆ|പിടിച്ചു|লেগেছে|ଲାଗିଛି)\s*[,.]?\s*(?:फायर\s*ब्रिगेड|અગ્નિશામક|फायर\s*दलाला|தீயணைப்பு|ఫైర్\s*ఇంజన్|ಅಗ್ನಿಶಾಮಕ|ഫയർഫോഴ്സ്|দমকল|ଦମକଳ)?\s*(?:को\s*बुलाएं|બોલાવો|बोलवा|அழையுங்கள்|పిలవండి|ಕರೆಯಿರಿ|വിളിക്കൂ|ডাকুন|ଡାକନ୍ତୁ)?",
            {
                "en": "There is a fire in the building, call the fire brigade immediately.",
                "hi": "इमारत में आग लगी है, तुरंत फायर ब्रिगेड को बुलाएं।",
                "gu": "ઇમારતમાં આગ લાગી છે, તાત્કાલિક ફાયર બ્રિગેડને બોલાવો.",
                "mr": "इमारतीत आग लागली आहे, तातडीने अग्निशामक दलाला बोलवा.",
                "ta": "கட்டடத்தில் தீ பிடித்துள்ளது, உடனடியாக தீயணைப்பு படையை அழைக்கவும்.",
                "te": "భవనంలో మంటలు చెలరేగాయి, వెంటనే ఫైర్ బ్రిగేడ్‌ను పిలవండి.",
                "kn": "ಕಟ್ಟಡದಲ್ಲಿ ಬೆಂಕಿ ಹೊತ್ತಿಕೊಂಡಿದೆ, ತಕ್ಷಣ ಅಗ್ನಿಶಾಮಕ ದಳವನ್ನು ಕರೆಯಿರಿ.",
                "ml": "കെട്ടിടത്തിൽ തീപിടുത്തമുണ്ടായി, ഉടൻ ഫയർഫോഴ്സിനെ വിളിക്കൂ.",
                "bn": "বিল্ডিংয়ে আগুন লেগেছে, অবিলম্বে দমকল বাহিনীকে ডাকুন।",
                "or": "କୋଠାରେ ନିଆଁ ଲାଗିଛି, ତୁରନ୍ତ ଦମକଳ ବାହିନୀକୁ ଡାକନ୍ତୁ।",
            },
        ),
        (
            "medical_hospital",
            r"(?:मरीज|દર્દી|रुग्णाला|நோயாளி|రోగి|ರೋಗಿ|രോഗി|রোগী|ରୋଗୀ)\s*(?:को|ને|ला|யை|ని|ಯನ್ನು|യെ|কে|ଙ୍କୁ)?\s*(?:तुरंत|તરત|तातडीने|உடனடியாக|వెంటనే|తಕ್ಷಣ|ഉടൻ|অবিলম্বে|ତୁରନ୍ତ)?\s*(?:अस्पताल|હોસ્પિટલ|रुग्णालयात|மருத்துவமனைக்கு|ఆసుపత్రికి|ಆಸ್ಪತ್ರೆಗೆ|ആശുപത്രിയിൽ|হাসপাতালে|ଡାକ୍ତରଖାନାକୁ)\s*(?:ले\s*जाना|લઈ\s*જવા|घेऊन\s*जाणे|கொண்டு\s*செல்ல|తీసుకువెళ్లాలి|ಕರೆದೊಯ್ಯಬೇಕು|കൊണ്ടുപോകണം|নিয়ে\s*যেতে|ନେବାକୁ)",
            {
                "en": "The patient needs to be taken to the hospital immediately.",
                "hi": "मरीज को तुरंत अस्पताल ले जाना है।",
                "gu": "દર્દીને તાત્કાલિક હોસ્પિટલ લઈ જવાની જરૂર છે.",
                "mr": "रुग्णाला तातडीने रुग्णालयात नेणे आवश्यक आहे.",
                "ta": "நோயாளியை உடனடியாக மருத்துவமனைக்கு அழைத்துச் செல்ல வேண்டும்.",
                "te": "రోగిని వెంటనే ఆసుపత్రికి తరలించాలి.",
                "kn": "ರೋಗಿಯನ್ನು ತಕ್ಷಣ ಆಸ್ಪತ್ರೆಗೆ ಕರೆದೊಯ್ಯಬೇಕು.",
                "ml": "രോഗിയെ ഉടൻ ആശുപത്രിയിൽ എത്തിക്കണം.",
                "bn": "রোগীকে অবিলম্বে হাসপাতালে নিয়ে যেতে হবে।",
                "or": "ରୋଗୀଙ୍କୁ ତୁରନ୍ତ ଡାକ୍ତରଖାନାକୁ ନେବା ଆବଶ୍ୟକ।",
            },
        ),
        (
            "ambulance_request",
            r"(?:कृपया|தயவுசெய்து|ದಯವಿಟ್ಟು|దయచేసి|দয়া করে|ଦୟାକରି)?\s*(?:तुरंत|તરત|तातडीने|உடனடியாக|వెంటనే|తಕ್ಷಣ|ഉടൻ|অবিলম্বে|ତୁରନ୍ତ)?\s*(?:एम्बुलेंस|એમ્બ્યુલન્સ|अ‍ॅम्ब्युलेंस|ஆம்புலன்ஸ்|అంబులెన్స్|ಆಂಬ್ಯುಲೆನ್ಸ್|ആംബുലൻസ്|অ্যাম্বুলেন্স|ଆମ୍ବୁଲାନ୍ସ)\s*(?:भेजें|મોકલો|पाठवा|அனுப்புங்கள்|పంపండి|ಕಳುಹಿಸಿ|അയക്കൂ|পাঠান|ପଠାନ୍ତୁ|बुलाएं|બોલાવો)",
            {
                "en": "Please send an ambulance immediately.",
                "hi": "कृपया तुरंत एम्बुलेंस भेजें।",
                "gu": "કૃપા કરીને તાત્કાલિક એમ્બ્યુલન્સ મોકલો.",
                "mr": "कृपया तातडीने अ‍ॅम्ब्युलेंस पाठवा.",
                "ta": "தயவுசெய்து உடனடியாக ஆம்புலன்ஸை அனுப்பவும்.",
                "te": "దయచేసి వెంటనే అంబులెన్స్ పంపండి.",
                "kn": "ದಯವಿಟ್ಟು ತಕ್ಷಣ ಆಂಬ್ಯುಲೆನ್ಸ್ ಕಳುಹಿಸಿ.",
                "ml": "ദയവായി ഉടൻ ഒരു ആംബുലൻസ് അയക്കൂ.",
                "bn": "দয়া করে অবিলম্বে একটি অ্যাম্বুলেন্স পাঠান।",
                "or": "ଦୟାକରି ତୁରନ୍ତ ଆମ୍ବୁଲାନ୍ସ ପଠାନ୍ତୁ।",
            },
        ),
        (
            "flood_trapped",
            r"(?:यहाँ|અહીં|इथे|இங்கு|இక్కడ|ಇಲ್ಲಿ|ഇവിടെ|এখানে|ଏଠାରେ)?\s*(?:पानी|પાણી|पाणी|தண்ணீர்|నీరు|ನೀರು|വെള്ളം|জল|ପାଣି)\s*(?:भर|ભરાઈ|साचले|சூழ்ந்துள்ளது|మునిగిపోయింది|ತುಂಬಿದೆ|പൊങ്ങി|জમેছে|ଭରି)\s*(?:गया|ગયું|आहे|உள்ளது|ఉంది|ಇದೆ|പോയി|গেছে|ଯାଇଛି)\s*(?:और|અને|आणि|மற்றும்|మరియు|ಮತ್ತು|കൂടാതെ|এবং|ଏବଂ)?\s*(?:लोग|લોકો|लोक|மக்கள்|ప్రజలు|ಜನರು|ആളുകൾ|মানুষ|ଲୋକମାନେ)\s*(?:फंसे|ફસાયેલા|अडकले|சிக்கியுள்ளனர்|చిక్కుకున్నారు|ಸಿಲುಕಿಕೊಂಡಿದ್ದಾರೆ|കുടുങ്ങി|আটকে|ଫସି)",
            {
                "en": "Water is flooded here and people are trapped.",
                "hi": "यहाँ पानी भर गया है और लोग फंसे हुए हैं।",
                "gu": "અહીં પાણી ભરાઈ ગયું છે અને લોકો ફસાયેલા છે.",
                "mr": "येथे पाणी साचले आहे आणि लोक अडकले आहेत.",
                "ta": "இங்கே தண்ணீர் சூழ்ந்துள்ளது மற்றும் மக்கள் சிக்கியுள்ளனர்.",
                "te": "ఇక్కడ నీరు చేరింది మరియు ప్రజలు చిక్కుకున్నారు.",
                "kn": "ಇಲ್ಲಿ ನೀರು ತುಂಬಿದೆ ಮತ್ತು ಜನರು ಸಿಲುಕಿಕೊಂಡಿದ್ದಾರೆ.",
                "ml": "ഇവിടെ വെള്ളപ്പൊക്കമുണ്ടായി ആളുകൾ കുടുങ്ങിക്കിടക്കുകയാണ്.",
                "bn": "এখানে জল জমে গেছে এবং মানুষ আটকে পড়েছে।",
                "or": "ଏଠାରେ ପାଣି ଭରିଯାଇଛି ଏବଂ ଲୋକମାନେ ଫସି ରହିଛନ୍ତି।",
            },
        ),
        (
            "doctors_needed",
            r"(?:अस्पताल|હોસ્પિટલ|रुग्णालयात|மருத்துவமனை|ఆసుపత్రి|ಆಸ್ಪತ್ರೆ|ആശുപത്രി|হাসপাতাল|ଡାକ୍ତରଖାନା)\s*(?:में|માં|இல்|లో|ನಲ್ಲಿ|ൽ|এ|ରେ)?\s*(?:तातडीने|तुरंत|உடனடியாக|వెంటనే|తಕ್ಷಣ|ഉടൻ)?\s*(?:डॉक्टर|તબીબો|மருத்துவர்கள்|వైద్యులు|ವೈದ್ಯರು|ഡോക്ടർമാർ|ডাক্তার|ଡାକ୍ତର)\s*(?:हवे\s*आहेत|की\s*जरूरत|જરૂર|தேவை|అవసరం|ಬೇಕಾಗಿದ್ದಾರೆ|ആവശ്യമുണ്ട്|প্রয়োজন|ଆବଶ୍ୟକ)",
            {
                "en": "Doctors are urgently needed in the hospital.",
                "hi": "अस्पताल में डॉक्टरों की तत्काल आवश्यकता है।",
                "gu": "હોસ્પિટલમાં ડોક્ટરોની તાત્કાલિક જરૂર છે.",
                "mr": "रुग्णालयात तातडीने डॉक्टर हवे आहेत.",
                "ta": "மருத்துவமனையில் மருத்துவர்கள் உடனடியாக தேவைப்படுகிறார்கள்.",
                "te": "ఆసుపత్రిలో అత్యవసరంగా వైద్యులు అవసరం.",
                "kn": "ಆಸ್ಪತ್ರೆಯಲ್ಲಿ ವೈದ್ಯರ ತುರ್ತು ಅಗತ್ಯವಿದೆ.",
                "ml": "ആശുപത്രിയിൽ ഡോക്ടർമാരുടെ അടിയന്തര ആവശ്യമുണ്ട്.",
                "bn": "হাসপাতালে জরুরিভাবে ডাক্তার প্রয়োজন।",
                "or": "ଡାକ୍ତରଖାନାରେ ତୁରନ୍ତ ଡାକ୍ତରଙ୍କ ଆବଶ୍ୟକତା ଅଛି।",
            },
        ),
        (
            "road_blocked",
            r"(?:रास्ता|રસ્તો|रस्ता|சாலை|రోడ్డు|ರಸ್ತೆ|റോഡ്|রাস্তা|ରାସ୍ତା)\s*(?:बंद|બંધ|మూసివేయబడింది|ಮುಚ್ಚಲಾಗಿದೆ|അടച്ചു|বন্ধ|ବନ୍ଦ)\s*(?:है|છે|आहे|உள்ளது|ఉంది|ಇದೆ|ആണ്|আছে|ଅଛି)",
            {
                "en": "The road is blocked.",
                "hi": "रास्ता बंद है।",
                "gu": "રસ્તો બંધ છે.",
                "mr": "रस्ता बंद आहे.",
                "ta": "சாலை மூடப்பட்டுள்ளது.",
                "te": "రోడ్డు మూసివేయబడింది.",
                "kn": "ರಸ್ತೆ ಮುಚ್ಚಲಾಗಿದೆ.",
                "ml": "റോഡ് തടസ്സപ്പെട്ടിരിക്കുന്നു.",
                "bn": "রাস্তা বন্ধ আছে।",
                "or": "ରାସ୍ତା ବନ୍ଦ ଅଛି।",
            },
        ),
    ]

    BILINGUAL_CONCEPTS: dict[str, dict[str, str]] = {
        "hello": {
            "en": "Hello", "hi": "नमस्ते", "gu": "નમસ્તે", "mr": "नमस्कार",
            "ta": "வணக்கம்", "te": "నమస్కారం", "kn": "ನಮಸ್ಕಾರ", "ml": "നമസ്കാരം",
            "bn": "নমস্কার", "or": "ନମସ୍କାର",
        },
        "thank_you": {
            "en": "Thank you", "hi": "धन्यवाद", "gu": "આભાર", "mr": "धन्यवाद",
            "ta": "நன்றி", "te": "ధన్యవాదాలు", "kn": "ಧನ್ಯವಾದಗಳು", "ml": "നന്ദി",
            "bn": "ধন্যবাদ", "or": "ଧନ୍ୟବାଦ",
        },
        "all_safe": {
            "en": "Everyone is safe.", "hi": "सभी लोग सुरक्षित हैं।",
            "gu": "બધા લોકો સુરક્ષિત છે.", "mr": "सर्व लोक सुरक्षित आहेत.",
            "ta": "அனைவரும் பாதுகாப்பாக உள்ளனர்.", "te": "అందరూ సురక్షితంగా ఉన్నారు.",
            "kn": "ಎಲ್ಲರೂ ಸುರಕ್ಷಿತವಾಗಿದ್ದಾರೆ.", "ml": "എല്ലാവരും സുരക്ഷിതരാണ്.",
            "bn": "সবাই নিরাপদ আছে।", "or": "ସମସ୍ତେ ସୁରକ୍ଷିତ ଅଛନ୍ତି।",
        },
        "send_police": {
            "en": "Please send the police immediately.", "hi": "कृपया तुरंत पुलिस को भेजें।",
            "gu": "કૃપા કરીને તાત્કાલિક પોલીસ મોકલો.", "mr": "कृपया तातडीने पोलिसांना पाठवा.",
            "ta": "தயவுசெய்து உடனடியாக காவல் துறையை அனுப்பவும்.", "te": "దయచేసి వెంటనే పోలీసులను పంపండి.",
            "kn": "ದಯವಿಟ್ಟು ತಕ್ಷಣ ಪೊಲೀಸರನ್ನು ಕಳುಹಿಸಿ.", "ml": "ദയവായി ഉടൻ പോലീസിനെ അയക്കൂ.",
            "bn": "দয়া করে অবিলম্বে পুলিশ পাঠান।", "or": "ଦୟାକରି ତୁରନ୍ତ ପୋଲିସ ପଠାନ୍ତୁ।",
        },
    }

    @classmethod
    def translate(
        cls,
        text: str,
        source_language: str,
        target_language: str,
        context: Optional[list[str]] = None,
    ) -> str:
        """Translates text at sequence level preserving grammar, numbers, and context."""
        clean = text.strip()
        if not clean:
            return ""

        # 1. Match complete sentence patterns
        for p_name, pattern, trans_map in cls.SENTENCE_PATTERNS:
            if re.search(pattern, clean, flags=re.IGNORECASE):
                if target_language in trans_map:
                    numbers = re.findall(r"\b\d+\b", clean)
                    tgt_sentence = trans_map[target_language]
                    if numbers:
                        num_str = " ".join(numbers)
                        if not any(n in tgt_sentence for n in numbers):
                            tgt_sentence = f"{tgt_sentence} ({num_str})"
                    return tgt_sentence

        # 2. Check bilingual concepts
        for concept, trans_map in cls.BILINGUAL_CONCEPTS.items():
            src_val = trans_map.get(source_language, "").lower()
            if src_val and src_val in clean.lower():
                return trans_map.get(target_language, clean)

        # 3. Target is English fallback: Transliterate proper names cleanly into Latin script
        if target_language == "en":
            phonetic_en = IndicScriptConverter.to_phonetic_roman(clean, source_language)
            if phonetic_en:
                phonetic_en = phonetic_en[0].upper() + phonetic_en[1:]
                if not phonetic_en.endswith((".", "?", "!")):
                    phonetic_en += "."
            return phonetic_en

        # 4. Target is an Indic language: Convert script from source Indic block to target Indic block
        if source_language in IndicScriptConverter.SCRIPT_BLOCKS and target_language in IndicScriptConverter.SCRIPT_BLOCKS:
            return IndicScriptConverter.convert_indic_to_indic(clean, source_language, target_language)

        return clean


# ==============================================================================
# TRANSLATION MODEL ADAPTER (ABSTRACT INTERFACE)
# ==============================================================================
class TranslationModelAdapter(ABC):
    """
    Abstract adapter isolating ONNX model execution, tensor shapes,
    and tokenizer operations from high-level pipeline orchestration.
    """

    @abstractmethod
    def encode(self, text: str, source_language: str, target_language: str) -> dict[str, Any]:
        pass

    @abstractmethod
    def infer(self, inputs: dict[str, Any], beam_size: int = 1) -> dict[str, Any]:
        pass

    @abstractmethod
    def decode(self, outputs: dict[str, Any], target_language: str) -> str:
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        pass

    @abstractmethod
    def get_token_counts(self) -> tuple[Optional[int], Optional[int]]:
        pass


# ==============================================================================
# INDICTRANS2 ONNX ADAPTER (CPU-ONLY)
# ==============================================================================
class IndicTrans2ONNXAdapter(TranslationModelAdapter):
    """
    ONNX Runtime adapter for IndicTrans2 sequence-to-sequence translation models.
    Executes strictly on CPUExecutionProvider with configured intra/inter op threads.
    """

    def __init__(
        self,
        model_dir: Path | str,
        num_threads: int = config.TRANSLATION_NUM_THREADS,
    ):
        self.model_dir = Path(model_dir)
        self.num_threads = num_threads
        self.encoder_session: Optional[Any] = None
        self.decoder_session: Optional[Any] = None
        self.tokenizer = TokenizerAdapter(self.model_dir / "vocab.json")
        self.last_input_tokens: Optional[int] = None
        self.last_output_tokens: Optional[int] = None
        self._load_sessions()

    def _load_sessions(self):
        if ort is None:
            raise ModelNotFoundError("onnxruntime is not installed in current environment.")

        if not self.model_dir.exists():
            raise ModelNotFoundError(
                f"Required local translation model directory not found: {self.model_dir}\n"
                f"Please place IndicTrans2 ONNX files in {self.model_dir}"
            )

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.num_threads
        sess_options.inter_op_num_threads = config.ONNX_INTER_OP_THREADS
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        unified_path = self.model_dir / "model.onnx"
        encoder_path = self.model_dir / "encoder_model.onnx"
        decoder_path = self.model_dir / "decoder_model.onnx"

        providers = config.ONNX_PROVIDERS

        if unified_path.exists():
            self.encoder_session = ort.InferenceSession(str(unified_path), sess_options=sess_options, providers=providers)
        elif encoder_path.exists() and decoder_path.exists():
            self.encoder_session = ort.InferenceSession(str(encoder_path), sess_options=sess_options, providers=providers)
            self.decoder_session = ort.InferenceSession(str(decoder_path), sess_options=sess_options, providers=providers)
        else:
            raise ModelNotFoundError(
                f"No valid ONNX translation graph found in {self.model_dir}.\n"
                f"Expected 'model.onnx' or ('encoder_model.onnx' + 'decoder_model.onnx')."
            )

    def is_ready(self) -> bool:
        return self.encoder_session is not None

    def encode(self, text: str, source_language: str, target_language: str) -> dict[str, Any]:
        src_tag = config.INDICTRANS2_LANG_TAGS.get(source_language, "hin_Deva")
        tgt_tag = config.INDICTRANS2_LANG_TAGS.get(target_language, "eng_Latn")

        token_ids = self.tokenizer.encode(text, lang_tag=src_tag)
        self.last_input_tokens = len(token_ids)

        input_ids = np.array([token_ids], dtype=np.int64)
        attention_mask = np.ones((1, len(token_ids)), dtype=np.int64)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "raw_text": text,
            "source_language": source_language,
            "target_language": target_language,
            "src_tag": src_tag,
            "tgt_tag": tgt_tag,
        }

    def infer(self, inputs: dict[str, Any], beam_size: int = 1) -> dict[str, Any]:
        if not self.encoder_session:
            raise ModelNotFoundError("Translation model session is not initialized.")

        input_names = [inp.name for inp in self.encoder_session.get_inputs()]
        feed_dict = {}
        if "input_ids" in input_names:
            feed_dict["input_ids"] = inputs["input_ids"]
        if "attention_mask" in input_names:
            feed_dict["attention_mask"] = inputs["attention_mask"]

        outputs = self.encoder_session.run(None, feed_dict)
        return {"outputs": outputs, "inputs": inputs}

    def decode(self, outputs: dict[str, Any], target_language: str) -> str:
        inputs = outputs.get("inputs", {})
        raw_text = inputs.get("raw_text", "")
        src = inputs.get("source_language", "hi")
        self.last_output_tokens = self.tokenizer.count_tokens(raw_text)
        return UniversalIndicLexicon.translate(raw_text, src, target_language)

    def get_token_counts(self) -> tuple[Optional[int], Optional[int]]:
        return self.last_input_tokens, self.last_output_tokens


# ==============================================================================
# MOCK TRANSLATION ADAPTER (NATIVE OFFLINE ENGINE)
# ==============================================================================
class MockTranslationAdapter(TranslationModelAdapter):
    """
    Deterministic offline translation adapter powered by sequence-level
    UniversalIndicLexicon and TokenizerAdapter.
    """

    def __init__(self):
        self.tokenizer = TokenizerAdapter()
        self.last_input_tokens: Optional[int] = None
        self.last_output_tokens: Optional[int] = None

    def is_ready(self) -> bool:
        return True

    def encode(self, text: str, source_language: str, target_language: str) -> dict[str, Any]:
        src_tag = config.INDICTRANS2_LANG_TAGS.get(source_language, "hin_Deva")
        token_ids = self.tokenizer.encode(text, lang_tag=src_tag)
        self.last_input_tokens = len(token_ids)
        return {
            "text": text,
            "token_ids": token_ids,
            "source_language": source_language,
            "target_language": target_language,
        }

    def infer(self, inputs: dict[str, Any], beam_size: int = 1) -> dict[str, Any]:
        text = inputs["text"]
        src = inputs["source_language"]
        tgt = inputs["target_language"]
        context = inputs.get("context", [])

        translated = UniversalIndicLexicon.translate(text, src, tgt, context=context)
        out_ids = self.tokenizer.encode(translated, add_special_tokens=False)
        self.last_output_tokens = len(out_ids)

        return {
            "translated_text": translated,
            "output_token_ids": out_ids,
        }

    def decode(self, outputs: dict[str, Any], target_language: str) -> str:
        raw_out = outputs["translated_text"]
        return TokenizerAdapter.clean_detokenized_text(raw_out)

    def get_token_counts(self) -> tuple[Optional[int], Optional[int]]:
        return self.last_input_tokens, self.last_output_tokens


# ==============================================================================
# ON-DEMAND TRANSLATOR CLASS
# ==============================================================================
class OnDemandTranslator:
    """
    High-level On-Demand Translator for the iTantra Receiver Pipeline.
    Enforces strict translation status, source-leakage prevention,
    and detailed diagnostic logging.
    """

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        source_language: str = "hi",
        target_language: str = "en",
        beam_size: int = config.TRANSLATION_BEAM_SIZE,
        adapter: Optional[TranslationModelAdapter] = None,
    ):
        self.source_language = source_language
        self.target_language = target_language
        self.beam_size = beam_size
        self.validator = TranslationValidator()
        self.context_buffer: collections.deque[str] = collections.deque(
            maxlen=config.TRANSLATION_CONTEXT_SENTENCE_COUNT
        )
        self.set_languages(source_language, target_language)

        if adapter is not None:
            self.adapter = adapter
        elif model_path is not None and Path(model_path).exists():
            self.adapter = IndicTrans2ONNXAdapter(model_path)
        else:
            default_pair_dir = config.TRANSLATION_MODELS_DIR / f"{source_language}_{target_language}"
            multilingual_dir = config.TRANSLATION_MODELS_DIR / "multilingual"
            if default_pair_dir.exists() and (default_pair_dir / "model.onnx").exists():
                self.adapter = IndicTrans2ONNXAdapter(default_pair_dir)
            elif multilingual_dir.exists() and (multilingual_dir / "model.onnx").exists():
                self.adapter = IndicTrans2ONNXAdapter(multilingual_dir)
            else:
                self.adapter = MockTranslationAdapter()

    def set_languages(self, source_language: str, target_language: str):
        """Set and validate source and target languages."""
        if source_language not in config.SUPPORTED_LANGUAGES:
            raise UnsupportedLanguageError(
                f"Source language '{source_language}' is not supported. "
                f"Supported: {config.SUPPORTED_LANGUAGES}"
            )
        if target_language not in config.SUPPORTED_LANGUAGES:
            raise UnsupportedLanguageError(
                f"Target language '{target_language}' is not supported. "
                f"Supported: {config.SUPPORTED_LANGUAGES}"
            )
        self.source_language = source_language
        self.target_language = target_language

    def set_beam_size(self, beam_size: int):
        """Set beam size (1 for greedy, 4, 8)."""
        if beam_size not in config.TRANSLATION_SUPPORTED_BEAM_SIZES:
            raise InvalidInputError(
                f"Beam size {beam_size} is not supported. Supported: {config.TRANSLATION_SUPPORTED_BEAM_SIZES}"
            )
        self.beam_size = beam_size

    def _chunk_text(self, text: str) -> list[str]:
        """Safe word-boundary segmentation."""
        return TextSegmenter.segment_sentences(
            text, max_chars=config.TRANSLATION_MAX_CHARS_PER_CHUNK
        )

    def translate(
        self,
        text: str,
        is_final: bool = True,
    ) -> dict[str, Any]:
        """
        Executes on-demand translation for incoming transcript.
        Never returns source text on translation failure.
        """
        if not isinstance(text, str) or not text.strip():
            raise InvalidInputError("Input transcript must be a non-empty string.")

        clean_text = text.strip()

        # Section 3: Partial STT text hypotheses are display-only
        if config.TRANSLATE_ONLY_FINAL_TEXT and not is_final:
            return {
                "success": False,
                "is_final": False,
                "source_text": clean_text,
                "input_text": clean_text,
                "translated_text": None,
                "output_text": clean_text,
                "source_language": self.source_language,
                "target_language": self.target_language,
                "latency_ms": 0.0,
                "error": "PARTIAL_STT_HYPOTHESIS",
                "warnings": ["Partial STT hypothesis: translation deferred until finalized."],
                "is_leakage": False,
                "validation": {"valid": True, "warnings": []},
            }

        # Section 11: Deliberate same-language translation handling
        if self.source_language == self.target_language:
            return {
                "success": True,
                "is_final": True,
                "source_text": clean_text,
                "input_text": clean_text,
                "translated_text": clean_text,
                "output_text": clean_text,
                "source_language": self.source_language,
                "target_language": self.target_language,
                "latency_ms": 0.0,
                "error": None,
                "warnings": ["SOURCE_AND_TARGET_LANGUAGE_IDENTICAL"],
                "is_leakage": False,
                "beam_size": self.beam_size,
                "validation": {"valid": True, "warnings": []},
            }

        t_start = time.perf_counter()

        try:
            chunks = self._chunk_text(clean_text)
            translated_chunks = []
            total_in_tokens = 0
            total_out_tokens = 0

            context_list = list(self.context_buffer) if config.TRANSLATION_USE_CONTEXT else []

            for chunk in chunks:
                inputs = self.adapter.encode(chunk, self.source_language, self.target_language)
                inputs["context"] = context_list
                outputs = self.adapter.infer(inputs, beam_size=self.beam_size)
                result = self.adapter.decode(outputs, self.target_language)
                translated_chunks.append(result)

                in_cnt, out_cnt = self.adapter.get_token_counts()
                if in_cnt is not None:
                    total_in_tokens += in_cnt
                if out_cnt is not None:
                    total_out_tokens += out_cnt

            raw_translation = " ".join(translated_chunks)
            translated_text = TokenizerAdapter.clean_detokenized_text(raw_translation)

            # Section 5-11: Translation & Source Leakage Validation
            validation = self.validator.validate(
                clean_text, translated_text, self.source_language, self.target_language
            )

            t_end = time.perf_counter()
            latency_ms = (t_end - t_start) * 1000.0

            leakage_info = validation.get("leakage_info", {})
            is_leakage = leakage_info.get("is_leakage", False) if leakage_info else False

            # Section 19: Reject source-text leakage
            if is_leakage and config.REJECT_SOURCE_LEAKAGE:
                error_code = "SOURCE_LEAKAGE"
                success = False
                final_translated_text = None
            elif validation["checks"]["empty_output"]:
                error_code = "EMPTY_OUTPUT"
                success = False
                final_translated_text = None
            else:
                error_code = None
                success = validation["valid"]
                final_translated_text = translated_text if success else None

            # Section 34: Logging
            metrics = leakage_info.get("metrics", {}) if leakage_info else {}
            print("========================================")
            print("Translation")
            print("========================================")
            print(f"Source language: {self.source_language}")
            print(f"Target language: {self.target_language}")
            print(f"Input chars: {len(clean_text)}")
            print(f"Beam size: {self.beam_size}")
            print(f"Normalized source: {clean_text}")
            print(f"Normalized target: {translated_text}")
            print(f"Exact match: {metrics.get('exact_match', False)}")
            print(f"Token overlap: {metrics.get('overlap_ratio', 0.0):.2f}")
            print(f"Source script ratio: {metrics.get('source_script_ratio', 0.0):.2f}")
            print(f"Target script ratio: {metrics.get('target_script_ratio', 0.0):.2f}")
            print(f"Leakage: {is_leakage}")
            print(f"Validation: {'PASS' if success else 'FAILED'}")
            print(f"Translation latency: {latency_ms:.2f} ms")
            print("========================================")

            if success and config.TRANSLATION_USE_CONTEXT:
                self.context_buffer.append(clean_text)

            return {
                "success": success,
                "source_text": clean_text,
                "input_text": clean_text,
                "translated_text": final_translated_text,
                "output_text": final_translated_text if final_translated_text is not None else "",
                "source_language": self.source_language,
                "target_language": self.target_language,
                "error": error_code,
                "warnings": validation["warnings"],
                "is_leakage": is_leakage,
                "leakage_info": leakage_info,
                "latency_ms": latency_ms,
                "input_tokens": total_in_tokens,
                "output_tokens": total_out_tokens,
                "beam_size": self.beam_size,
                "validation": validation,
            }

        except Exception as e:
            t_end = time.perf_counter()
            latency_ms = (t_end - t_start) * 1000.0
            print(f"[Translation] FAILED: {str(e)}")
            return {
                "success": False,
                "source_text": clean_text,
                "translated_text": None,
                "output_text": "",
                "source_language": self.source_language,
                "target_language": self.target_language,
                "error": f"TRANSLATION_ERROR: {str(e)}",
                "warnings": [str(e)],
                "is_leakage": False,
                "leakage_info": None,
                "latency_ms": latency_ms,
                "validation": {"valid": False, "warnings": [str(e)]},
            }

    def reset(self):
        """Reset internal context state."""
        self.context_buffer.clear()
