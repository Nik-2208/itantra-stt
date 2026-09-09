"""
iTantra Receiver Web App - Translation Engine (web_app/pipeline/translation_engine.py)
=====================================================================================
Modular translation engine wrapping AI4Bharat IndicTrans2.
Features:
- Immediate bypass when source_language == target_language (0ms).
- Incremental phrase/sentence chunking.
- High-fidelity offline Indic-to-Indic and Indic-to-English translation.
"""

import time
from typing import Generator, Tuple, Optional, List
from web_app.config.settings import LanguageCode, INDICTRANS2_TAGS
from translation import OfflineIndicTranslationEngine, OnDemandTranslator

class TranslationEngine:
    """
    Offline translation engine with same-language bypass and phrase/sentence streaming.
    """

    def __init__(self):
        self._translator = OnDemandTranslator()

    def translate_full(
        self, text: str, source_lang: LanguageCode, target_lang: LanguageCode
    ) -> Tuple[str, float]:
        """
        Translates text. If source == target, returns text directly in 0 ms.
        Returns: (translated_text, latency_ms)
        """
        if source_lang == target_lang:
            return text, 0.0

        t0 = time.perf_counter()
        src_code = source_lang.value.lower()
        tgt_code = target_lang.value.lower()

        # Check phrase match in offline dictionary or model
        result = OfflineIndicTranslationEngine.translate_text(text, src_code, tgt_code)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return result, latency_ms

    def translate_stream(
        self, text: str, source_lang: LanguageCode, target_lang: LanguageCode
    ) -> Generator[Tuple[str, bool, float], None, None]:
        """
        Yields (chunk_text, is_final, latency_ms).
        Splits text by punctuation boundaries and translates clause-by-clause.
        """
        if source_lang == target_lang:
            yield text, True, 0.0
            return

        import re
        # Clause and sentence punctuation in Indic and Latin scripts
        delimiters = r"([।\n.!?]+)"
        tokens = re.split(delimiters, text)

        chunks: List[str] = []
        current = ""
        for tok in tokens:
            if not tok:
                continue
            if re.match(delimiters, tok):
                current += tok
                chunks.append(current.strip())
                current = ""
            else:
                current += tok
        if current.strip():
            chunks.append(current.strip())

        if not chunks:
            chunks = [text]

        t0 = time.perf_counter()
        for idx, chunk in enumerate(chunks):
            is_final = (idx == len(chunks) - 1)
            src_code = source_lang.value.lower()
            tgt_code = target_lang.value.lower()
            t_chunk_0 = time.perf_counter()
            translated_chunk = OfflineIndicTranslationEngine.translate_text(chunk, src_code, tgt_code)
            chunk_latency = (time.perf_counter() - t_chunk_0) * 1000.0
            yield translated_chunk, is_final, chunk_latency
