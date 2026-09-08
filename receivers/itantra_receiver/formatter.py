"""
iTantra Receiver Pipeline - Structured Output Formatter (formatter.py)
======================================================================
Model-independent formatting layer that transforms pipeline results into
a predictable, structured data object suitable for the future Android UI.

Architecture Rules:
- ZERO ML inference, ZERO translation, ZERO TTS.
- Deterministic, portable data structures (readily portable to Kotlin / C++ data classes).
- Handles source, translation, language display names, emergency status, and UI rendering strings.
"""

from datetime import datetime, timezone
from typing import Any, Optional
import config


class ReceiverFormatter:
    """
    Constructs structured receiver output dictionary from raw pipeline artifacts.
    """

    def format_output(
        self,
        source_text: str,
        translated_text: str,
        source_language: str,
        target_language: str,
        emergency_result: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Formats raw pipeline text and metadata into a structured receiver payload.
        """
        clean_source = source_text.strip() if isinstance(source_text, str) else ""
        clean_trans = translated_text.strip() if isinstance(translated_text, str) else ""

        src_name = config.LANGUAGE_NAMES.get(source_language, source_language.upper())
        tgt_name = config.LANGUAGE_NAMES.get(target_language, target_language.upper())

        # Emergency metadata handling
        if emergency_result is None:
            is_emergency = False
            priority = "NORMAL"
            matched_keywords = []
        else:
            is_emergency = bool(emergency_result.get("is_emergency", False))
            priority = str(emergency_result.get("priority", "P1" if is_emergency else "NORMAL"))
            matched_keywords = list(emergency_result.get("matched_keywords", []))

        # Status & Display generation
        status = "EMERGENCY" if is_emergency else "NORMAL"
        headline = f"[{status}] Incoming from {src_name}" if is_emergency else f"Received ({src_name} → {tgt_name})"
        body = clean_trans if clean_trans else clean_source

        # Human-readable formatted string for quick display
        formatted_summary = (
            f"SOURCE ({src_name}):\n"
            f'"{clean_source}"\n\n'
            f"TRANSLATION ({tgt_name}):\n"
            f'"{clean_trans}"\n\n'
            f"STATUS: {status}"
        )
        if is_emergency and matched_keywords:
            formatted_summary += f"\nKEYWORDS: {', '.join(matched_keywords)}"

        return {
            "source": {
                "language": source_language,
                "language_name": src_name,
                "text": clean_source,
            },
            "translation": {
                "language": target_language,
                "language_name": tgt_name,
                "text": clean_trans,
            },
            "emergency": {
                "is_emergency": is_emergency,
                "priority": priority,
                "matched_keywords": matched_keywords,
            },
            "display": {
                "headline": headline,
                "body": body,
                "status": status,
                "formatted_summary": formatted_summary,
            },
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        }
