"""
iTantra Receiver Pipeline - Main Orchestrator (pipeline.py)
===========================================================
Orchestrates the end-to-end receiver pipeline:
Incoming Source Transcript -> Language Validation -> On-Demand Translation ->
Formatting -> TTS Synthesis -> Audio Export -> Structured Return Payload.

Architecture Rules:
- Pure offline execution (zero network calls, zero cloud dependencies).
- Strict CPU-only ONNX execution.
- Precise nanosecond/microsecond measurement using time.perf_counter().
- Stable, language-agnostic data contract ready for Kotlin/C++ consumption.
"""

import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

import config
from formatter import ReceiverFormatter
from translation import (
    OnDemandTranslator,
    InvalidInputError,
    UnsupportedLanguageError,
    TranslationError,
)
from tts import OfflineTTS, TTSError


class ReceiverPipeline:
    """
    Main Receiver Pipeline coordinating translation, formatting, TTS, and timing.
    """

    def __init__(
        self,
        translator: Optional[OnDemandTranslator] = None,
        tts: Optional[OfflineTTS] = None,
        formatter: Optional[ReceiverFormatter] = None,
        output_dir: Path | str = config.TEMP_AUDIO_DIR,
    ):
        self.translator = translator if translator is not None else OnDemandTranslator()
        self.tts = tts if tts is not None else OfflineTTS()
        self.formatter = formatter if formatter is not None else ReceiverFormatter()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def translate_and_speak(
        self,
        text: str,
        source_language: str,
        target_language: str,
        emergency_result: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Executes on-demand translation, output formatting, and speech synthesis.
        """
        # Step 1: Input Validation
        if not isinstance(text, str) or not text.strip():
            raise InvalidInputError("Input transcript must be a non-empty string.")

        clean_text = text.strip()

        # Step 2: Set languages
        self.translator.set_languages(source_language, target_language)
        self.tts.set_language(target_language)

        # Global timer t0
        t0 = time.perf_counter()

        # Step 3: On-Demand Translation
        t_trans_0 = time.perf_counter()
        trans_result = self.translator.translate(clean_text)
        t_trans_1 = time.perf_counter()
        translation_latency_ms = (t_trans_1 - t_trans_0) * 1000.0
        translated_text = trans_result["output_text"]

        # Step 4: Structured Formatting
        t_form_0 = time.perf_counter()
        formatted_output = self.formatter.format_output(
            source_text=clean_text,
            translated_text=translated_text,
            source_language=source_language,
            target_language=target_language,
            emergency_result=emergency_result,
        )
        t_form_1 = time.perf_counter()
        formatter_latency_ms = (t_form_1 - t_form_0) * 1000.0

        # Step 5: TTS Speech Synthesis
        t_tts_0 = time.perf_counter()
        # If translation output is empty, synthesize source text
        tts_text = translated_text if translated_text else clean_text
        audio_array = self.tts.synthesize(tts_text)
        t_tts_1 = time.perf_counter()
        tts_latency_ms = (t_tts_1 - t_tts_0) * 1000.0

        # Step 6: Audio WAV file export
        t_io_0 = time.perf_counter()
        file_id = f"audio_{int(time.time())}_{uuid.uuid4().hex[:6]}.wav"
        audio_path = self.output_dir / file_id
        saved_path, _ = self.tts.save_audio(audio_array, audio_path)
        t_io_1 = time.perf_counter()
        audio_write_latency_ms = (t_io_1 - t_io_0) * 1000.0

        # Global timer t1
        t1 = time.perf_counter()
        total_latency_ms = (t1 - t0) * 1000.0

        # Audio duration & RTF calculation
        sample_rate = self.tts.sample_rate
        audio_duration_sec = len(audio_array) / float(sample_rate) if sample_rate > 0 else 0.0
        audio_duration_ms = audio_duration_sec * 1000.0

        # Real-Time Factor (RTF)
        tts_rtf = (tts_latency_ms / audio_duration_ms) if audio_duration_ms > 0 else 0.0
        end_to_end_rtf = (total_latency_ms / audio_duration_ms) if audio_duration_ms > 0 else 0.0

        # Print stage summary to console
        print("========================================")
        print("iTantra Receiver Pipeline")
        print("========================================")
        print(f"Translation:   {translation_latency_ms:8.2f} ms")
        print(f"Formatting:    {formatter_latency_ms:8.2f} ms")
        print(f"TTS:           {tts_latency_ms:8.2f} ms")
        print(f"Audio write:   {audio_write_latency_ms:8.2f} ms")
        print(f"TOTAL:         {total_latency_ms:8.2f} ms")
        print(f"Audio Duration:{audio_duration_ms:8.2f} ms (TTS RTF: {tts_rtf:.3f})")
        print("========================================")

        return {
            "source_text": clean_text,
            "source_language": source_language,
            "translated_text": translated_text,
            "target_language": target_language,
            "input": {
                "text": clean_text,
                "language": source_language,
            },
            "translation": {
                "text": translated_text,
                "language": target_language,
            },
            "audio": {
                "path": str(saved_path),
                "sample_rate": sample_rate,
                "duration_ms": audio_duration_ms,
            },
            "tts": {
                "audio_path": str(saved_path),
                "sample_rate": sample_rate,
                "duration_ms": audio_duration_ms,
            },
            "formatted_output": formatted_output,
            "timing": {
                "translation_latency_ms": translation_latency_ms,
                "formatter_latency_ms": formatter_latency_ms,
                "tts_latency_ms": tts_latency_ms,
                "audio_write_latency_ms": audio_write_latency_ms,
                "total_latency_ms": total_latency_ms,
            },
            "benchmark": {
                "translation_ms": translation_latency_ms,
                "formatting_ms": formatter_latency_ms,
                "tts_ms": tts_latency_ms,
                "audio_io_ms": audio_write_latency_ms,
                "total_ms": total_latency_ms,
                "tts_rtf": tts_rtf,
                "end_to_end_rtf": end_to_end_rtf,
            },
        }
