"""
iTantra Receiver Pipeline - Main Orchestrator (pipeline.py)
===========================================================
Orchestrates the end-to-end receiver pipeline:
Incoming Source Transcript -> Language Validation -> Safe Normalization ->
Word-Boundary Segmentation -> Tokenization -> Context-Aware Translation ->
Detokenization -> Strict Source-Leakage & Quality Validation ->
(TTS Speech Synthesis ONLY on Successful Translation) -> WAV Export ->
Structured Return Payload (Section 35 & 42 Contracts).

Architecture Rules:
- Pure offline execution (zero network calls, zero cloud dependencies).
- Strict CPU-only ONNX execution.
- TTS Safety Gate: Failed or leaked translation NEVER reaches TTS.
- Precise measurement using time.perf_counter().
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
    TokenizerAdapter,
    TextSegmenter,
    IndicScriptConverter,
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
        voice_id: Optional[str] = None,
        is_final: bool = True,
        emergency_result: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Executes on-demand translation, output formatting, and speech synthesis.
        Enforces TTS safety gate: TTS runs ONLY when translation succeeds.
        """
        t_global_0 = time.perf_counter()

        if not isinstance(text, str) or not text.strip():
            raise InvalidInputError("Input transcript must be a non-empty string.")

        # 1. Text Normalization
        t_norm_0 = time.perf_counter()
        clean_text = IndicScriptConverter.sanitize_text(text)
        t_norm_1 = time.perf_counter()
        normalization_ms = (t_norm_1 - t_norm_0) * 1000.0

        # 2. Set languages
        self.translator.set_languages(source_language, target_language)
        self.tts.set_voice(target_language, voice_id)

        # 3. Segmentation
        t_seg_0 = time.perf_counter()
        segments = TextSegmenter.segment_sentences(
            clean_text, max_chars=config.TRANSLATION_MAX_CHARS_PER_CHUNK
        )
        t_seg_1 = time.perf_counter()
        segmentation_ms = (t_seg_1 - t_seg_0) * 1000.0

        # 4. Tokenization
        t_tok_0 = time.perf_counter()
        in_tokens = self.translator.adapter.encode(clean_text, source_language, target_language)
        t_tok_1 = time.perf_counter()
        tokenization_ms = (t_tok_1 - t_tok_0) * 1000.0

        # 5. Translation Inference & Validation
        t_trans_0 = time.perf_counter()
        trans_result = self.translator.translate(clean_text, is_final=is_final)
        t_trans_1 = time.perf_counter()
        translation_ms = (t_trans_1 - t_trans_0) * 1000.0

        translated_text = trans_result.get("translated_text")
        trans_success = trans_result.get("success", False)
        val_info = trans_result.get("validation", {"valid": trans_success, "warnings": []})

        t_detok_0 = time.perf_counter()
        cleaned_trans = TokenizerAdapter.clean_detokenized_text(translated_text) if translated_text else ""
        t_detok_1 = time.perf_counter()
        detokenization_ms = (t_detok_1 - t_detok_0) * 1000.0
        validation_ms = 0.0

        # 6. Structured Formatting
        t_form_0 = time.perf_counter()
        formatted_output = self.formatter.format_output(
            source_text=clean_text,
            translated_text=cleaned_trans if trans_success else "",
            source_language=source_language,
            target_language=target_language,
            emergency_result=emergency_result,
        )
        t_form_1 = time.perf_counter()
        formatting_ms = (t_form_1 - t_form_0) * 1000.0

        # Section 20 & 29: HARD TTS SAFETY GATE
        # Never speak failed or leaked translation output!
        if not trans_success:
            t_global_1 = time.perf_counter()
            total_ms = (t_global_1 - t_global_0) * 1000.0
            print(f"[Pipeline] Translation Failed ({trans_result.get('error')}) | TTS Skipped.")

            return {
                "status": "TRANSLATION_FAILED",
                "source_text": clean_text,
                "source_language": source_language,
                "translated_text": None,
                "target_language": target_language,
                "formatted_output": formatted_output,
                "audio": None,
                "input": {
                    "text": clean_text,
                    "language": source_language,
                },
                "translation": {
                    "success": False,
                    "text": None,
                    "translated_text": None,
                    "source_text": clean_text,
                    "source_language": source_language,
                    "target_language": target_language,
                    "language": target_language,
                    "segments": segments,
                    "beam_size": self.translator.beam_size,
                    "error": trans_result.get("error", "TRANSLATION_FAILED"),
                    "warnings": trans_result.get("warnings", []),
                    "validation": val_info,
                    "leakage_info": trans_result.get("leakage_info"),
                },
                "tts": None,
                "timing": {
                    "normalization_ms": normalization_ms,
                    "segmentation_ms": segmentation_ms,
                    "tokenization_ms": tokenization_ms,
                    "translation_ms": translation_ms,
                    "detokenization_ms": detokenization_ms,
                    "validation_ms": validation_ms,
                    "tts_preprocess_ms": 0.0,
                    "tts_inference_ms": 0.0,
                    "tts_postprocess_ms": 0.0,
                    "audio_write_ms": 0.0,
                    "formatting_ms": formatting_ms,
                    "total_ms": total_ms,
                    "translation_latency_ms": translation_ms,
                    "formatter_latency_ms": formatting_ms,
                    "tts_latency_ms": 0.0,
                    "audio_write_latency_ms": 0.0,
                    "total_latency_ms": total_ms,
                },
                "benchmark": {
                    "translation_ms": translation_ms,
                    "formatting_ms": formatting_ms,
                    "tts_ms": None,
                    "audio_io_ms": None,
                    "total_ms": total_ms,
                    "tts_rtf": None,
                    "end_to_end_rtf": None,
                },
            }

        # 7. TTS Speech Synthesis (Executed ONLY on translation success)
        audio_array = self.tts.synthesize(cleaned_trans)
        tts_timing = self.tts.get_last_timing()
        tts_metrics = self.tts.get_last_metrics()

        tts_preprocess_ms = tts_timing.get("preprocess_ms", 0.0)
        tts_inference_ms = tts_timing.get("inference_ms", 0.0)
        tts_postprocess_ms = tts_timing.get("postprocess_ms", 0.0)

        # 8. Audio WAV file export
        t_io_0 = time.perf_counter()
        file_id = f"audio_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.wav"
        audio_path = self.output_dir / file_id
        saved_path, _ = self.tts.save_audio(audio_array, audio_path)
        t_io_1 = time.perf_counter()
        audio_write_ms = (t_io_1 - t_io_0) * 1000.0

        t_global_1 = time.perf_counter()
        total_ms = (t_global_1 - t_global_0) * 1000.0

        return {
            "status": "SUCCESS",
            "source_text": clean_text,
            "source_language": source_language,
            "translated_text": cleaned_trans,
            "target_language": target_language,
            "formatted_output": formatted_output,
            "audio": {
                "path": str(saved_path),
                "sample_rate": tts_metrics["sample_rate"],
                "duration_ms": tts_metrics["duration_ms"],
            },
            "input": {
                "text": clean_text,
                "language": source_language,
            },
            "translation": {
                "success": True,
                "text": cleaned_trans,
                "translated_text": cleaned_trans,
                "source_text": clean_text,
                "source_language": source_language,
                "target_language": target_language,
                "language": target_language,
                "segments": segments,
                "beam_size": self.translator.beam_size,
                "error": None,
                "warnings": trans_result.get("warnings", []),
                "validation": val_info,
                "leakage_info": trans_result.get("leakage_info"),
            },
            "tts": {
                "language": target_language,
                "voice_id": tts_metrics["voice_id"],
                "accent_id": tts_metrics["accent_id"],
                "speaker_id": tts_metrics["speaker_id"],
                "audio_path": str(saved_path),
                "sample_rate": tts_metrics["sample_rate"],
                "duration_ms": tts_metrics["duration_ms"],
                "rtf": tts_metrics["rtf"],
                "peak_amplitude": tts_metrics["peak_amplitude"],
                "rms": tts_metrics["rms"],
                "clipping_samples": tts_metrics["clipping_samples"],
            },
            "timing": {
                "normalization_ms": normalization_ms,
                "segmentation_ms": segmentation_ms,
                "tokenization_ms": tokenization_ms,
                "translation_ms": translation_ms,
                "detokenization_ms": detokenization_ms,
                "validation_ms": validation_ms,
                "tts_preprocess_ms": tts_preprocess_ms,
                "tts_inference_ms": tts_inference_ms,
                "tts_postprocess_ms": tts_postprocess_ms,
                "audio_write_ms": audio_write_ms,
                "formatting_ms": formatting_ms,
                "total_ms": total_ms,
                "translation_latency_ms": translation_ms,
                "formatter_latency_ms": formatting_ms,
                "tts_latency_ms": tts_timing.get("total_ms", 0.0),
                "audio_write_latency_ms": audio_write_ms,
                "total_latency_ms": total_ms,
            },
            "benchmark": {
                "translation_ms": translation_ms,
                "formatting_ms": formatting_ms,
                "tts_ms": tts_timing.get("total_ms", 0.0),
                "audio_io_ms": audio_write_ms,
                "total_ms": total_ms,
                "tts_rtf": tts_metrics["rtf"],
                "end_to_end_rtf": (total_ms / tts_metrics["duration_ms"]) if tts_metrics["duration_ms"] > 0 else 0.0,
            },
        }
