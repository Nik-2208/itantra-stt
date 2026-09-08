"""
Module 4: Pipeline Orchestrator (VoicePipeline)
================================================
Chains Silero VAD -> Streaming STT -> Emergency Classifier with high-precision
time.perf_counter() latency profiling.

Ensures fully offline execution, CPU-only inference, and console/UI benchmark logs.
"""

import sys
import time
import logging
from pathlib import Path
import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import librosa
except ImportError:
    librosa = None

try:
    from .config import (
        VAD_MODEL_PATH,
        STT_MODEL_DIR,
        KEYWORDS_DIR,
        DEFAULT_SAMPLE_RATE,
        VAD_THRESHOLD,
        VAD_MIN_SILENCE_MS,
        VAD_MIN_SPEECH_MS,
        VAD_SPEECH_PAD_MS,
        STT_CHUNK_MS,
        STT_BEAM_SIZE,
        DEFAULT_LANGUAGE,
    )
    from .vad import SileroVAD
    from .stt import StreamingSTT
    from .emergency_classifier import EmergencyClassifier
except ImportError:
    from config import (
        VAD_MODEL_PATH,
        STT_MODEL_DIR,
        KEYWORDS_DIR,
        DEFAULT_SAMPLE_RATE,
        VAD_THRESHOLD,
        VAD_MIN_SILENCE_MS,
        VAD_MIN_SPEECH_MS,
        VAD_SPEECH_PAD_MS,
        STT_CHUNK_MS,
        STT_BEAM_SIZE,
        DEFAULT_LANGUAGE,
    )
    from vad import SileroVAD
    from stt import StreamingSTT
    from emergency_classifier import EmergencyClassifier

logger = logging.getLogger("VoicePipeline")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class VoicePipeline:
    """
    End-to-end Voice Pipeline orchestrator chaining VAD -> STT -> Emergency Classifier.
    Measures and logs millisecond-level per-stage latencies.
    """

    def __init__(
        self,
        vad_model_path: str = None,
        stt_model_dir: str = None,
        keywords_dir: str = None,
    ):
        self.vad = SileroVAD(model_path=vad_model_path)
        self.stt = StreamingSTT(model_dir=stt_model_dir)
        self.classifier = EmergencyClassifier(keywords_dir=keywords_dir)

    def load_audio(self, audio_source, target_sr: int = DEFAULT_SAMPLE_RATE) -> tuple[np.ndarray, int]:
        """
        Loads audio from file path or tuple (sr, numpy_array) and resamples to target_sr (16kHz) mono float32.
        """
        if isinstance(audio_source, (str, Path)):
            path_str = str(audio_source)
            if sf is not None:
                try:
                    data, sr = sf.read(path_str, dtype="float32")
                    if data.ndim > 1:
                        data = np.mean(data, axis=1)
                    if sr != target_sr:
                        if librosa is not None:
                            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
                        else:
                            # Basic linear interpolation fallback if librosa is absent
                            num_samples = int(len(data) * target_sr / sr)
                            data = np.interp(
                                np.linspace(0, len(data), num_samples),
                                np.arange(len(data)),
                                data
                            ).astype(np.float32)
                        sr = target_sr
                    return data, sr
                except Exception as e:
                    logger.warning(f"soundfile failed to read {path_str}: {e}. Trying librosa...")

            if librosa is not None:
                data, sr = librosa.load(path_str, sr=target_sr, mono=True)
                return data.astype(np.float32), sr

            raise RuntimeError("Neither soundfile nor librosa available to load audio file.")

        elif isinstance(audio_source, tuple) and len(audio_source) == 2:
            sr, data = audio_source
            if not isinstance(data, np.ndarray):
                data = np.array(data, dtype=np.float32)
            else:
                data = data.astype(np.float32, copy=False)

            if data.ndim > 1:
                data = np.mean(data, axis=1)

            # Normalize integer formats (int16/int32) to [-1.0, 1.0] float
            if data.dtype == np.int16:
                data = data / 32768.0
            elif data.dtype == np.int32:
                data = data / 2147483648.0

            if sr != target_sr and librosa is not None:
                data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
                sr = target_sr

            return data, sr

        elif isinstance(audio_source, np.ndarray):
            data = audio_source.astype(np.float32, copy=False)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            return data, target_sr

        else:
            raise ValueError(f"Unsupported audio source format: {type(audio_source)}")

    def process_audio(
        self,
        audio_source,
        language_code: str = DEFAULT_LANGUAGE,
        vad_threshold: float = VAD_THRESHOLD,
        vad_min_silence_ms: int = VAD_MIN_SILENCE_MS,
        vad_min_speech_ms: int = VAD_MIN_SPEECH_MS,
        vad_speech_pad_ms: int = VAD_SPEECH_PAD_MS,
        beam_size: int = STT_BEAM_SIZE,
    ) -> dict:
        """
        Executes full offline voice pipeline (VAD -> Streaming STT -> Emergency Classifier)
        and returns detailed result dictionary with stage latencies.
        """
        # Load & preprocess audio
        audio, sr = self.load_audio(audio_source, target_sr=DEFAULT_SAMPLE_RATE)
        audio_duration_s = len(audio) / float(sr)

        # ---------------------------------------------------------------------
        # STAGE 1: Silero VAD
        # ---------------------------------------------------------------------
        t_vad_start = time.perf_counter()
        vad_segments = self.vad.get_speech_segments(
            audio,
            threshold=vad_threshold,
            min_speech_ms=vad_min_speech_ms,
            min_silence_ms=vad_min_silence_ms,
            speech_pad_ms=vad_speech_pad_ms,
        )
        vad_latency_ms = (time.perf_counter() - t_vad_start) * 1000.0

        # Convert segment sample indices to timestamps (seconds)
        vad_segments_time_s = [
            (round(s / float(sr), 3), round(e / float(sr), 3))
            for s, e in vad_segments
        ]

        # ---------------------------------------------------------------------
        # STAGE 2: Streaming STT (Chunked feed simulation)
        # ---------------------------------------------------------------------
        t_stt_start = time.perf_counter()
        self.stt.set_language(language_code)
        self.stt.set_beam_size(beam_size)

        partial_transcripts = []
        final_transcript_parts = []
        early_emergency_result = None

        chunk_samples = int(STT_CHUNK_MS * sr / 1000)

        for start_sample, end_sample in vad_segments:
            speech_region = audio[start_sample:end_sample]
            self.stt.reset_stream()

            # Feed speech region in 100ms chunks to simulate streaming mic
            for chunk_offset in range(0, len(speech_region), chunk_samples):
                chunk = speech_region[chunk_offset : chunk_offset + chunk_samples]
                partial_text = self.stt.transcribe_chunk(chunk)
                
                timestamp_s = round((start_sample + chunk_offset) / float(sr), 2)
                partial_transcripts.append({"timestamp_s": timestamp_s, "text": partial_text})

                # Check emergency classification on partial hypothesis
                if not early_emergency_result or not early_emergency_result["is_emergency"]:
                    self.classifier.load_language(language_code)
                    partial_check = self.classifier.classify(partial_text)
                    if partial_check["is_emergency"]:
                        early_emergency_result = partial_check

            segment_final = self.stt.finalize()
            if segment_final:
                final_transcript_parts.append(segment_final)

        final_transcript = " ".join(final_transcript_parts)
        stt_latency_ms = (time.perf_counter() - t_stt_start) * 1000.0

        # ---------------------------------------------------------------------
        # STAGE 3: Emergency Classifier
        # ---------------------------------------------------------------------
        t_class_start = time.perf_counter()
        self.classifier.load_language(language_code)
        emergency_result = self.classifier.classify(final_transcript)
        classifier_latency_ms = (time.perf_counter() - t_class_start) * 1000.0

        # If emergency was detected early on partial hypothesis, elevate flag immediately
        if early_emergency_result and early_emergency_result["is_emergency"]:
            emergency_result["is_emergency"] = True
            emergency_result["priority"] = "P0"
            combined_keywords = list(dict.fromkeys(
                emergency_result["matched_keywords"] + early_emergency_result["matched_keywords"]
            ))
            emergency_result["matched_keywords"] = combined_keywords

        total_latency_ms = vad_latency_ms + stt_latency_ms + classifier_latency_ms
        rtf = round(total_latency_ms / (audio_duration_s * 1000.0 + 1e-6), 4)

        # Print benchmark output to console
        self._log_benchmark_summary(
            audio_duration_s=audio_duration_s,
            num_segments=len(vad_segments),
            vad_latency_ms=vad_latency_ms,
            stt_latency_ms=stt_latency_ms,
            classifier_latency_ms=classifier_latency_ms,
            total_latency_ms=total_latency_ms,
            rtf=rtf,
            emergency_result=emergency_result,
        )

        return {
            "sample_rate": sr,
            "audio_duration_s": round(audio_duration_s, 2),
            "vad_segments": vad_segments,
            "vad_segments_time_s": vad_segments_time_s,
            "vad_latency_ms": round(vad_latency_ms, 2),
            "vad_debug_log": self.vad.debug_log,
            "partial_transcripts": partial_transcripts,
            "final_transcript": final_transcript,
            "stt_latency_ms": round(stt_latency_ms, 2),
            "emergency_result": emergency_result,
            "classifier_latency_ms": round(classifier_latency_ms, 2),
            "total_latency_ms": round(total_latency_ms, 2),
            "real_time_factor": rtf,
        }

    def _log_benchmark_summary(
        self,
        audio_duration_s: float,
        num_segments: int,
        vad_latency_ms: float,
        stt_latency_ms: float,
        classifier_latency_ms: float,
        total_latency_ms: float,
        rtf: float,
        emergency_result: dict,
    ):
        """Prints clean console benchmark report for performance profiling."""
        print("\n" + "=" * 60)
        print("iTantra Voice Pipeline Latency Benchmark Report")
        print("=" * 60)
        print(f"Audio Duration        : {audio_duration_s:.2f} s")
        print(f"Speech Segments Found : {num_segments}")
        print("-" * 60)
        print(f"1. VAD Stage Latency  : {vad_latency_ms:.2f} ms")
        print(f"2. STT Stage Latency  : {stt_latency_ms:.2f} ms")
        print(f"3. Classifier Latency : {classifier_latency_ms:.2f} ms")
        print("-" * 60)
        print(f"TOTAL PIPELINE LATENCY: {total_latency_ms:.2f} ms")
        print(f"Real-Time Factor (RTF): {rtf:.4f}x (Lower is faster)")
        print("-" * 60)
        print(f"Emergency Priority    : {emergency_result.get('priority', 'P2')}")
        print(f"Matched Keywords      : {emergency_result.get('matched_keywords', [])}")
        print("=" * 60 + "\n")
