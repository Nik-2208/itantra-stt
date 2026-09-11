"""
Module 4: Pipeline Orchestrator (VoicePipeline)
================================================
Chains Silero VAD -> Streaming STT -> Emergency Classifier with high-precision
streaming latency profiling (time.perf_counter()).

Features:
- True generator-based streaming pipeline: process_stream(audio_generator)
- Sub-frame and chunk-by-chunk event emissions
- Real-time latency tracking: First-Token Latency (FTL), Chunk Execution Latency,
  Endpointing Latency (EOUL), and Real-Time Factor (RTF).
- Fully offline CPU execution.
"""

import sys
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Generator, List, Dict, Any, Optional, Tuple
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
        KWS_MODEL_DIR,
        WAKE_WORD_KEYWORDS_PATH,
        DEFAULT_SAMPLE_RATE,
        VAD_THRESHOLD,
        VAD_MIN_SILENCE_MS,
        VAD_MIN_SPEECH_MS,
        VAD_SPEECH_PAD_MS,
        STT_CHUNK_MS,
        STT_BEAM_SIZE,
        DEFAULT_LANGUAGE,
        WAKE_WORD_THRESHOLD,
        WAKE_WORD_SCORE,
        WAKE_WORD_COOLDOWN_MS,
        WAKE_WORD_PRE_ROLL_MS,
        COMMAND_TIMEOUT_MS,
        COMMAND_SILENCE_TIMEOUT_MS,
        WAKE_PHRASES,
    )
    from .vad import SileroVAD, VADEventType, VADEvent
    from .stt import StreamingSTT, STTChunkResult
    from .emergency_classifier import EmergencyClassifier
    from .wake_word import WakeWordDetector, WakeWordEvent
except ImportError:
    from config import (
        VAD_MODEL_PATH,
        STT_MODEL_DIR,
        KEYWORDS_DIR,
        KWS_MODEL_DIR,
        WAKE_WORD_KEYWORDS_PATH,
        DEFAULT_SAMPLE_RATE,
        VAD_THRESHOLD,
        VAD_MIN_SILENCE_MS,
        VAD_MIN_SPEECH_MS,
        VAD_SPEECH_PAD_MS,
        STT_CHUNK_MS,
        STT_BEAM_SIZE,
        DEFAULT_LANGUAGE,
        WAKE_WORD_THRESHOLD,
        WAKE_WORD_SCORE,
        WAKE_WORD_COOLDOWN_MS,
        WAKE_WORD_PRE_ROLL_MS,
        COMMAND_TIMEOUT_MS,
        COMMAND_SILENCE_TIMEOUT_MS,
        WAKE_PHRASES,
    )
    from vad import SileroVAD, VADEventType, VADEvent
    from stt import StreamingSTT, STTChunkResult
    from emergency_classifier import EmergencyClassifier
    from wake_word import WakeWordDetector, WakeWordEvent

logger = logging.getLogger("VoicePipeline")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class PipelineState(str, Enum):
    """Explicit state machine for hands-free voice trigger and comms."""
    IDLE_LISTENING = "IDLE_LISTENING"
    WAKE_DETECTED = "WAKE_DETECTED"
    COMMAND_CAPTURE = "COMMAND_CAPTURE"
    PROCESSING = "PROCESSING"
    NORMAL = "NORMAL"
    EMERGENCY = "EMERGENCY"
    TRANSMITTING = "TRANSMITTING"
    COOLDOWN = "COOLDOWN"


class StreamPipelineEventType(str, Enum):
    STATE_CHANGED = "STATE_CHANGED"
    WAKE_WORD_DETECTED = "WAKE_WORD_DETECTED"
    VAD_STATE = "VAD_STATE"
    PARTIAL_TRANSCRIPT = "PARTIAL_TRANSCRIPT"
    FINAL_TRANSCRIPT = "FINAL_TRANSCRIPT"
    EMERGENCY_ALERT = "EMERGENCY_ALERT"
    TRANSMISSION = "TRANSMISSION"
    METRICS_UPDATE = "METRICS_UPDATE"


@dataclass
class StreamPipelineEvent:
    event_type: StreamPipelineEventType
    timestamp_s: float
    data: Dict[str, Any]


@dataclass
class StreamMetrics:
    total_audio_duration_s: float = 0.0
    vad_latency_ms: float = 0.0
    stt_latency_ms: float = 0.0
    classifier_latency_ms: float = 0.0
    first_token_latency_ms: Optional[float] = None
    endpointing_latency_ms: Optional[float] = None
    chunk_latencies_ms: List[float] = field(default_factory=list)
    total_latency_ms: float = 0.0
    rtf: float = 0.0
    # Wake-word metrics
    wake_detection_latency_ms: Optional[float] = None
    wake_to_capture_latency_ms: Optional[float] = None
    total_wake_to_decision_latency_ms: Optional[float] = None
    cpu_percent: Optional[float] = None
    ram_rss_mb: Optional[float] = None

    @property
    def avg_chunk_latency_ms(self) -> float:
        return float(np.mean(self.chunk_latencies_ms)) if self.chunk_latencies_ms else 0.0

    @property
    def p95_chunk_latency_ms(self) -> float:
        return float(np.percentile(self.chunk_latencies_ms, 95)) if self.chunk_latencies_ms else 0.0


class VoicePipeline:
    """
    End-to-end Voice Pipeline orchestrator chaining VAD -> STT -> Emergency Classifier.
    Supports both generator-based frame-by-frame streaming and batch audio evaluation.
    """

    def __init__(
        self,
        vad_model_path: str = None,
        stt_model_dir: str = None,
        keywords_dir: str = None,
        kws_model_dir: str = None,
        kws_keywords_path: str = None,
    ):
        self.vad = SileroVAD(model_path=vad_model_path)
        self.stt = StreamingSTT(model_dir=stt_model_dir)
        self.classifier = EmergencyClassifier(keywords_dir=keywords_dir)
        self.wake_detector = WakeWordDetector(
            model_dir=kws_model_dir,
            keywords_path=kws_keywords_path,
        )
        self.current_state = PipelineState.IDLE_LISTENING

    def load_audio(self, audio_source, target_sr: int = DEFAULT_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
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
                            num_samples = int(len(data) * target_sr / sr)
                            data = np.interp(
                                np.linspace(0, len(data), num_samples),
                                np.arange(len(data)),
                                data
                            ).astype(np.float32)
                        sr = target_sr
                    return data, sr
                except Exception as e:
                    logger.warning(f"soundfile failed to read {path_str}: {e}. Trying librosa/wave/afconvert...")

            if librosa is not None:
                try:
                    data, sr = librosa.load(path_str, sr=target_sr, mono=True)
                    return data.astype(np.float32), sr
                except Exception as e:
                    logger.warning(f"librosa failed to read {path_str}: {e}. Trying wave/afconvert...")

            # Direct wave attempt (for standard WAV files)
            import wave
            try:
                with wave.open(path_str, "rb") as wf:
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    sr = wf.getframerate()
                    n_frames = wf.getnframes()
                    raw_bytes = wf.readframes(n_frames)

                    if sampwidth == 2:
                        data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    elif sampwidth == 4:
                        data = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
                    elif sampwidth == 1:
                        data = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                    else:
                        data = np.frombuffer(raw_bytes, dtype=np.float32)

                    if n_channels > 1:
                        data = data.reshape(-1, n_channels).mean(axis=1)

                    if sr != target_sr:
                        num_samples = int(len(data) * target_sr / sr)
                        data = np.interp(
                            np.linspace(0, len(data), num_samples),
                            np.arange(len(data)),
                            data
                        ).astype(np.float32)
                        sr = target_sr

                    return data, sr
            except Exception:
                pass

            # Universal native converter fallback for MP3 / M4A / WebM / AAC / OGG
            import subprocess
            import tempfile
            import os
            import shutil

            # Try macOS native afconvert
            afconvert_bin = shutil.which("afconvert") or "/usr/bin/afconvert"
            if os.path.exists(afconvert_bin):
                tmp_wav = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp_wav = tmp.name
                    cmd = [afconvert_bin, "-f", "WAVE", "-d", f"LEI16@{target_sr}", "-c", "1", path_str, tmp_wav]
                    res = subprocess.run(cmd, capture_output=True, text=True)
                    if res.returncode == 0 and os.path.exists(tmp_wav):
                        with wave.open(tmp_wav, "rb") as wf:
                            raw = wf.readframes(wf.getnframes())
                            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                            return data, target_sr
                except Exception as e:
                    logger.warning(f"afconvert failed for {path_str}: {e}")
                finally:
                    if tmp_wav and os.path.exists(tmp_wav):
                        try:
                            os.remove(tmp_wav)
                        except OSError:
                            pass

            # Try ffmpeg fallback if available
            ffmpeg_bin = shutil.which("ffmpeg")
            if ffmpeg_bin:
                tmp_wav = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp_wav = tmp.name
                    cmd = [ffmpeg_bin, "-y", "-i", path_str, "-ar", str(target_sr), "-ac", "1", "-f", "wav", tmp_wav]
                    res = subprocess.run(cmd, capture_output=True, text=True)
                    if res.returncode == 0 and os.path.exists(tmp_wav):
                        with wave.open(tmp_wav, "rb") as wf:
                            raw = wf.readframes(wf.getnframes())
                            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                            return data, target_sr
                except Exception as e:
                    logger.warning(f"ffmpeg failed for {path_str}: {e}")
                finally:
                    if tmp_wav and os.path.exists(tmp_wav):
                        try:
                            os.remove(tmp_wav)
                        except OSError:
                            pass

            raise RuntimeError(
                f"Could not load audio file '{path_str}'. Format not readable by standard wave reader, "
                f"and native conversion utilities were unavailable."
            )

        elif isinstance(audio_source, tuple) and len(audio_source) == 2:
            sr, data = audio_source
            if not isinstance(data, np.ndarray):
                data = np.array(data, dtype=np.float32)
            else:
                data = data.astype(np.float32, copy=False)

            if data.ndim > 1:
                data = np.mean(data, axis=1)

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

    def process_stream(
        self,
        frame_generator: Generator[np.ndarray, None, None],
        language_code: str = DEFAULT_LANGUAGE,
        vad_threshold: float = VAD_THRESHOLD,
        vad_min_silence_ms: int = VAD_MIN_SILENCE_MS,
        vad_min_speech_ms: int = VAD_MIN_SPEECH_MS,
        vad_speech_pad_ms: int = VAD_SPEECH_PAD_MS,
        beam_size: int = STT_BEAM_SIZE,
        enable_wake_word: bool = False,
        wake_threshold: float = WAKE_WORD_THRESHOLD,
        command_timeout_ms: int = COMMAND_TIMEOUT_MS,
        command_silence_timeout_ms: int = COMMAND_SILENCE_TIMEOUT_MS,
        cooldown_ms: int = WAKE_WORD_COOLDOWN_MS,
    ) -> Generator[StreamPipelineEvent, None, StreamMetrics]:
        """
        True online streaming generator.
        Supports both direct PTT speech streaming and hands-free Wake-Word trigger state machine.
        Takes real-time 32ms audio frames from a single shared mic/stream generator and yields events immediately.
        """
        import resource

        t_wall_start = time.perf_counter()
        t_cpu_start = time.process_time()

        self.vad.reset_state()
        self.stt.set_language(language_code)
        self.stt.set_beam_size(beam_size)
        self.stt.reset_stream()
        self.classifier.load_language(language_code)
        self.wake_detector.reset()

        metrics = StreamMetrics()
        sr = DEFAULT_SAMPLE_RATE
        stt_chunk_samples = int(STT_CHUNK_MS * sr / 1000)

        accumulated_speech_samples = []
        speech_start_perf_time = None
        speech_end_perf_time = None
        total_samples_processed = 0

        vad_total_time = 0.0
        stt_total_time = 0.0
        classifier_total_time = 0.0

        # Wake-word state machine tracking
        t_wake_detected = None
        command_start_time = None
        last_speech_time = None
        speech_started_in_command = False
        cooldown_end_time = 0.0
        wake_keyword = None

        if enable_wake_word:
            self.current_state = PipelineState.IDLE_LISTENING
            yield StreamPipelineEvent(
                event_type=StreamPipelineEventType.STATE_CHANGED,
                timestamp_s=0.0,
                data={"from_state": None, "to_state": PipelineState.IDLE_LISTENING.value, "reason": "Initial idle listening"},
            )

        for frame in frame_generator:
            total_samples_processed += len(frame)
            current_timestamp = total_samples_processed / float(sr)

            # =========================================================================
            # PATH A: WAKE-WORD STATE MACHINE (Hands-Free Voice Trigger)
            # =========================================================================
            if enable_wake_word:
                # -------------------------------------------------------------
                # STATE 1: IDLE_LISTENING -> Only lightweight KWS runs
                # -------------------------------------------------------------
                if self.current_state == PipelineState.IDLE_LISTENING:
                    wake_ev = self.wake_detector.process_frame(frame, threshold=wake_threshold)
                    if wake_ev is not None:
                        t_wake_detected = time.perf_counter()
                        wake_keyword = wake_ev.keyword
                        metrics.wake_detection_latency_ms = wake_ev.latency_ms

                        # Transition to WAKE_DETECTED
                        prev_state = self.current_state
                        self.current_state = PipelineState.WAKE_DETECTED
                        yield StreamPipelineEvent(
                            event_type=StreamPipelineEventType.STATE_CHANGED,
                            timestamp_s=round(current_timestamp, 3),
                            data={
                                "from_state": prev_state.value,
                                "to_state": PipelineState.WAKE_DETECTED.value,
                                "keyword": wake_ev.keyword,
                            },
                        )
                        yield StreamPipelineEvent(
                            event_type=StreamPipelineEventType.WAKE_WORD_DETECTED,
                            timestamp_s=round(current_timestamp, 3),
                            data={
                                "keyword": wake_ev.keyword,
                                "latency_ms": wake_ev.latency_ms,
                                "confidence": wake_ev.confidence,
                            },
                        )

                        # Transition immediately to COMMAND_CAPTURE
                        t_cap_start = time.perf_counter()
                        metrics.wake_to_capture_latency_ms = round((t_cap_start - t_wake_detected) * 1000.0, 3)

                        self.current_state = PipelineState.COMMAND_CAPTURE
                        yield StreamPipelineEvent(
                            event_type=StreamPipelineEventType.STATE_CHANGED,
                            timestamp_s=round(current_timestamp, 3),
                            data={
                                "from_state": PipelineState.WAKE_DETECTED.value,
                                "to_state": PipelineState.COMMAND_CAPTURE.value,
                                "transition_latency_ms": metrics.wake_to_capture_latency_ms,
                            },
                        )

                        # Initialize capture buffers
                        self.vad.reset_state()
                        self.stt.reset_stream()
                        accumulated_speech_samples.clear()
                        speech_start_perf_time = None
                        speech_started_in_command = False
                        command_start_time = current_timestamp
                        last_speech_time = current_timestamp

                        # Feed pre-roll audio into speech accumulator if available
                        pre_roll = wake_ev.pre_roll_audio
                        if len(pre_roll) > 0:
                            accumulated_speech_samples.extend(pre_roll[-stt_chunk_samples:])

                    continue  # In IDLE_LISTENING, VAD and STT stay dormant!

                # -------------------------------------------------------------
                # STATE 2: COMMAND_CAPTURE -> VAD + Incremental STT
                # -------------------------------------------------------------
                elif self.current_state == PipelineState.COMMAND_CAPTURE:
                    # VAD frame evaluation
                    t_v0 = time.perf_counter()
                    vad_event = self.vad.process_stream_frame(
                        frame,
                        threshold=vad_threshold,
                        min_speech_ms=vad_min_speech_ms,
                        min_silence_ms=vad_min_silence_ms,
                        speech_pad_ms=vad_speech_pad_ms,
                    )
                    vad_total_time += (time.perf_counter() - t_v0)

                    yield StreamPipelineEvent(
                        event_type=StreamPipelineEventType.VAD_STATE,
                        timestamp_s=round(current_timestamp, 3),
                        data={
                            "vad_event": vad_event.event_type.value,
                            "prob": round(vad_event.prob, 4),
                            "is_speech": vad_event.is_speech,
                        },
                    )

                    if vad_event.event_type in (VADEventType.SPEECH_START, VADEventType.SPEECH_ACTIVE):
                        speech_started_in_command = True
                        last_speech_time = current_timestamp
                        if speech_start_perf_time is None:
                            speech_start_perf_time = time.perf_counter()

                        if vad_event.segment_audio is not None and len(accumulated_speech_samples) == 0:
                            accumulated_speech_samples.extend(vad_event.segment_audio)
                        else:
                            accumulated_speech_samples.extend(frame)

                        # Incremental STT chunking
                        if len(accumulated_speech_samples) >= stt_chunk_samples:
                            chunk_to_proc = np.array(accumulated_speech_samples[:stt_chunk_samples], dtype=np.float32)
                            accumulated_speech_samples = accumulated_speech_samples[stt_chunk_samples:]

                            t_s0 = time.perf_counter()
                            stt_res = self.stt.process_chunk(chunk_to_proc)
                            stt_total_time += (time.perf_counter() - t_s0)
                            metrics.chunk_latencies_ms.append(stt_res.chunk_latency_ms)

                            if metrics.first_token_latency_ms is None and stt_res.partial_text and speech_start_perf_time:
                                metrics.first_token_latency_ms = (time.perf_counter() - speech_start_perf_time) * 1000.0

                            yield StreamPipelineEvent(
                                event_type=StreamPipelineEventType.PARTIAL_TRANSCRIPT,
                                timestamp_s=round(current_timestamp, 3),
                                data={
                                    "partial_transcript": stt_res.partial_text,
                                    "chunk_latency_ms": round(stt_res.chunk_latency_ms, 2),
                                },
                            )
                    elif vad_event.event_type == VADEventType.SPEECH_END:
                        last_speech_time = current_timestamp

                    # Check if command capture is complete:
                    # 1. Speech finished and silence duration reached
                    # 2. Or maximum command timeout exceeded
                    silence_elapsed = (current_timestamp - last_speech_time) if last_speech_time else 0.0
                    cmd_elapsed = (current_timestamp - command_start_time) if command_start_time else 0.0

                    is_silence_end = speech_started_in_command and (silence_elapsed >= (command_silence_timeout_ms / 1000.0))
                    is_timeout = cmd_elapsed >= (command_timeout_ms / 1000.0)

                    if is_silence_end or is_timeout:
                        # Transition to PROCESSING
                        prev_state = self.current_state
                        self.current_state = PipelineState.PROCESSING
                        yield StreamPipelineEvent(
                            event_type=StreamPipelineEventType.STATE_CHANGED,
                            timestamp_s=round(current_timestamp, 3),
                            data={
                                "from_state": prev_state.value,
                                "to_state": PipelineState.PROCESSING.value,
                                "reason": "silence_timeout" if is_silence_end else "max_command_timeout",
                            },
                        )

                        # Flush remaining audio
                        if len(accumulated_speech_samples) > 0:
                            tail = np.array(accumulated_speech_samples, dtype=np.float32)
                            t_s0 = time.perf_counter()
                            self.stt.process_chunk(tail)
                            stt_total_time += (time.perf_counter() - t_s0)
                            accumulated_speech_samples.clear()

                        t_fin0 = time.perf_counter()
                        raw_transcript = self.stt.finalize()
                        stt_total_time += (time.perf_counter() - t_fin0)

                        # Clean wake phrase from command
                        clean_transcript = self.wake_detector.strip_wake_phrase(raw_transcript)

                        # Emergency Classification (Wake detection itself NEVER means emergency)
                        t_c0 = time.perf_counter()
                        emergency_res = self.classifier.classify(clean_transcript)
                        classifier_total_time += (time.perf_counter() - t_c0)

                        is_emergency = emergency_res.get("is_emergency", False)
                        decision = "EMERGENCY" if is_emergency else "NORMAL"
                        priority = emergency_res.get("priority", "P0" if is_emergency else "P2")

                        # Transition to NORMAL or EMERGENCY
                        prev_state = self.current_state
                        self.current_state = PipelineState.EMERGENCY if is_emergency else PipelineState.NORMAL
                        yield StreamPipelineEvent(
                            event_type=StreamPipelineEventType.STATE_CHANGED,
                            timestamp_s=round(current_timestamp, 3),
                            data={
                                "from_state": prev_state.value,
                                "to_state": self.current_state.value,
                                "decision": decision,
                                "priority": priority,
                            },
                        )

                        if t_wake_detected:
                            metrics.total_wake_to_decision_latency_ms = (time.perf_counter() - t_wake_detected) * 1000.0

                        yield StreamPipelineEvent(
                            event_type=StreamPipelineEventType.FINAL_TRANSCRIPT,
                            timestamp_s=round(current_timestamp, 3),
                            data={
                                "final_transcript": clean_transcript,
                                "raw_transcript": raw_transcript,
                                "emergency_result": emergency_res,
                                "decision": decision,
                                "wake_keyword": wake_keyword,
                                "endpointing_latency_ms": round(silence_elapsed * 1000.0, 2),
                            },
                        )

                        # Transition to TRANSMITTING
                        prev_state = self.current_state
                        self.current_state = PipelineState.TRANSMITTING
                        yield StreamPipelineEvent(
                            event_type=StreamPipelineEventType.STATE_CHANGED,
                            timestamp_s=round(current_timestamp, 3),
                            data={
                                "from_state": prev_state.value,
                                "to_state": PipelineState.TRANSMITTING.value,
                                "channel": "SOS_PRIORITY_CHANNEL" if is_emergency else "STANDARD_COMM_CHANNEL",
                            },
                        )
                        yield StreamPipelineEvent(
                            event_type=StreamPipelineEventType.TRANSMISSION,
                            timestamp_s=round(current_timestamp, 3),
                            data={
                                "priority": priority,
                                "payload": clean_transcript,
                                "channel": "SOS_PRIORITY_CHANNEL" if is_emergency else "STANDARD_COMM_CHANNEL",
                                "timestamp_s": round(current_timestamp, 3),
                            },
                        )

                        # Transition to COOLDOWN
                        prev_state = self.current_state
                        self.current_state = PipelineState.COOLDOWN
                        cooldown_end_time = current_timestamp + (cooldown_ms / 1000.0)
                        yield StreamPipelineEvent(
                            event_type=StreamPipelineEventType.STATE_CHANGED,
                            timestamp_s=round(current_timestamp, 3),
                            data={
                                "from_state": prev_state.value,
                                "to_state": PipelineState.COOLDOWN.value,
                                "cooldown_ms": cooldown_ms,
                            },
                        )

                # -------------------------------------------------------------
                # STATE 3: COOLDOWN -> Debounce window before returning to IDLE
                # -------------------------------------------------------------
                elif self.current_state == PipelineState.COOLDOWN:
                    if current_timestamp >= cooldown_end_time:
                        prev_state = self.current_state
                        self.current_state = PipelineState.IDLE_LISTENING
                        self.wake_detector.reset()
                        self.vad.reset_state()
                        self.stt.reset_stream()
                        yield StreamPipelineEvent(
                            event_type=StreamPipelineEventType.STATE_CHANGED,
                            timestamp_s=round(current_timestamp, 3),
                            data={
                                "from_state": prev_state.value,
                                "to_state": PipelineState.IDLE_LISTENING.value,
                                "reason": "Cooldown period elapsed",
                            },
                        )

            # =========================================================================
            # PATH B: DIRECT STREAMING (PTT Mode without Wake Word)
            # =========================================================================
            else:
                # --- STAGE 1: Streaming VAD ---
                t_v0 = time.perf_counter()
                vad_event = self.vad.process_stream_frame(
                    frame,
                    threshold=vad_threshold,
                    min_speech_ms=vad_min_speech_ms,
                    min_silence_ms=vad_min_silence_ms,
                    speech_pad_ms=vad_speech_pad_ms,
                )
                vad_total_time += (time.perf_counter() - t_v0)

                yield StreamPipelineEvent(
                    event_type=StreamPipelineEventType.VAD_STATE,
                    timestamp_s=round(current_timestamp, 3),
                    data={
                        "vad_event": vad_event.event_type.value,
                        "prob": round(vad_event.prob, 4),
                        "is_speech": vad_event.is_speech,
                    },
                )

                # --- Handle VAD Transitions ---
                if vad_event.event_type == VADEventType.SPEECH_START:
                    self.stt.reset_stream()
                    speech_start_perf_time = time.perf_counter()
                    accumulated_speech_samples = []
                    if vad_event.segment_audio is not None:
                        accumulated_speech_samples.extend(vad_event.segment_audio)
                    else:
                        accumulated_speech_samples.extend(frame)

                elif vad_event.event_type == VADEventType.SPEECH_ACTIVE:
                    accumulated_speech_samples.extend(frame)

                    # Process incremental STT when accumulated chunk reaches target size (100ms)
                    if len(accumulated_speech_samples) >= stt_chunk_samples:
                        chunk_to_process = np.array(accumulated_speech_samples[:stt_chunk_samples], dtype=np.float32)
                        accumulated_speech_samples = accumulated_speech_samples[stt_chunk_samples:]

                        # --- STAGE 2: Incremental STT ---
                        t_s0 = time.perf_counter()
                        stt_result = self.stt.process_chunk(chunk_to_process)
                        stt_chunk_time = (time.perf_counter() - t_s0)
                        stt_total_time += stt_chunk_time
                        metrics.chunk_latencies_ms.append(stt_result.chunk_latency_ms)

                        # First Token Latency (FTL) measurement
                        if metrics.first_token_latency_ms is None and stt_result.partial_text and speech_start_perf_time:
                            metrics.first_token_latency_ms = (time.perf_counter() - speech_start_perf_time) * 1000.0

                        yield StreamPipelineEvent(
                            event_type=StreamPipelineEventType.PARTIAL_TRANSCRIPT,
                            timestamp_s=round(current_timestamp, 3),
                            data={
                                "partial_transcript": stt_result.partial_text,
                                "chunk_latency_ms": round(stt_result.chunk_latency_ms, 2),
                            },
                        )

                        # --- STAGE 3: Early Emergency Classification ---
                        t_c0 = time.perf_counter()
                        partial_emergency = self.classifier.classify(stt_result.partial_text)
                        classifier_total_time += (time.perf_counter() - t_c0)

                        if partial_emergency.get("is_emergency", False):
                            yield StreamPipelineEvent(
                                event_type=StreamPipelineEventType.EMERGENCY_ALERT,
                                timestamp_s=round(current_timestamp, 3),
                                data=partial_emergency,
                            )

                elif vad_event.event_type == VADEventType.SPEECH_END:
                    speech_end_perf_time = time.perf_counter()

                    # Process any trailing speech audio
                    if len(accumulated_speech_samples) > 0:
                        tail_chunk = np.array(accumulated_speech_samples, dtype=np.float32)
                        t_s0 = time.perf_counter()
                        self.stt.process_chunk(tail_chunk)
                        stt_total_time += (time.perf_counter() - t_s0)
                        accumulated_speech_samples.clear()

                    # Finalize STT
                    t_fin0 = time.perf_counter()
                    final_text = self.stt.finalize()
                    stt_total_time += (time.perf_counter() - t_fin0)

                    # Final Classification
                    t_c0 = time.perf_counter()
                    final_emergency = self.classifier.classify(final_text)
                    classifier_total_time += (time.perf_counter() - t_c0)

                    endpointing_ms = (time.perf_counter() - speech_end_perf_time) * 1000.0
                    metrics.endpointing_latency_ms = endpointing_ms

                    yield StreamPipelineEvent(
                        event_type=StreamPipelineEventType.FINAL_TRANSCRIPT,
                        timestamp_s=round(current_timestamp, 3),
                        data={
                            "final_transcript": final_text,
                            "emergency_result": final_emergency,
                            "endpointing_latency_ms": round(endpointing_ms, 2),
                        },
                    )

        # Handle unclosed speech region at end of stream
        if enable_wake_word and self.current_state == PipelineState.COMMAND_CAPTURE:
            prev_state = self.current_state
            self.current_state = PipelineState.PROCESSING
            yield StreamPipelineEvent(
                event_type=StreamPipelineEventType.STATE_CHANGED,
                timestamp_s=round(total_samples_processed / float(sr), 3),
                data={
                    "from_state": prev_state.value,
                    "to_state": PipelineState.PROCESSING.value,
                    "reason": "end_of_stream",
                },
            )

            if len(accumulated_speech_samples) > 0:
                tail = np.array(accumulated_speech_samples, dtype=np.float32)
                t_s0 = time.perf_counter()
                self.stt.process_chunk(tail)
                stt_total_time += (time.perf_counter() - t_s0)
                accumulated_speech_samples.clear()

            t_fin0 = time.perf_counter()
            raw_transcript = self.stt.finalize()
            stt_total_time += (time.perf_counter() - t_fin0)

            clean_transcript = self.wake_detector.strip_wake_phrase(raw_transcript)

            t_c0 = time.perf_counter()
            emergency_res = self.classifier.classify(clean_transcript)
            classifier_total_time += (time.perf_counter() - t_c0)

            is_emergency = emergency_res.get("is_emergency", False)
            decision = "EMERGENCY" if is_emergency else "NORMAL"
            priority = emergency_res.get("priority", "P0" if is_emergency else "P2")

            prev_state = self.current_state
            self.current_state = PipelineState.EMERGENCY if is_emergency else PipelineState.NORMAL
            yield StreamPipelineEvent(
                event_type=StreamPipelineEventType.STATE_CHANGED,
                timestamp_s=round(total_samples_processed / float(sr), 3),
                data={
                    "from_state": prev_state.value,
                    "to_state": self.current_state.value,
                    "decision": decision,
                    "priority": priority,
                },
            )

            if t_wake_detected:
                metrics.total_wake_to_decision_latency_ms = (time.perf_counter() - t_wake_detected) * 1000.0

            yield StreamPipelineEvent(
                event_type=StreamPipelineEventType.FINAL_TRANSCRIPT,
                timestamp_s=round(total_samples_processed / float(sr), 3),
                data={
                    "final_transcript": clean_transcript,
                    "raw_transcript": raw_transcript,
                    "emergency_result": emergency_res,
                    "decision": decision,
                    "wake_keyword": wake_keyword,
                    "endpointing_latency_ms": 0.0,
                },
            )

            prev_state = self.current_state
            self.current_state = PipelineState.TRANSMITTING
            yield StreamPipelineEvent(
                event_type=StreamPipelineEventType.STATE_CHANGED,
                timestamp_s=round(total_samples_processed / float(sr), 3),
                data={
                    "from_state": prev_state.value,
                    "to_state": PipelineState.TRANSMITTING.value,
                    "channel": "SOS_PRIORITY_CHANNEL" if is_emergency else "STANDARD_COMM_CHANNEL",
                },
            )
            yield StreamPipelineEvent(
                event_type=StreamPipelineEventType.TRANSMISSION,
                timestamp_s=round(total_samples_processed / float(sr), 3),
                data={
                    "priority": priority,
                    "payload": clean_transcript,
                    "channel": "SOS_PRIORITY_CHANNEL" if is_emergency else "STANDARD_COMM_CHANNEL",
                    "timestamp_s": round(total_samples_processed / float(sr), 3),
                },
            )

            # Return state to IDLE_LISTENING
            prev_state = self.current_state
            self.current_state = PipelineState.IDLE_LISTENING
            self.wake_detector.reset()
            self.vad.reset_state()
            self.stt.reset_stream()
            yield StreamPipelineEvent(
                event_type=StreamPipelineEventType.STATE_CHANGED,
                timestamp_s=round(total_samples_processed / float(sr), 3),
                data={
                    "from_state": prev_state.value,
                    "to_state": PipelineState.IDLE_LISTENING.value,
                    "reason": "Stream ended, pipeline returned to idle listening",
                },
            )

        elif (self.vad.triggered or self.stt.current_partial or len(accumulated_speech_samples) > 0) and not enable_wake_word:
            if len(accumulated_speech_samples) > 0:
                tail_chunk = np.array(accumulated_speech_samples, dtype=np.float32)
                self.stt.process_chunk(tail_chunk)
            final_text = self.stt.finalize()
            if final_text:
                final_emergency = self.classifier.classify(final_text)
                yield StreamPipelineEvent(
                    event_type=StreamPipelineEventType.FINAL_TRANSCRIPT,
                    timestamp_s=round(total_samples_processed / float(sr), 3),
                    data={
                        "final_transcript": final_text,
                        "emergency_result": final_emergency,
                        "endpointing_latency_ms": 0.0,
                    },
                )

        # Finalize Metrics
        metrics.total_audio_duration_s = total_samples_processed / float(sr)
        metrics.vad_latency_ms = vad_total_time * 1000.0
        metrics.stt_latency_ms = stt_total_time * 1000.0
        metrics.classifier_latency_ms = classifier_total_time * 1000.0
        metrics.total_latency_ms = metrics.vad_latency_ms + metrics.stt_latency_ms + metrics.classifier_latency_ms
        metrics.rtf = round(metrics.total_latency_ms / (metrics.total_audio_duration_s * 1000.0 + 1e-6), 4)

        # Measure actual CPU and RAM metrics
        total_wall_sec = time.perf_counter() - t_wall_start
        total_cpu_sec = time.process_time() - t_cpu_start
        metrics.cpu_percent = round((total_cpu_sec / (total_wall_sec + 1e-6)) * 100.0, 2)
        try:
            metrics.ram_rss_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0), 2)
        except Exception:
            metrics.ram_rss_mb = 0.0

        return metrics

    def process_audio(
        self,
        audio_source,
        language_code: str = DEFAULT_LANGUAGE,
        vad_threshold: float = VAD_THRESHOLD,
        vad_min_silence_ms: int = VAD_MIN_SILENCE_MS,
        vad_min_speech_ms: int = VAD_MIN_SPEECH_MS,
        vad_speech_pad_ms: int = VAD_SPEECH_PAD_MS,
        beam_size: int = STT_BEAM_SIZE,
        enable_wake_word: bool = False,
        wake_threshold: float = WAKE_WORD_THRESHOLD,
        command_timeout_ms: int = COMMAND_TIMEOUT_MS,
    ) -> dict:
        """
        Processes audio input through the streaming pipeline engine.
        Returns a detailed result dictionary with streaming latencies and wake detection stats.
        """
        audio, sr = self.load_audio(audio_source, target_sr=DEFAULT_SAMPLE_RATE)
        frame_size = self.vad.window_samples  # 512 samples = 32ms

        def frame_generator():
            for i in range(0, len(audio), frame_size):
                yield audio[i : i + frame_size]

        stream = self.process_stream(
            frame_generator=frame_generator(),
            language_code=language_code,
            vad_threshold=vad_threshold,
            vad_min_silence_ms=vad_min_silence_ms,
            vad_min_speech_ms=vad_min_speech_ms,
            vad_speech_pad_ms=vad_speech_pad_ms,
            beam_size=beam_size,
            enable_wake_word=enable_wake_word,
            wake_threshold=wake_threshold,
            command_timeout_ms=command_timeout_ms,
        )

        partial_transcripts = []
        final_transcripts = []
        emergency_result = None
        early_emergency = None
        wake_word_event = None
        state_transitions = []
        transmission_event = None

        try:
            while True:
                event = next(stream)
                if event.event_type == StreamPipelineEventType.STATE_CHANGED:
                    state_transitions.append({
                        "timestamp_s": event.timestamp_s,
                        "state": event.data.get("to_state"),
                        "data": event.data,
                    })
                elif event.event_type == StreamPipelineEventType.WAKE_WORD_DETECTED:
                    wake_word_event = event.data
                elif event.event_type == StreamPipelineEventType.PARTIAL_TRANSCRIPT:
                    partial_transcripts.append({
                        "timestamp_s": event.timestamp_s,
                        "text": event.data["partial_transcript"],
                        "chunk_latency_ms": event.data.get("chunk_latency_ms", 0.0),
                    })
                elif event.event_type == StreamPipelineEventType.FINAL_TRANSCRIPT:
                    final_transcripts.append(event.data["final_transcript"])
                    emergency_result = event.data["emergency_result"]
                elif event.event_type == StreamPipelineEventType.EMERGENCY_ALERT:
                    early_emergency = event.data
                elif event.event_type == StreamPipelineEventType.TRANSMISSION:
                    transmission_event = event.data
        except StopIteration as e:
            metrics = e.value

        final_transcript = " ".join(final_transcripts).strip()
        if emergency_result is None:
            self.classifier.load_language(language_code)
            emergency_result = self.classifier.classify(final_transcript)

        if early_emergency and early_emergency.get("is_emergency", False):
            emergency_result["is_emergency"] = True
            emergency_result["priority"] = "P0"
            emergency_result["matched_keywords"] = list(dict.fromkeys(
                emergency_result.get("matched_keywords", []) + early_emergency.get("matched_keywords", [])
            ))

        # Compute speech segments for visualization compatibility
        vad_segments = self.vad.get_speech_segments(
            audio,
            threshold=vad_threshold,
            min_speech_ms=vad_min_speech_ms,
            min_silence_ms=vad_min_silence_ms,
            speech_pad_ms=vad_speech_pad_ms,
        )
        vad_segments_time_s = [
            (round(s / float(sr), 3), round(e / float(sr), 3))
            for s, e in vad_segments
        ]

        self._log_benchmark_summary(
            audio_duration_s=metrics.total_audio_duration_s,
            num_segments=len(vad_segments),
            vad_latency_ms=metrics.vad_latency_ms,
            stt_latency_ms=metrics.stt_latency_ms,
            classifier_latency_ms=metrics.classifier_latency_ms,
            total_latency_ms=metrics.total_latency_ms,
            rtf=metrics.rtf,
            emergency_result=emergency_result,
            metrics=metrics,
            enable_wake_word=enable_wake_word,
            wake_word_event=wake_word_event,
        )

        decision = "EMERGENCY" if emergency_result.get("is_emergency", False) else "NORMAL"

        return {
            "sample_rate": sr,
            "audio_duration_s": round(metrics.total_audio_duration_s, 2),
            "vad_segments": vad_segments,
            "vad_segments_time_s": vad_segments_time_s,
            "vad_latency_ms": round(metrics.vad_latency_ms, 2),
            "vad_debug_log": self.vad.debug_log,
            "partial_transcripts": partial_transcripts,
            "final_transcript": final_transcript,
            "stt_latency_ms": round(metrics.stt_latency_ms, 2),
            "emergency_result": emergency_result,
            "decision": decision,
            "classifier_latency_ms": round(metrics.classifier_latency_ms, 2),
            "total_latency_ms": round(metrics.total_latency_ms, 2),
            "real_time_factor": metrics.rtf,
            "first_token_latency_ms": round(metrics.first_token_latency_ms, 2) if metrics.first_token_latency_ms is not None else None,
            "endpointing_latency_ms": round(metrics.endpointing_latency_ms, 2) if metrics.endpointing_latency_ms is not None else None,
            "avg_chunk_latency_ms": round(metrics.avg_chunk_latency_ms, 2),
            "p95_chunk_latency_ms": round(metrics.p95_chunk_latency_ms, 2),
            # Wake-word metrics & state transitions
            "enable_wake_word": enable_wake_word,
            "wake_word_detected": wake_word_event is not None,
            "wake_word_event": wake_word_event,
            "state_transitions": state_transitions,
            "transmission_event": transmission_event,
            "wake_detection_latency_ms": metrics.wake_detection_latency_ms,
            "wake_to_capture_latency_ms": metrics.wake_to_capture_latency_ms,
            "total_wake_to_decision_latency_ms": metrics.total_wake_to_decision_latency_ms,
            "cpu_percent": metrics.cpu_percent,
            "ram_rss_mb": metrics.ram_rss_mb,
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
        metrics: Optional[StreamMetrics] = None,
        enable_wake_word: bool = False,
        wake_word_event: Optional[dict] = None,
    ):
        """Prints clean console benchmark report for performance profiling."""
        print("\n" + "=" * 65)
        print("iTantra Streaming Voice Pipeline Latency Benchmark Report")
        print("=" * 65)
        print(f"Audio Duration          : {audio_duration_s:.2f} s")
        print(f"Speech Segments Found   : {num_segments}")
        if enable_wake_word:
            kw = wake_word_event.get("keyword") if wake_word_event else "None detected"
            w_lat = f"{metrics.wake_detection_latency_ms:.2f} ms" if metrics and metrics.wake_detection_latency_ms else "N/A"
            w2c = f"{metrics.wake_to_capture_latency_ms:.2f} ms" if metrics and metrics.wake_to_capture_latency_ms else "N/A"
            w2d = f"{metrics.total_wake_to_decision_latency_ms:.2f} ms" if metrics and metrics.total_wake_to_decision_latency_ms else "N/A"
            print(f"Wake Word Mode          : ENABLED (Keyword: {kw})")
            print(f"Wake Detection Latency  : {w_lat}")
            print(f"Wake -> Capture Latency : {w2c}")
            print(f"Wake -> Decision Latency: {w2d}")
        print("-" * 65)
        print(f"1. VAD Stage Latency    : {vad_latency_ms:.2f} ms")
        print(f"2. STT Stage Latency    : {stt_latency_ms:.2f} ms")
        print(f"3. Classifier Latency   : {classifier_latency_ms:.2f} ms")
        print("-" * 65)
        print(f"TOTAL PIPELINE LATENCY  : {total_latency_ms:.2f} ms")
        print(f"Real-Time Factor (RTF)  : {rtf:.4f}x (Lower is faster)")
        if metrics:
            ftl = f"{metrics.first_token_latency_ms:.2f} ms" if metrics.first_token_latency_ms is not None else "N/A"
            eoul = f"{metrics.endpointing_latency_ms:.2f} ms" if metrics.endpointing_latency_ms is not None else "N/A"
            print(f"First-Token Latency     : {ftl}")
            print(f"Avg Chunk Latency       : {metrics.avg_chunk_latency_ms:.2f} ms (p95: {metrics.p95_chunk_latency_ms:.2f} ms)")
            print(f"Endpointing Latency     : {eoul}")
            if metrics.cpu_percent is not None:
                print(f"Process CPU Load        : {metrics.cpu_percent:.1f}%")
            if metrics.ram_rss_mb is not None:
                print(f"Process RAM (Max RSS)   : {metrics.ram_rss_mb:.2f} MB")
        print("-" * 65)
        print(f"Emergency Decision      : {'🚨 EMERGENCY' if emergency_result.get('is_emergency') else '✅ NORMAL'} ({emergency_result.get('priority', 'P2')})")
        print(f"Matched Keywords        : {emergency_result.get('matched_keywords', [])}")
        print("=" * 65 + "\n")
