"""
Module 1: Silero VAD Wrapper (SileroVAD & StreamVAD)
===================================================
Streaming-style interface for Silero Voice Activity Detection using ONNX Runtime
(CPUExecutionProvider only).

Supports true online frame-by-frame streaming processing with event-driven state transitions:
- SPEECH_START (with pre-speech padding ring-buffer)
- SPEECH_ACTIVE (continuous streaming speech frames)
- SPEECH_END (silence duration reached + post-speech pad)
- SILENCE
"""

import sys
import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Generator
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from .config import (
        VAD_MODEL_PATH,
        VAD_SAMPLE_RATE,
        VAD_WINDOW_SAMPLES,
        VAD_THRESHOLD,
        VAD_MIN_SPEECH_MS,
        VAD_MIN_SILENCE_MS,
        VAD_SPEECH_PAD_MS,
    )
except ImportError:
    from config import (
        VAD_MODEL_PATH,
        VAD_SAMPLE_RATE,
        VAD_WINDOW_SAMPLES,
        VAD_THRESHOLD,
        VAD_MIN_SILENCE_MS,
        VAD_MIN_SPEECH_MS,
        VAD_SPEECH_PAD_MS,
    )

logger = logging.getLogger("SileroVAD")


class VADEventType(str, Enum):
    SILENCE = "SILENCE"
    SPEECH_START = "SPEECH_START"
    SPEECH_ACTIVE = "SPEECH_ACTIVE"
    SPEECH_END = "SPEECH_END"


@dataclass
class VADEvent:
    event_type: VADEventType
    prob: float
    frame: np.ndarray
    timestamp_s: float
    is_speech: bool
    segment_audio: Optional[np.ndarray] = None


class SileroVAD:
    """
    Silero VAD ONNX wrapper for frame-by-frame streaming processing and segment extraction.
    Ensures zero GPU dependency (CPUExecutionProvider).
    """

    def __init__(self, model_path: str = None, sample_rate: int = VAD_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.model_path = Path(model_path) if model_path else Path(VAD_MODEL_PATH)
        self.window_samples = VAD_WINDOW_SAMPLES

        self.session = None
        self.is_onnx_loaded = False
        self.input_names = []
        self.output_names = []

        # ONNX Recurrent State Tensors
        self._state = None
        self._h = None
        self._c = None
        self._sr_tensor = np.array(self.sample_rate, dtype=np.int64)
        self._context = np.zeros(64, dtype=np.float32)

        # Streaming state machine variables
        self.triggered = False
        self.consecutive_speech_frames = 0
        self.silence_samples = 0
        self.current_speech_frames = []
        self.pre_buffer = deque()
        self.sample_count = 0
        self.debug_log = []

        self._init_model()

    def _init_model(self):
        """Attempts to load ONNX model using CPUExecutionProvider."""
        if not self.model_path.exists():
            logger.warning(f"VAD ONNX model file not found at {self.model_path}. Fallback mode active.")
            return

        if ort is None:
            logger.warning("onnxruntime is not installed. Fallback VAD mode active.")
            return

        try:
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"]
            )
            
            input_meta = self.session.get_inputs()
            output_meta = self.session.get_outputs()
            
            self.input_names = [inp.name for inp in input_meta]
            self.output_names = [out.name for out in output_meta]

            self._init_states()
            self.is_onnx_loaded = True
            logger.info(f"Loaded Silero VAD ONNX model successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load VAD ONNX model: {e}. Falling back to energy heuristic.")
            self.session = None
            self.is_onnx_loaded = False

    def _init_states(self):
        """Initializes internal ONNX recurrent state tensors to zeros."""
        if not self.session:
            return

        for inp in self.session.get_inputs():
            shape = [dim if isinstance(dim, int) else 1 for dim in inp.shape]
            if inp.name == "state":
                self._state = np.zeros(shape if shape else [2, 1, 128], dtype=np.float32)
            elif inp.name == "h":
                self._h = np.zeros(shape if shape else [2, 1, 64], dtype=np.float32)
            elif inp.name == "c":
                self._c = np.zeros(shape if shape else [2, 1, 64], dtype=np.float32)
            elif inp.name == "sr":
                self._sr_tensor = np.array(self.sample_rate, dtype=np.int64)
        self._context = np.zeros(64, dtype=np.float32)

    def reset_state(self):
        """Resets VAD model recurrent state and internal state machine counters."""
        self._init_states()
        self._context = np.zeros(64, dtype=np.float32)
        self.triggered = False
        self.consecutive_speech_frames = 0
        self.silence_samples = 0
        self.current_speech_frames = []
        self.pre_buffer.clear()
        self.sample_count = 0
        self.debug_log = []

    def process_frame(self, frame: np.ndarray) -> float:
        """
        Processes one 32ms (e.g. 512-sample) audio frame at 16kHz.
        Returns speech probability between 0.0 and 1.0.
        """
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame, dtype=np.float32)
        else:
            frame = frame.astype(np.float32, copy=False)

        if len(frame) < self.window_samples:
            padded_frame = np.zeros(self.window_samples, dtype=np.float32)
            padded_frame[: len(frame)] = frame
            frame = padded_frame
        elif len(frame) > self.window_samples:
            frame = frame[: self.window_samples]

        # Silero VAD v5 requires 64 rolling context samples + 512 frame samples = 576 samples
        frame_input = np.concatenate([self._context, frame])
        self._context = frame[-64:].copy()
        frame_input = np.expand_dims(frame_input, axis=0)

        if self.is_onnx_loaded and self.session is not None:
            try:
                feed_dict = {}
                for inp_name in self.input_names:
                    if inp_name in ("input", "x", "audio"):
                        feed_dict[inp_name] = frame_input
                    elif inp_name == "sr":
                        feed_dict[inp_name] = self._sr_tensor if self._sr_tensor is not None else np.array(self.sample_rate, dtype=np.int64)
                    elif inp_name == "state":
                        if self._state is None:
                            self._state = np.zeros([2, 1, 128], dtype=np.float32)
                        feed_dict[inp_name] = self._state
                    elif inp_name == "h":
                        if self._h is None:
                            self._h = np.zeros([2, 1, 64], dtype=np.float32)
                        feed_dict[inp_name] = self._h
                    elif inp_name == "c":
                        if self._c is None:
                            self._c = np.zeros([2, 1, 64], dtype=np.float32)
                        feed_dict[inp_name] = self._c

                outputs = self.session.run(None, feed_dict)
                prob = float(outputs[0].squeeze())

                for idx, out_name in enumerate(self.output_names):
                    if out_name in ("state", "stateN") and len(outputs) > idx:
                        self._state = outputs[idx]
                    elif out_name == "hn" and len(outputs) > idx:
                        self._h = outputs[idx]
                    elif out_name == "cn" and len(outputs) > idx:
                        self._c = outputs[idx]

                return float(np.clip(prob, 0.0, 1.0))
            except Exception as e:
                logger.debug(f"ONNX inference error: {e}. Falling back to energy calculation.")

        # Fallback Energy Heuristic (when ONNX model file is absent)
        rms = np.sqrt(np.mean(frame**2) + 1e-10)
        prob = min(1.0, float(rms / 0.04))
        return prob

    def process_stream_frame(
        self,
        frame: np.ndarray,
        threshold: float = VAD_THRESHOLD,
        min_speech_ms: int = VAD_MIN_SPEECH_MS,
        min_silence_ms: int = VAD_MIN_SILENCE_MS,
        speech_pad_ms: int = VAD_SPEECH_PAD_MS,
    ) -> VADEvent:
        """
        True online streaming frame processor.
        Maintains ring-buffered pre-speech padding and emits instant transition events.
        """
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame, dtype=np.float32)
        else:
            frame = frame.astype(np.float32, copy=False)

        prob = self.process_frame(frame)
        timestamp_s = self.sample_count / float(self.sample_rate)
        self.sample_count += len(frame)

        # Calculate limits in samples
        pad_frames_count = max(1, int((speech_pad_ms * self.sample_rate / 1000) / self.window_samples))
        min_silence_samples = int(min_silence_ms * self.sample_rate / 1000)
        min_speech_samples = int(min_speech_ms * self.sample_rate / 1000)

        # Maintain pre-speech ring buffer
        if not self.triggered:
            self.pre_buffer.append(frame)
            if len(self.pre_buffer) > pad_frames_count:
                self.pre_buffer.popleft()

        event_type = VADEventType.SILENCE
        segment_audio = None

        if not self.triggered:
            if prob >= threshold:
                self.consecutive_speech_frames += 1
                if self.consecutive_speech_frames >= 2:
                    self.triggered = True
                    event_type = VADEventType.SPEECH_START
                    # Prepend pre-speech buffered frames
                    pre_frames = list(self.pre_buffer)
                    self.current_speech_frames = pre_frames + [frame]
                    self.pre_buffer.clear()
                    segment_audio = np.concatenate(self.current_speech_frames) if self.current_speech_frames else frame
            else:
                self.consecutive_speech_frames = 0
        else:  # Speech is currently active
            self.current_speech_frames.append(frame)
            if prob >= threshold:
                self.silence_samples = 0
                event_type = VADEventType.SPEECH_ACTIVE
            else:
                self.silence_samples += len(frame)
                if self.silence_samples >= min_silence_samples:
                    total_speech_audio = np.concatenate(self.current_speech_frames) if self.current_speech_frames else frame
                    if len(total_speech_audio) >= min_speech_samples:
                        event_type = VADEventType.SPEECH_END
                        segment_audio = total_speech_audio
                    else:
                        event_type = VADEventType.SILENCE

                    self.triggered = False
                    self.consecutive_speech_frames = 0
                    self.silence_samples = 0
                    self.current_speech_frames = []
                else:
                    event_type = VADEventType.SPEECH_ACTIVE

        # Log frame decision
        self.debug_log.append(
            {
                "frame_idx": len(self.debug_log),
                "start_sample": self.sample_count - len(frame),
                "timestamp_s": round(timestamp_s, 3),
                "probability": round(prob, 4),
                "triggered": self.triggered,
                "event": event_type.value if event_type != VADEventType.SILENCE else None,
            }
        )

        return VADEvent(
            event_type=event_type,
            prob=prob,
            frame=frame,
            timestamp_s=round(timestamp_s, 3),
            is_speech=self.triggered or event_type == VADEventType.SPEECH_START,
            segment_audio=segment_audio,
        )

    def get_speech_segments(
        self,
        audio: np.ndarray,
        threshold: float = VAD_THRESHOLD,
        min_speech_ms: int = VAD_MIN_SPEECH_MS,
        min_silence_ms: int = VAD_MIN_SILENCE_MS,
        speech_pad_ms: int = VAD_SPEECH_PAD_MS,
    ) -> list[tuple[int, int]]:
        """
        Feeds audio sequentially frame-by-frame and returns list of (start_sample, end_sample) speech regions.
        """
        self.reset_state()
        total_samples = len(audio)
        window_size = self.window_samples

        segments = []
        current_start = None

        for i in range(0, total_samples, window_size):
            frame = audio[i : i + window_size]
            event = self.process_stream_frame(
                frame,
                threshold=threshold,
                min_speech_ms=min_speech_ms,
                min_silence_ms=min_silence_ms,
                speech_pad_ms=speech_pad_ms,
            )

            if event.event_type == VADEventType.SPEECH_START and current_start is None:
                current_start = max(0, i - int(speech_pad_ms * self.sample_rate / 1000))
            elif event.event_type == VADEventType.SPEECH_END and current_start is not None:
                current_end = min(total_samples, i + window_size + int(speech_pad_ms * self.sample_rate / 1000))
                segments.append((current_start, current_end))
                current_start = None

        if current_start is not None:
            segments.append((current_start, total_samples))

        return segments
