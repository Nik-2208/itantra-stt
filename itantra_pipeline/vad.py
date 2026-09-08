"""
Module 1: Silero VAD Wrapper (SileroVAD)
========================================
Streaming-style interface for Silero Voice Activity Detection using ONNX Runtime
(CPUExecutionProvider only).

Simulates real-time frame-by-frame processing over audio streams and implements
a state machine with configurable thresholds, minimum speech/silence durations,
and boundary padding.
"""

import sys
import logging
from pathlib import Path
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
        VAD_MIN_SPEECH_MS,
        VAD_MIN_SILENCE_MS,
        VAD_SPEECH_PAD_MS,
    )

logger = logging.getLogger("SileroVAD")


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
        self._sr_tensor = None

        # Debug & logging history
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
            # Force CPU Execution Provider to reflect low-end mobile hardware reality
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

        # Prepare state tensors matching Silero VAD v4 / v5 ONNX inputs
        for inp in self.session.get_inputs():
            shape = [dim if isinstance(dim, int) else 1 for dim in inp.shape]
            if inp.name == "state":
                self._state = np.zeros(shape if shape else [2, 1, 128], dtype=np.float32)
            elif inp.name == "h":
                self._h = np.zeros(shape if shape else [2, 1, 64], dtype=np.float32)
            elif inp.name == "c":
                self._c = np.zeros(shape if shape else [2, 1, 64], dtype=np.float32)
            elif inp.name == "sr":
                self._sr_tensor = np.array([self.sample_rate], dtype=np.int64)

    def reset_state(self):
        """Resets VAD model recurrent state and internal state machine counters."""
        self._init_states()

    def process_frame(self, frame: np.ndarray) -> float:
        """
        Processes one 20ms/32ms (e.g. 512-sample) audio frame at 16kHz.
        Returns speech probability between 0.0 and 1.0.
        """
        # Ensure 1D float32 numpy array
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame, dtype=np.float32)
        else:
            frame = frame.astype(np.float32, copy=False)

        # Pad frame to window_samples if shorter
        if len(frame) < self.window_samples:
            padded_frame = np.zeros(self.window_samples, dtype=np.float32)
            padded_frame[: len(frame)] = frame
            frame = padded_frame
        elif len(frame) > self.window_samples:
            frame = frame[: self.window_samples]

        # Reshape to batch dimension [1, window_samples]
        frame_input = np.expand_dims(frame, axis=0)

        if self.is_onnx_loaded and self.session is not None:
            try:
                feed_dict = {}
                for inp_name in self.input_names:
                    if inp_name in ("input", "x", "audio"):
                        feed_dict[inp_name] = frame_input
                    elif inp_name == "sr":
                        feed_dict[inp_name] = self._sr_tensor if self._sr_tensor is not None else np.array([self.sample_rate], dtype=np.int64)
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

                # Update state variables from model outputs
                for idx, out_name in enumerate(self.output_names):
                    if out_name == "state" and len(outputs) > idx:
                        self._state = outputs[idx]
                    elif out_name == "hn" and len(outputs) > idx:
                        self._h = outputs[idx]
                    elif out_name == "cn" and len(outputs) > idx:
                        self._c = outputs[idx]

                return float(np.clip(prob, 0.0, 1.0))
            except Exception as e:
                logger.debug(f"ONNX inference error: {e}. Falling back to energy calculation.")

        # Fallback Energy Heuristic (when ONNX model file is absent)
        # Useful for offline tests prior to downloading 2MB silero ONNX file
        rms = np.sqrt(np.mean(frame**2) + 1e-10)
        prob = min(1.0, float(rms / 0.04))
        return prob

    def get_speech_segments(
        self,
        audio: np.ndarray,
        threshold: float = VAD_THRESHOLD,
        min_speech_ms: int = VAD_MIN_SPEECH_MS,
        min_silence_ms: int = VAD_MIN_SILENCE_MS,
        speech_pad_ms: int = VAD_SPEECH_PAD_MS,
    ) -> list[tuple[int, int]]:
        """
        Feeds audio sequentially frame-by-frame (simulating streaming mic pipeline)
        and runs standard Silero VAD state machine:
          - 2 consecutive frames above threshold -> speech START
          - min_silence_ms of frames below threshold -> speech END
          - speech_pad_ms added to both ends of segments
        Returns list of (start_sample, end_sample) speech regions.
        Logs decisions to self.debug_log.
        """
        self.reset_state()
        self.debug_log = []

        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        else:
            audio = audio.astype(np.float32, copy=False)

        total_samples = len(audio)
        window_size = self.window_samples

        min_speech_samples = int(min_speech_ms * self.sample_rate / 1000)
        min_silence_samples = int(min_silence_ms * self.sample_rate / 1000)
        speech_pad_samples = int(speech_pad_ms * self.sample_rate / 1000)

        triggered = False
        temp_start = 0
        consecutive_speech_frames = 0
        silence_samples = 0

        raw_segments = []
        frame_idx = 0

        for i in range(0, total_samples, window_size):
            frame = audio[i : i + window_size]
            prob = self.process_frame(frame)
            timestamp_s = i / self.sample_rate

            event = None

            if not triggered:
                if prob >= threshold:
                    consecutive_speech_frames += 1
                    if consecutive_speech_frames >= 2:
                        triggered = True
                        temp_start = max(0, i - (consecutive_speech_frames - 1) * window_size)
                        event = "SPEECH_START"
                else:
                    consecutive_speech_frames = 0
            else:  # triggered is True
                if prob >= threshold:
                    silence_samples = 0
                else:
                    silence_samples += len(frame)
                    if silence_samples >= min_silence_samples:
                        speech_end = i
                        seg_length = speech_end - temp_start

                        if seg_length >= min_speech_samples:
                            raw_segments.append((temp_start, speech_end))
                            event = "SPEECH_END"
                        else:
                            event = "BURST_DISCARDED"

                        triggered = False
                        consecutive_speech_frames = 0
                        silence_samples = 0

            self.debug_log.append(
                {
                    "frame_idx": frame_idx,
                    "start_sample": i,
                    "timestamp_s": round(timestamp_s, 3),
                    "probability": round(prob, 4),
                    "triggered": triggered,
                    "event": event,
                }
            )

            frame_idx += 1

        # Handle speech segment open at audio end
        if triggered:
            speech_end = total_samples
            if speech_end - temp_start >= min_speech_samples:
                raw_segments.append((temp_start, speech_end))

        # Apply padding and merge overlapping segments
        padded_segments = []
        for start, end in raw_segments:
            p_start = max(0, start - speech_pad_samples)
            p_end = min(total_samples, end + speech_pad_samples)

            if padded_segments and p_start <= padded_segments[-1][1]:
                # Merge overlapping
                prev_start, prev_end = padded_segments.pop()
                padded_segments.append((prev_start, max(prev_end, p_end)))
            else:
                padded_segments.append((p_start, p_end))

        return padded_segments
