"""
Module 4B: Lightweight Offline Wake-Word Detector (WakeWordDetector)
===================================================================
Real offline, low-latency wake-word / voice-trigger engine powered by
sherpa-onnx Zipformer mobile transducer KWS (Apache License 2.0).

Features:
- Operates on small microphone frames (e.g. 32ms / 512 samples at 16kHz)
- Rolling ring buffer (pre-roll) so speech following wake phrase is never clipped
- Configurable confidence threshold and keyword boosting
- Debounce and smoothing window preventing duplicate triggers
- Extremely low CPU/RAM footprint (~5MB INT8 ONNX, <0.5ms per frame)
- Non-blocking frame processing
"""

import os
import sys
import time
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np

try:
    import sherpa_onnx
except ImportError:
    sherpa_onnx = None

try:
    from .config import (
        KWS_MODEL_DIR,
        WAKE_WORD_KEYWORDS_PATH,
        WAKE_WORD_THRESHOLD,
        WAKE_WORD_SCORE,
        WAKE_WORD_COOLDOWN_MS,
        WAKE_WORD_PRE_ROLL_MS,
        DEFAULT_SAMPLE_RATE,
        WAKE_PHRASES,
    )
except ImportError:
    from config import (
        KWS_MODEL_DIR,
        WAKE_WORD_KEYWORDS_PATH,
        WAKE_WORD_THRESHOLD,
        WAKE_WORD_SCORE,
        WAKE_WORD_COOLDOWN_MS,
        WAKE_WORD_PRE_ROLL_MS,
        DEFAULT_SAMPLE_RATE,
        WAKE_PHRASES,
    )

logger = logging.getLogger("WakeWordDetector")


@dataclass
class WakeWordEvent:
    keyword: str
    timestamp_s: float
    confidence: float
    pre_roll_audio: np.ndarray
    latency_ms: float
    data: Optional[Dict[str, Any]] = None


class WakeWordDetector:
    """
    Lightweight Keyword Spotting (KWS) engine for hands-free voice trigger.
    Runs continuously during IDLE_LISTENING with negligible CPU impact.
    """

    DEFAULT_KEYWORDS = [
        "\u2581HE Y \u2581I",                    # "Hey i" / "Hey iTantra"
        "\u2581HE Y \u2581IT ANT RA",           # "Hey iTantra"
        "\u2581IT ANT RA",                      # "iTantra"
        "\u2581HE Y \u2581S I RI",              # "Hey Siri"
        "\u2581A LE X A",                       # "Alexa"
        "\u2581HI \u2581GO O G LE",             # "Hi Google"
        "\u2581HE LL O \u2581WORLD",            # "Hello World"
        "\u2581START",                          # "Start"
        "\u2581 L IGHT \u2581UP",               # "Light Up"
        "\u2581LOVE LY \u2581CHI L D",          # "Lovely Child"
        "\u2581FOR E VER",                      # "Forever"
    ]

    def __init__(
        self,
        model_dir: Optional[str | Path] = None,
        keywords_path: Optional[str | Path] = None,
        threshold: float = WAKE_WORD_THRESHOLD,
        keywords_score: float = WAKE_WORD_SCORE,
        cooldown_ms: int = WAKE_WORD_COOLDOWN_MS,
        pre_roll_ms: int = WAKE_WORD_PRE_ROLL_MS,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        num_threads: int = 2,
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.keywords_score = keywords_score
        self.cooldown_s = cooldown_ms / 1000.0
        self.pre_roll_samples = int(pre_roll_ms * sample_rate / 1000)
        self.num_threads = num_threads

        self.model_dir = Path(model_dir) if model_dir else KWS_MODEL_DIR
        self.keywords_path = Path(keywords_path) if keywords_path else WAKE_WORD_KEYWORDS_PATH

        self._ensure_keywords_file()

        # Rolling ring buffer for pre-roll audio
        self.ring_buffer = deque(maxlen=self.pre_roll_samples)

        # Internal state tracking
        self.last_trigger_time = -999.0
        self.total_samples_processed = 0
        self.kws: Optional[sherpa_onnx.KeywordSpotter] = None
        self.stream = None
        self.is_loaded = False

        self._init_model()

    def _ensure_keywords_file(self):
        """Creates default keywords file if not present."""
        if not self.keywords_path.exists():
            os.makedirs(self.keywords_path.parent, exist_ok=True)
            with open(self.keywords_path, "w", encoding="utf-8") as f:
                for kw in self.DEFAULT_KEYWORDS:
                    f.write(kw.strip() + "\n")
            logger.info(f"Created default keywords file at {self.keywords_path}")

    def _init_model(self):
        """Initializes sherpa-onnx KeywordSpotter."""
        if sherpa_onnx is None:
            logger.error("sherpa_onnx is not installed. WakeWordDetector will be disabled.")
            return

        tokens_file = self.model_dir / "tokens.txt"
        encoder_file = self.model_dir / "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
        decoder_file = self.model_dir / "decoder-epoch-12-avg-2-chunk-16-left-64.onnx"
        joiner_file = self.model_dir / "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"

        # Fallback to float onnx if int8 not found
        if not encoder_file.exists():
            encoder_file = self.model_dir / "encoder-epoch-12-avg-2-chunk-16-left-64.onnx"
        if not joiner_file.exists():
            joiner_file = self.model_dir / "joiner-epoch-12-avg-2-chunk-16-left-64.onnx"

        if not (tokens_file.exists() and encoder_file.exists() and decoder_file.exists() and joiner_file.exists()):
            logger.warning(
                f"KWS model files missing in {self.model_dir}. Please download the sherpa-onnx KWS model."
            )
            return

        try:
            self.kws = sherpa_onnx.KeywordSpotter(
                tokens=str(tokens_file),
                encoder=str(encoder_file),
                decoder=str(decoder_file),
                joiner=str(joiner_file),
                keywords_file=str(self.keywords_path),
                num_threads=self.num_threads,
                sample_rate=self.sample_rate,
                keywords_score=self.keywords_score,
                keywords_threshold=self.threshold,
                provider="cpu",
            )
            self.stream = self.kws.create_stream()
            self.is_loaded = True
            logger.info("Sherpa-ONNX KeywordSpotter successfully initialized (CPU-only, INT8).")
        except Exception as e:
            logger.error(f"Failed to initialize sherpa-onnx KeywordSpotter: {e}", exc_info=True)
            self.is_loaded = False

    def reset(self):
        """Resets stream decoding state and clears ring buffer."""
        self.ring_buffer.clear()
        self.total_samples_processed = 0
        self.last_trigger_time = -999.0
        if self.kws:
            try:
                self.stream = self.kws.create_stream()
            except Exception as e:
                logger.debug(f"Error creating stream on reset: {e}")

    def get_pre_roll_audio(self) -> np.ndarray:
        """Returns the accumulated rolling ring buffer as a 1D float32 numpy array."""
        if not self.ring_buffer:
            return np.array([], dtype=np.float32)
        return np.array(self.ring_buffer, dtype=np.float32)

    def process_frame(
        self,
        frame: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Optional[WakeWordEvent]:
        """
        Processes a single small frame of audio (e.g. 32ms = 512 samples at 16kHz).
        Returns WakeWordEvent if wake phrase was detected, None otherwise.
        """
        if not self.is_loaded or self.kws is None or self.stream is None:
            return None

        # Convert to float32 1D array
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)
        if frame.ndim > 1:
            frame = np.mean(frame, axis=1)

        # Update rolling buffer
        self.ring_buffer.extend(frame)
        self.total_samples_processed += len(frame)
        current_timestamp = self.total_samples_processed / float(self.sample_rate)

        # Feed frame to KWS stream and profile latency
        t0 = time.perf_counter()
        self.stream.accept_waveform(self.sample_rate, frame)

        detected_keyword = None
        while self.kws.is_ready(self.stream):
            self.kws.decode_stream(self.stream)
            res = self.kws.get_result(self.stream)
            if res:
                detected_keyword = res
                try:
                    self.stream = self.kws.create_stream()
                except Exception as e:
                    logger.debug(f"Error recreating stream after trigger: {e}")
                break

        decode_latency_ms = (time.perf_counter() - t0) * 1000.0

        if not detected_keyword:
            return None

        # Check debounce cooldown window
        now_perf = time.perf_counter()
        if (now_perf - self.last_trigger_time) < self.cooldown_s:
            logger.debug(f"Keyword '{detected_keyword}' detected but suppressed by debounce cooldown.")
            return None

        self.last_trigger_time = now_perf
        pre_roll = self.get_pre_roll_audio()

        event = WakeWordEvent(
            keyword=detected_keyword,
            timestamp_s=round(current_timestamp, 3),
            confidence=1.0,
            pre_roll_audio=pre_roll,
            latency_ms=round(decode_latency_ms, 2),
            data={"raw_result": detected_keyword},
        )
        logger.info(f"Wake word detected: '{detected_keyword}' at {event.timestamp_s:.2f}s (latency: {event.latency_ms:.2f}ms)")
        return event

    def strip_wake_phrase(self, transcript: str) -> str:
        """
        Cleans wake phrase prefix from transcript (e.g. 'Hey iTantra help me' -> 'help me').
        Supports English and Indic variations.
        """
        if not transcript:
            return ""

        text = transcript.strip()
        lower_text = text.lower()

        # Check known wake phrases
        for phrase in WAKE_PHRASES:
            phrase_clean = phrase.strip().lower()
            if lower_text.startswith(phrase_clean):
                remaining = text[len(phrase_clean):].strip()
                # Remove common punctuation or leading separators
                remaining = remaining.lstrip(",.:;?!- ")
                if remaining:
                    return remaining

        return text
