"""
iTantra Receiver Web App - Benchmarking Timer & Resource Monitor (web_app/benchmark/timer.py)
==============================================================================================
High-precision nanosecond timer capturing pipeline milestones (t0 to t14),
calculating TTFA (Time To First Audio), RTF (Real-Time Factor), and memory/CPU usage.
"""

import os
import time
import psutil
from typing import Dict, Any, Optional
from web_app.schemas.message import BenchmarkMetrics

class PipelineBenchmarkTimer:
    """
    Precision milestone recorder for receiver-side ML pipeline.
    """

    def __init__(self, message_id: str, source_lang: str, target_lang: str, message_type: str, mode: str = "OPTIMIZED", chunk_size: int = 32):
        self.message_id = message_id
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.message_type = message_type
        self.mode = mode
        self.chunk_size = chunk_size
        self.process = psutil.Process(os.getpid())

        # Milestones (perf_counter floats)
        self.t0 = time.perf_counter()  # Text submitted
        self.t1 = 0.0  # Text displayed
        self.t2 = 0.0  # Emergency start
        self.t3 = 0.0  # Emergency complete
        self.t4 = 0.0  # Translation start
        self.t5 = 0.0  # First translation chunk
        self.t6 = 0.0  # Translation complete
        self.t7 = 0.0  # TTS start
        self.t8 = 0.0  # First text token processed
        self.t9 = 0.0  # First speech token generated
        self.t10 = 0.0 # First codec output
        self.t11 = 0.0 # First browser audio chunk dispatched
        self.t12 = 0.0 # Playback start
        self.t13 = 0.0 # TTS complete
        self.t14 = 0.0 # Playback complete

        self.audio_duration_sec: float = 0.0
        self.audio_chunks_count: int = 0
        self.underruns: int = 0
        self.codec_failures: int = 0
        self.initial_ram_mb = self._get_ram_mb()
        self.peak_ram_mb = self.initial_ram_mb

    def _get_ram_mb(self) -> float:
        try:
            return self.process.memory_info().rss / (1024.0 * 1024.0)
        except Exception:
            return 0.0

    def sample_memory(self):
        cur = self._get_ram_mb()
        if cur > self.peak_ram_mb:
            self.peak_ram_mb = cur

    def mark_text_displayed(self):
        self.t1 = time.perf_counter()
        self.sample_memory()

    def mark_emergency_start(self):
        self.t2 = time.perf_counter()

    def mark_emergency_complete(self):
        self.t3 = time.perf_counter()
        self.sample_memory()

    def mark_translation_start(self):
        self.t4 = time.perf_counter()

    def mark_first_translation_chunk(self):
        if self.t5 == 0.0:
            self.t5 = time.perf_counter()

    def mark_translation_complete(self):
        self.t6 = time.perf_counter()
        if self.t5 == 0.0:
            self.t5 = self.t6
        self.sample_memory()

    def mark_tts_start(self):
        self.t7 = time.perf_counter()

    def mark_first_text_token(self):
        if self.t8 == 0.0:
            self.t8 = time.perf_counter()

    def mark_first_speech_token(self):
        if self.t9 == 0.0:
            self.t9 = time.perf_counter()

    def mark_first_codec_audio(self):
        if self.t10 == 0.0:
            self.t10 = time.perf_counter()

    def mark_first_browser_audio(self):
        if self.t11 == 0.0:
            self.t11 = time.perf_counter()
        if self.t12 == 0.0:
            # First playback start is estimated from t11 + slight buffer latency (~15ms)
            self.t12 = self.t11 + 0.015
        self.sample_memory()

    def mark_audio_chunk(self, chunk_samples: int, sample_rate: int = 44100):
        self.audio_chunks_count += 1
        self.audio_duration_sec += (chunk_samples / sample_rate)
        self.sample_memory()

    def mark_tts_complete(self):
        self.t13 = time.perf_counter()
        if self.t14 == 0.0:
            self.t14 = self.t13 + self.audio_duration_sec
        self.sample_memory()

    def mark_playback_complete(self):
        self.t14 = time.perf_counter()
        self.sample_memory()

    def compute_metrics(self, text_length: int) -> BenchmarkMetrics:
        """
        Computes all stage latencies in milliseconds, TTFA, E2E, and RTF.
        """
        now = time.perf_counter()
        if self.t1 == 0.0: self.t1 = now
        if self.t3 == 0.0: self.t3 = self.t2 if self.t2 > 0 else now
        if self.t6 == 0.0: self.t6 = self.t4 if self.t4 > 0 else now
        if self.t5 == 0.0: self.t5 = self.t6
        if self.t13 == 0.0: self.t13 = now
        if self.t11 == 0.0: self.t11 = self.t13
        if self.t12 == 0.0: self.t12 = self.t11
        if self.t14 == 0.0: self.t14 = self.t13 + self.audio_duration_sec

        display_latency_ms = (self.t1 - self.t0) * 1000.0
        emergency_latency_ms = (self.t3 - self.t2) * 1000.0 if self.t2 > 0 else 0.0
        trans_first_chunk_ms = (self.t5 - self.t4) * 1000.0 if self.t4 > 0 else 0.0
        trans_total_ms = (self.t6 - self.t4) * 1000.0 if self.t4 > 0 else 0.0
        tts_first_token_ms = (self.t9 - self.t7) * 1000.0 if (self.t9 > 0 and self.t7 > 0) else 0.0
        codec_latency_ms = (self.t10 - self.t9) * 1000.0 if (self.t10 > 0 and self.t9 > 0) else 0.0
        ttfa_ms = (self.t11 - self.t7) * 1000.0 if (self.t11 > 0 and self.t7 > 0) else (self.t13 - self.t7) * 1000.0
        e2e_latency_ms = (self.t12 - self.t0) * 1000.0
        tts_total_ms = (self.t13 - self.t7) * 1000.0 if self.t7 > 0 else 0.0

        # RTF = TTS compute time / Audio duration
        rtf = (tts_total_ms / 1000.0) / max(0.01, self.audio_duration_sec)

        try:
            cpu_percent = psutil.cpu_percent(interval=None)
        except Exception:
            cpu_percent = 0.0

        return BenchmarkMetrics(
            message_id=self.message_id,
            source_language=self.source_lang,
            target_language=self.target_lang,
            message_type=self.message_type,
            mode=self.mode,
            text_length=text_length,
            t0_submit=self.t0,
            t1_display=self.t1,
            t2_emergency_start=self.t2,
            t3_emergency_complete=self.t3,
            t4_translation_start=self.t4,
            t5_first_translation_chunk=self.t5,
            t6_translation_complete=self.t6,
            t7_tts_start=self.t7,
            t8_first_text_token=self.t8,
            t9_first_speech_token=self.t9,
            t10_first_codec_audio=self.t10,
            t11_first_browser_audio=self.t11,
            t12_playback_start=self.t12,
            t13_tts_complete=self.t13,
            t14_playback_complete=self.t14,
            display_latency_ms=round(display_latency_ms, 2),
            emergency_latency_ms=round(emergency_latency_ms, 2),
            translation_first_chunk_ms=round(trans_first_chunk_ms, 2),
            translation_total_ms=round(trans_total_ms, 2),
            tts_first_token_ms=round(tts_first_token_ms, 2),
            codec_latency_ms=round(codec_latency_ms, 2),
            ttfa_ms=round(ttfa_ms, 2),
            e2e_latency_ms=round(e2e_latency_ms, 2),
            tts_total_ms=round(tts_total_ms, 2),
            audio_duration_sec=round(self.audio_duration_sec, 2),
            rtf=round(rtf, 3),
            peak_ram_mb=round(self.peak_ram_mb, 2),
            current_ram_mb=round(self._get_ram_mb(), 2),
            cpu_percent=round(cpu_percent, 1),
            token_chunk_size=self.chunk_size,
            audio_chunks_count=self.audio_chunks_count,
            underruns=self.underruns,
            codec_failures=self.codec_failures,
        )
