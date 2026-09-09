"""
iTantra Receiver Web App - Schemas & Data Contracts (web_app/schemas/message.py)
=================================================================================
Transport-independent models matching the receiver pipeline contracts for
seamless future migration to Android/Kotlin.
"""

import time
import uuid
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from web_app.config.settings import LanguageCode, MessageType

class VoiceProfile(BaseModel):
    voice_id: str = "default_neutral"
    language: LanguageCode = LanguageCode.HI
    accent_label: str = "Standard / Neutral"
    style_tag: str = "NORMAL"  # NORMAL, ALERT, SOS
    reference_audio_path: Optional[str] = None
    speaker_embedding: Optional[List[float]] = None
    is_default: bool = True

class ReceivedMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = "node_sender_01"
    source_language: LanguageCode = LanguageCode.HI
    target_language: LanguageCode = LanguageCode.HI
    timestamp: float = Field(default_factory=time.time)
    message_type: MessageType = MessageType.NORMAL
    text: str
    sequence_number: int = 1
    voice_profile: Optional[VoiceProfile] = None

class EmergencyResult(BaseModel):
    is_emergency: bool = False
    priority: MessageType = MessageType.NORMAL
    matched_terms: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    detected_pass: str = "none"  # "source", "translated", "both", "none"

class BenchmarkMetrics(BaseModel):
    message_id: str
    source_language: str
    target_language: str
    message_type: str
    mode: str = "OPTIMIZED"  # "BASELINE" or "OPTIMIZED"
    text_length: int
    t0_submit: float
    t1_display: float
    t2_emergency_start: float
    t3_emergency_complete: float
    t4_translation_start: float
    t5_first_translation_chunk: float
    t6_translation_complete: float
    t7_tts_start: float
    t8_first_text_token: float
    t9_first_speech_token: float
    t10_first_codec_audio: float
    t11_first_browser_audio: float
    t12_playback_start: float
    t13_tts_complete: float
    t14_playback_complete: float
    # Calculated latencies (ms)
    display_latency_ms: float = 0.0
    emergency_latency_ms: float = 0.0
    translation_first_chunk_ms: float = 0.0
    translation_total_ms: float = 0.0
    tts_first_token_ms: float = 0.0
    codec_latency_ms: float = 0.0
    ttfa_ms: float = 0.0  # Time To First Audio
    e2e_latency_ms: float = 0.0
    tts_total_ms: float = 0.0
    # Resources
    audio_duration_sec: float = 0.0
    rtf: float = 0.0
    peak_ram_mb: float = 0.0
    current_ram_mb: float = 0.0
    cpu_percent: float = 0.0
    token_chunk_size: int = 32
    audio_chunks_count: int = 0
    underruns: int = 0
    codec_failures: int = 0
