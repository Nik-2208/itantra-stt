"""
iTantra Receiver Web App - WebSocket Event Protocol (web_app/schemas/events.py)
==============================================================================
Typed events streamed over WebSocket to the browser client.
"""

from enum import Enum
from typing import Optional, Any, Dict
from pydantic import BaseModel, Field
import time

class EventType(str, Enum):
    TEXT_RECEIVED = "TEXT_RECEIVED"
    ORIGINAL_DISPLAYED = "ORIGINAL_DISPLAYED"
    EMERGENCY_DETECTED = "EMERGENCY_DETECTED"
    TRANSLATION_STARTED = "TRANSLATION_STARTED"
    TRANSLATION_CHUNK = "TRANSLATION_CHUNK"
    TRANSLATION_COMPLETE = "TRANSLATION_COMPLETE"
    TTS_STARTED = "TTS_STARTED"
    SPEECH_TOKEN_CHUNK = "SPEECH_TOKEN_CHUNK"
    AUDIO_CHUNK = "AUDIO_CHUNK"
    TTS_FIRST_AUDIO = "TTS_FIRST_AUDIO"
    TTS_COMPLETE = "TTS_COMPLETE"
    BENCHMARK_RESULT = "BENCHMARK_RESULT"
    ERROR = "ERROR"

class WebSocketEvent(BaseModel):
    event_type: EventType
    timestamp: float = Field(default_factory=time.time)
    message_id: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
