"""
iTantra Voice Pipeline Configuration & Hyperparameters
======================================================
ALL hyperparameters live here as named constants for transparent tuning.
No hyperparameter should ever be hardcoded inline.
"""

import os
from pathlib import Path

# Base Directory Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
KEYWORDS_DIR = BASE_DIR / "keywords"

# Ensure essential directories exist
os.makedirs(MODELS_DIR / "stt", exist_ok=True)
os.makedirs(KEYWORDS_DIR, exist_ok=True)

# Audio Defaults
DEFAULT_SAMPLE_RATE = 16000  # Pipeline expects 16kHz mono audio

# -----------------------------------------------------------------------------
# MODULE 1: Silero VAD Hyperparameters
# -----------------------------------------------------------------------------
VAD_MODEL_PATH = MODELS_DIR / "silero_vad.onnx"
VAD_SAMPLE_RATE = 16000      # Silero VAD operates at 16kHz
VAD_WINDOW_SAMPLES = 512     # 512 samples = 32ms frame size at 16kHz

VAD_THRESHOLD = 0.5          # Speech probability threshold (0.0 to 1.0)
VAD_MIN_SPEECH_MS = 250      # Ignore speech bursts shorter than this (ms)
VAD_MIN_SILENCE_MS = 200     # Silence duration (ms) before declaring speech-end
VAD_SPEECH_PAD_MS = 30       # Pad detected speech boundaries (ms) on both sides

# -----------------------------------------------------------------------------
# MODULE 2: Streaming STT Hyperparameters
# -----------------------------------------------------------------------------
STT_MODEL_DIR = MODELS_DIR / "stt"
STT_SAMPLE_RATE = 16000      # STT input audio sample rate
STT_CHUNK_MS = 100           # Simulates real streaming: 100ms chunks fed to STT
STT_DECODING = "greedy"      # Default decoding mode ("greedy" or "beam_search")
STT_BEAM_SIZE = 1            # Beam size toggle (1 = greedy, 4, 8)

# Supported 10 Indic / target languages
STT_LANGUAGES = [
    "hi",  # Hindi
    "gu",  # Gujarati
    "mr",  # Marathi
    "kn",  # Kannada
    "ml",  # Malayalam
    "ta",  # Tamil
    "te",  # Telugu
    "or",  # Odia
    "bn",  # Bengali
    "en",  # English
]

# Language display labels for UI dropdown
LANGUAGE_NAMES = {
    "hi": "Hindi (hi)",
    "gu": "Gujarati (gu)",
    "mr": "Marathi (mr)",
    "kn": "Kannada (kn)",
    "ml": "Malayalam (ml)",
    "ta": "Tamil (ta)",
    "te": "Telugu (te)",
    "or": "Odia (or)",
    "bn": "Bengali (bn)",
    "en": "English (en)",
}

# -----------------------------------------------------------------------------
# MODULE 3: Emergency Classifier Hyperparameters
# -----------------------------------------------------------------------------
DEFAULT_LANGUAGE = "hi"
PRIORITY_EMERGENCY = "P0"
PRIORITY_NORMAL = "P2"

# -----------------------------------------------------------------------------
# MODULE 4: Wake-Word / Voice-Trigger Hyperparameters (sherpa-onnx KWS)
# -----------------------------------------------------------------------------
KWS_MODEL_DIR = MODELS_DIR / "kws" / "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
WAKE_WORD_KEYWORDS_PATH = KEYWORDS_DIR / "kws_keywords.txt"

WAKE_WORD_THRESHOLD = 0.10       # KWS trigger probability threshold (0.0 to 1.0)
WAKE_WORD_SCORE = 2.0           # Boosting score of keyword tokens in beam search
WAKE_WORD_COOLDOWN_MS = 1500     # Debounce period (ms) to prevent duplicate triggering
WAKE_WORD_PRE_ROLL_MS = 500      # Ring buffer lookback (ms) preserved on wake trigger
COMMAND_TIMEOUT_MS = 5000        # Max audio duration (ms) for command speech capture
COMMAND_SILENCE_TIMEOUT_MS = 1200 # Silence duration after speech to end command capture

# Known wake phrases and phonetic transcripts for command prefix stripping
WAKE_PHRASES = [
    "hey itantra",
    "hey i tantra",
    "i tantra",
    "itantra",
    "hey siri",
    "alexa",
    "hello world",
    "start itantra",
    "हे आईटन्ट्रा",
    "हे आईतंत्रा",
    "आईटन्ट्रा",
    "आईतंत्रा",
    "ए आई टैन ट्र",
    "ए आईटैन्ट्रा",
    "ए आई टंट्रा",
    "ए आईटन्ट्रा",
    "ए आई टेंट्रा",
    "ए आई",
    "hey i",
    "hey",
]

