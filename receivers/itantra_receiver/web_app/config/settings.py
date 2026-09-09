"""
iTantra Receiver Web App - Configuration & Settings (web_app/config/settings.py)
================================================================================
Central settings and constants for the local receiver prototype.
Strict offline, CPU-first, low-latency streaming audio parameters.
"""

from enum import Enum
from pathlib import Path
from typing import List, Dict

BASE_DIR = Path(__file__).resolve().parent.parent.parent
WEB_APP_DIR = BASE_DIR / "web_app"
DATA_DIR = WEB_APP_DIR / "data"
LEXICONS_DIR = DATA_DIR / "emergency_lexicons"
TEMP_AUDIO_DIR = BASE_DIR / "temp_audio"
TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
LEXICONS_DIR.mkdir(parents=True, exist_ok=True)

class LanguageCode(str, Enum):
    HI = "HI"  # Hindi
    GU = "GU"  # Gujarati
    MR = "MR"  # Marathi
    KN = "KN"  # Kannada
    ML = "ML"  # Malayalam
    TA = "TA"  # Tamil
    TE = "TE"  # Telugu
    OR = "OR"  # Odia
    BN = "BN"  # Bengali
    EN = "EN"  # English

LANGUAGE_NAMES: Dict[LanguageCode, str] = {
    LanguageCode.HI: "Hindi",
    LanguageCode.GU: "Gujarati",
    LanguageCode.MR: "Marathi",
    LanguageCode.KN: "Kannada",
    LanguageCode.ML: "Malayalam",
    LanguageCode.TA: "Tamil",
    LanguageCode.TE: "Telugu",
    LanguageCode.OR: "Odia",
    LanguageCode.BN: "Bengali",
    LanguageCode.EN: "English",
}

# IndicTrans2 script and language tags
INDICTRANS2_TAGS: Dict[LanguageCode, str] = {
    LanguageCode.HI: "hin_Deva",
    LanguageCode.GU: "guj_Gujr",
    LanguageCode.MR: "mar_Deva",
    LanguageCode.KN: "kan_Knda",
    LanguageCode.ML: "mal_Mlym",
    LanguageCode.TA: "tam_Taml",
    LanguageCode.TE: "tel_Telu",
    LanguageCode.OR: "ory_Orya",
    LanguageCode.BN: "ben_Beng",
    LanguageCode.EN: "eng_Latn",
}

# Indic-Mio Language Prompt Tags
INDIC_MIO_TAGS: Dict[LanguageCode, str] = {
    LanguageCode.HI: "<|hindi|>",
    LanguageCode.GU: "<|gujarati|>",
    LanguageCode.MR: "<|marathi|>",
    LanguageCode.KN: "<|kannada|>",
    LanguageCode.ML: "<|malayalam|>",
    LanguageCode.TA: "<|tamil|>",
    LanguageCode.TE: "<|telugu|>",
    LanguageCode.OR: "<|odia|>",
    LanguageCode.BN: "<|bengali|>",
    LanguageCode.EN: "<|english|>",
}

class MessageType(str, Enum):
    NORMAL = "NORMAL"
    ALERT = "ALERT"
    SOS = "SOS"

# Audio Settings
SAMPLE_RATE = 22050  # 22050 Hz standard sample rate
CHANNELS = 1         # Mono
BYTES_PER_SAMPLE = 2 # 16-bit PCM

# Streaming chunk sizes for speech tokens to benchmark
DEFAULT_TOKEN_CHUNK_SIZE = 32
BENCHMARK_TOKEN_CHUNK_SIZES = [16, 24, 32, 48, 64]

# Model Inference Configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
IS_OFFLINE = True
