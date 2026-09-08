"""
iTantra Receiver Pipeline - Global Configuration Module (config.py)
===================================================================
ALL tunable hyperparameters, constants, supported languages, model paths,
voice/accent specifications, thread configurations, and leakage thresholds are centralized here.

Architecture Rules:
- Never bury tunable values inside other modules.
- Strict CPU-only execution (CPUExecutionProvider).
- Zero network dependencies.
"""

from pathlib import Path
from typing import Any

# ==============================================================================
# BASE DIRECTORY PATHS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
TRANSLATION_MODELS_DIR = MODELS_DIR / "translation"
TTS_MODELS_DIR = MODELS_DIR / "tts"
LEXICONS_DIR = BASE_DIR / "lexicons"
TEST_AUDIO_DIR = BASE_DIR / "test_audio"
REPORTS_DIR = BASE_DIR / "reports"
TEMP_AUDIO_DIR = BASE_DIR / "temp_audio"

# Ensure directories exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TRANSLATION_MODELS_DIR.mkdir(parents=True, exist_ok=True)
TTS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
LEXICONS_DIR.mkdir(parents=True, exist_ok=True)
TEST_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# ONNX RUNTIME EXECUTION PROVIDERS & THREADS (STRICTLY CPU ONLY)
# ==============================================================================
# Must never contain CUDAExecutionProvider, TensorrtExecutionProvider, DirectML, CoreML.
ONNX_PROVIDERS = ["CPUExecutionProvider"]
ONNX_INTRA_OP_THREADS: int = 1
ONNX_INTER_OP_THREADS: int = 1

# ==============================================================================
# TRANSLATION HYPERPARAMETERS & POLICIES
# ==============================================================================
# Section 3: High-Priority Translation Fix
# Translate only finalized STT utterances; partial STT hypotheses are display-only.
TRANSLATE_ONLY_FINAL_TEXT: bool = True

# Section 12: Minimum text length in characters for translation
MIN_TRANSLATION_CONTEXT_CHARS: int = 2

# Section 8: Bounded Context Translation
TRANSLATION_USE_CONTEXT: bool = True
TRANSLATION_CONTEXT_SENTENCE_COUNT: int = 2

# Default beam size for decoding:
# 1 = Greedy decoding (lowest latency, deterministic)
# 4 = Medium quality beam search
# 8 = High quality beam search
TRANSLATION_BEAM_SIZE: int = 1
TRANSLATION_SUPPORTED_BEAM_SIZES: list[int] = [1, 4, 8]

# Maximum sequence length tokens
TRANSLATION_MAX_INPUT_TOKENS: int = 256
TRANSLATION_MAX_OUTPUT_TOKENS: int = 256

# Sampling parameters (when beam_size=1, greedy is used)
TRANSLATION_TEMPERATURE: float = 1.0
TRANSLATION_REPETITION_PENALTY: float = 1.0

# ONNX Runtime thread count for CPU translation inference
TRANSLATION_NUM_THREADS: int = 4

# Maximum character count per segment before safe sentence chunking
TRANSLATION_MAX_CHARS_PER_CHUNK: int = 250
TRANSLATION_CHUNK_LONG_TEXT: bool = True

# Behavior when source_language == target_language:
# "pass_through" -> returns input text directly with 0.0ms inference
# "error" -> raises UnsupportedLanguageError / InvalidInputError
TRANSLATION_SAME_LANG_BEHAVIOR: str = "pass_through"

# ==============================================================================
# SOURCE LEAKAGE DETECTION THRESHOLDS & SCRIPT CONFIG (Section 5, 8, 33)
# ==============================================================================
ENABLE_SOURCE_LEAKAGE_CHECK: bool = True
REJECT_SOURCE_LEAKAGE: bool = True
ALLOW_SAME_LANGUAGE_PASSTHROUGH: bool = True

# Minimum similarity ratio to classify as exact match
SOURCE_LEAKAGE_EXACT_MATCH_THRESHOLD: float = 0.95

# Token Jaccard overlap threshold above which leakage is strongly suspected
SOURCE_LEAKAGE_TOKEN_OVERLAP_THRESHOLD: float = 0.80

# Minimum ratio of source script characters to suspect source script dominance
SOURCE_SCRIPT_MIN_RATIO: float = 0.40

# Minimum ratio of target script characters for target script conformity
TARGET_SCRIPT_MIN_RATIO: float = 0.50

# Language to primary script block mapping
LANGUAGE_SCRIPTS: dict[str, str] = {
    "hi": "Devanagari",
    "mr": "Devanagari",
    "bn": "Bengali",
    "gu": "Gujarati",
    "or": "Odia",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "en": "Latin",
}

# ==============================================================================
# TTS HYPERPARAMETERS & PROSODY (Indic-TTS / FastPitch / VITS / OneCore)
# ==============================================================================
# Synthesis sample rate in Hz (e.g., 22050 Hz for Indic-TTS FastPitch / HiFi-GAN)
TTS_SAMPLE_RATE: int = 22050

# Speech rate multiplier (1.0 is normal speed)
TTS_SPEED: float = 1.0

# Prosody pause controls (in milliseconds)
TTS_PAUSE_MS: int = 150
TTS_SENTENCE_PAUSE_MS: int = 300

# Output volume multiplier (0.0 to 1.0)
TTS_VOLUME: float = 1.0

# Maximum character length accepted in a single TTS synthesis chunk
TTS_MAX_TEXT_LENGTH: int = 500

# ONNX Runtime thread count for CPU TTS inference
TTS_NUM_THREADS: int = 4

# Speaker ID for multi-speaker Indic-TTS models (default female=0, male=1 or language-specific)
TTS_DEFAULT_SPEAKER_ID: int = 0

# ==============================================================================
# AUDIO SPECIFICATIONS & QUALITY CONSTRAINTS
# ==============================================================================
INPUT_SAMPLE_RATE: int = 16000     # Expected sender transcript audio reference rate
OUTPUT_SAMPLE_RATE: int = 22050    # Synthesized receiver audio output sample rate
AUDIO_CHANNELS: int = 1            # Strict mono audio
AUDIO_DTYPE: str = "float32"       # Mono float32 numeric representation
OUTPUT_AUDIO_FORMAT: str = "wav"   # Lossless uncompressed PCM WAV container
MAX_ALLOWED_AMPLITUDE: float = 1.0 # Amplitude bound before normalization
CLIPPING_THRESHOLD: float = 0.999  # Absolute amplitude threshold to count clipping samples

# ==============================================================================
# BENCHMARKING PARAMETERS
# ==============================================================================
# Warmup runs to stabilize CPU caches and ONNX session execution before measurement
BENCHMARK_WARMUP_RUNS: int = 1

# Number of measured runs to compute statistical aggregates (min, max, mean, median, P95, std)
BENCHMARK_MEASURED_RUNS: int = 5

# Distinguish cold-start vs warm-start benchmarking
BENCHMARK_INCLUDE_MODEL_LOAD: bool = False
BENCHMARK_INCLUDE_AUDIO_IO: bool = True

# ==============================================================================
# SUPPORTED LANGUAGES
# ==============================================================================
# Primary 10 languages supported by iTantra (Indic + English)
SUPPORTED_LANGUAGES: list[str] = [
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

# Human-readable display mapping
LANGUAGE_NAMES: dict[str, str] = {
    "hi": "Hindi",
    "gu": "Gujarati",
    "mr": "Marathi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "ta": "Tamil",
    "te": "Telugu",
    "or": "Odia",
    "bn": "Bengali",
    "en": "English",
}

# IndicTrans2 Language Tag Mapping (Format: <lang>_Deva, <lang>_Gujr, etc.)
INDICTRANS2_LANG_TAGS: dict[str, str] = {
    "hi": "hin_Deva",
    "gu": "guj_Gujr",
    "mr": "mar_Deva",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "or": "ory_Orya",
    "bn": "ben_Beng",
    "en": "eng_Latn",
}

# ==============================================================================
# REAL LOCAL VOICE & ACCENT CONFIGURATIONS
# ==============================================================================
# Decouples language, voice_id, accent_id, speaker_id.
# Only populated from local voices and verified models actually installed.
VOICE_CONFIGS: dict[str, list[dict[str, Any]]] = {
    "en": [
        {
            "voice_id": "en_in_heera",
            "name": "Microsoft Heera (en-IN)",
            "accent_id": "indian_english",
            "accent_name": "Indian English",
            "speaker_id": "female_01",
            "token": "MSTTS_V110_enIN_HeeraM",
            "sample_rate": 22050,
            "accent_supported": True,
        },
        {
            "voice_id": "en_in_ravi",
            "name": "Microsoft Ravi (en-IN)",
            "accent_id": "indian_english",
            "accent_name": "Indian English",
            "speaker_id": "male_01",
            "token": "MSTTS_V110_enIN_RaviM",
            "sample_rate": 22050,
            "accent_supported": True,
        },
        {
            "voice_id": "en_us_zira",
            "name": "Microsoft Zira (en-US)",
            "accent_id": "us_english",
            "accent_name": "US English",
            "speaker_id": "female_02",
            "token": "MSTTS_V110_enUS_ZiraM",
            "sample_rate": 22050,
            "accent_supported": True,
        },
        {
            "voice_id": "en_us_david",
            "name": "Microsoft David (en-US)",
            "accent_id": "us_english",
            "accent_name": "US English",
            "speaker_id": "male_02",
            "token": "MSTTS_V110_enUS_DavidM",
            "sample_rate": 22050,
            "accent_supported": True,
        },
    ],
    "hi": [
        {
            "voice_id": "hi_in_heera",
            "name": "Indian Voice Phonetic (Heera)",
            "accent_id": "indian_native",
            "accent_name": "Indian Native",
            "speaker_id": "female_01",
            "token": "MSTTS_V110_enIN_HeeraM",
            "sample_rate": 22050,
            "accent_supported": True,
        },
        {
            "voice_id": "hi_in_ravi",
            "name": "Indian Voice Phonetic (Ravi)",
            "accent_id": "indian_native",
            "accent_name": "Indian Native",
            "speaker_id": "male_01",
            "token": "MSTTS_V110_enIN_RaviM",
            "sample_rate": 22050,
            "accent_supported": True,
        },
    ],
    "gu": [
        {
            "voice_id": "gu_in_heera",
            "name": "Gujarati Phonetic Voice (Heera)",
            "accent_id": "indian_native",
            "accent_name": "Indian Native",
            "speaker_id": "female_01",
            "token": "MSTTS_V110_enIN_HeeraM",
            "sample_rate": 22050,
            "accent_supported": True,
        }
    ],
    "mr": [
        {
            "voice_id": "mr_in_heera",
            "name": "Marathi Phonetic Voice (Heera)",
            "accent_id": "indian_native",
            "accent_name": "Indian Native",
            "speaker_id": "female_01",
            "token": "MSTTS_V110_enIN_HeeraM",
            "sample_rate": 22050,
            "accent_supported": True,
        }
    ],
    "ta": [
        {
            "voice_id": "ta_in_heera",
            "name": "Tamil Phonetic Voice (Heera)",
            "accent_id": "indian_native",
            "accent_name": "Indian Native",
            "speaker_id": "female_01",
            "token": "MSTTS_V110_enIN_HeeraM",
            "sample_rate": 22050,
            "accent_supported": True,
        }
    ],
    "te": [
        {
            "voice_id": "te_in_heera",
            "name": "Telugu Phonetic Voice (Heera)",
            "accent_id": "indian_native",
            "accent_name": "Indian Native",
            "speaker_id": "female_01",
            "token": "MSTTS_V110_enIN_HeeraM",
            "sample_rate": 22050,
            "accent_supported": True,
        }
    ],
    "kn": [
        {
            "voice_id": "kn_in_heera",
            "name": "Kannada Phonetic Voice (Heera)",
            "accent_id": "indian_native",
            "accent_name": "Indian Native",
            "speaker_id": "female_01",
            "token": "MSTTS_V110_enIN_HeeraM",
            "sample_rate": 22050,
            "accent_supported": True,
        }
    ],
    "ml": [
        {
            "voice_id": "ml_in_heera",
            "name": "Malayalam Phonetic Voice (Heera)",
            "accent_id": "indian_native",
            "accent_name": "Indian Native",
            "speaker_id": "female_01",
            "token": "MSTTS_V110_enIN_HeeraM",
            "sample_rate": 22050,
            "accent_supported": True,
        }
    ],
    "bn": [
        {
            "voice_id": "bn_in_heera",
            "name": "Bengali Phonetic Voice (Heera)",
            "accent_id": "indian_native",
            "accent_name": "Indian Native",
            "speaker_id": "female_01",
            "token": "MSTTS_V110_enIN_HeeraM",
            "sample_rate": 22050,
            "accent_supported": True,
        }
    ],
    "or": [
        {
            "voice_id": "or_in_heera",
            "name": "Odia Phonetic Voice (Heera)",
            "accent_id": "indian_native",
            "accent_name": "Indian Native",
            "speaker_id": "female_01",
            "token": "MSTTS_V110_enIN_HeeraM",
            "sample_rate": 22050,
            "accent_supported": True,
        }
    ],
}
