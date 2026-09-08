"""
iTantra Receiver Pipeline - Global Configuration Module (config.py)
===================================================================
ALL tunable hyperparameters, constants, supported languages, model paths,
and audio specifications are centralized here.

Architecture Rule:
- Never bury tunable values inside other modules.
- Strict CPU-only execution (CPUExecutionProvider).
- Zero network dependencies.
"""

from pathlib import Path

# ==============================================================================
# BASE DIRECTORY PATHS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
TRANSLATION_MODELS_DIR = MODELS_DIR / "translation"
TTS_MODELS_DIR = MODELS_DIR / "tts"
TEST_AUDIO_DIR = BASE_DIR / "test_audio"
REPORTS_DIR = BASE_DIR / "reports"
TEMP_AUDIO_DIR = BASE_DIR / "temp_audio"

# Ensure directories exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TRANSLATION_MODELS_DIR.mkdir(parents=True, exist_ok=True)
TTS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
TEST_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# ONNX RUNTIME EXECUTION PROVIDERS (STRICTLY CPU ONLY)
# ==============================================================================
# Must never contain CUDAExecutionProvider, TensorrtExecutionProvider, DirectML, CoreML.
ONNX_PROVIDERS = ["CPUExecutionProvider"]

# ==============================================================================
# TRANSLATION HYPERPARAMETERS (IndicTrans2 ONNX)
# ==============================================================================
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

# Maximum character count per segment before chunking long transcripts
TRANSLATION_MAX_CHARS_PER_CHUNK: int = 250
TRANSLATION_CHUNK_LONG_TEXT: bool = True

# Behavior when source_language == target_language:
# "pass_through" -> returns input text directly with 0.0ms inference
# "error" -> raises UnsupportedLanguageError / InvalidInputError
TRANSLATION_SAME_LANG_BEHAVIOR: str = "pass_through"

# ==============================================================================
# TTS HYPERPARAMETERS (Indic-TTS / FastPitch / VITS ONNX)
# ==============================================================================
# Synthesis sample rate in Hz (e.g., 22050 Hz for Indic-TTS FastPitch / HiFi-GAN)
TTS_SAMPLE_RATE: int = 22050

# Speech rate multiplier (1.0 is normal speed)
TTS_SPEED: float = 1.0

# Output volume multiplier
TTS_VOLUME: float = 1.0

# Maximum character length accepted in a single TTS synthesis chunk
TTS_MAX_TEXT_LENGTH: int = 500

# ONNX Runtime thread count for CPU TTS inference
TTS_NUM_THREADS: int = 4

# Speaker ID for multi-speaker Indic-TTS models (default female=0, male=1 or language-specific)
TTS_DEFAULT_SPEAKER_ID: int = 0

# ==============================================================================
# AUDIO SPECIFICATIONS
# ==============================================================================
INPUT_SAMPLE_RATE: int = 16000     # Expected sender transcript audio reference rate
OUTPUT_SAMPLE_RATE: int = 22050    # Synthesized receiver audio output sample rate
AUDIO_CHANNELS: int = 1            # Strict mono audio
AUDIO_DTYPE: str = "float32"       # Mono float32 numeric representation
OUTPUT_AUDIO_FORMAT: str = "wav"   # Lossless uncompressed PCM WAV container

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
