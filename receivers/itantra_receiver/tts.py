"""
iTantra Receiver Pipeline - Offline Text-to-Speech Module (tts.py)
==================================================================
Synthesizes speech from translated text using:
1. AI4Bharat Indic-TTS ONNX graphs (FastPitch/VITS + HiFi-GAN) with CPUExecutionProvider.
2. Windows Native Offline Speech Engine (SAPI / SpVoice) for crystal-clear natural speech synthesis.
3. Safe mono float32 audio normalization, anti-clipping safeguards, and lossless WAV export.

Architecture Rules:
- CPUExecutionProvider ONLY (No CUDA, No GPU).
- Pure offline execution (zero cloud calls, zero network egress).
- Clean spoken human voice output (strictly NO robotic sine-wave beep tones).
- Mono float32 numeric contract at explicit sample rate (22050 Hz).
"""

import os
import subprocess
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf

try:
    import onnxruntime as ort
except ImportError:
    ort = None

import config


# ==============================================================================
# CUSTOM EXCEPTIONS
# ==============================================================================
class TTSError(Exception):
    """Base exception for TTS pipeline failures."""
    pass


class ModelNotFoundError(TTSError):
    """Raised when a local TTS model or vocoder is missing."""
    pass


class UnsupportedLanguageError(TTSError):
    """Raised when an unsupported TTS language code is requested."""
    pass


class InvalidInputError(TTSError):
    """Raised when invalid or empty text is provided for synthesis."""
    pass


# ==============================================================================
# TTS MODEL ADAPTER (ABSTRACT INTERFACE)
# ==============================================================================
class TTSModelAdapter(ABC):
    """
    Abstract adapter isolating model-specific tensor operations, acoustic synthesis,
    and vocoding from the core pipeline logic.
    """

    @abstractmethod
    def prepare_inputs(self, text: str, language_code: str) -> dict[str, Any]:
        """Convert input text and language code into model input tensors."""
        pass

    @abstractmethod
    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute ONNX or offline synthesis inference."""
        pass

    @abstractmethod
    def postprocess(self, outputs: dict[str, Any]) -> tuple[np.ndarray, int]:
        """Convert raw tensor/audio outputs to normalized mono float32 audio and sample rate."""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if model sessions and assets are loaded."""
        pass


# ==============================================================================
# INDIC-TTS ONNX ADAPTER (FASTPITCH / VITS / HIFIGAN)
# ==============================================================================
class IndicTTSFastPitchVitsONNXAdapter(TTSModelAdapter):
    """
    ONNX Runtime adapter for AI4Bharat Indic-TTS models.
    Supports FastPitch acoustic + HiFi-GAN vocoder or end-to-end VITS graphs on CPU.
    """

    def __init__(self, model_dir: Path | str, num_threads: int = config.TTS_NUM_THREADS):
        self.model_dir = Path(model_dir)
        self.num_threads = num_threads
        self.acoustic_session: Optional[Any] = None
        self.vocoder_session: Optional[Any] = None
        self.sample_rate = config.TTS_SAMPLE_RATE
        self._load_sessions()

    def _load_sessions(self):
        if ort is None:
            raise ModelNotFoundError("onnxruntime is not installed in the current environment.")

        if not self.model_dir.exists():
            raise ModelNotFoundError(
                f"Required local TTS model directory not found: {self.model_dir}\n"
                f"Please place Indic-TTS ONNX model files locally in {self.model_dir}"
            )

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.num_threads
        sess_options.inter_op_num_threads = self.num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        unified_path = self.model_dir / "model.onnx"
        acoustic_path = self.model_dir / "acoustic.onnx"
        vocoder_path = self.model_dir / "vocoder.onnx"

        providers = config.ONNX_PROVIDERS

        if unified_path.exists():
            self.acoustic_session = ort.InferenceSession(str(unified_path), sess_options=sess_options, providers=providers)
        elif acoustic_path.exists():
            self.acoustic_session = ort.InferenceSession(str(acoustic_path), sess_options=sess_options, providers=providers)
            if vocoder_path.exists():
                self.vocoder_session = ort.InferenceSession(str(vocoder_path), sess_options=sess_options, providers=providers)
        else:
            raise ModelNotFoundError(
                f"No valid TTS ONNX files found in {self.model_dir}.\n"
                f"Expected 'model.onnx' or 'acoustic.onnx'."
            )

    def is_ready(self) -> bool:
        return self.acoustic_session is not None

    def prepare_inputs(self, text: str, language_code: str) -> dict[str, Any]:
        tokens = [ord(c) % 256 for c in text[:config.TTS_MAX_TEXT_LENGTH]]
        text_ids = np.array([tokens], dtype=np.int64)
        text_lengths = np.array([len(tokens)], dtype=np.int64)
        speaker_id = np.array([config.TTS_DEFAULT_SPEAKER_ID], dtype=np.int64)
        pace = np.array([config.TTS_SPEED], dtype=np.float32)

        return {
            "text_ids": text_ids,
            "text_lengths": text_lengths,
            "speaker_id": speaker_id,
            "pace": pace,
            "raw_text": text,
        }

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not self.acoustic_session:
            raise ModelNotFoundError("TTS model session is not initialized.")

        feed_dict = {}
        input_names = [inp.name for inp in self.acoustic_session.get_inputs()]
        for key in ["text_ids", "text_lengths", "speaker_id", "pace"]:
            if key in input_names:
                feed_dict[key] = inputs[key]

        acoustic_outs = self.acoustic_session.run(None, feed_dict)

        if self.vocoder_session is not None:
            mel = acoustic_outs[0]
            vocoder_inputs = {self.vocoder_session.get_inputs()[0].name: mel}
            raw_audio = self.vocoder_session.run(None, vocoder_inputs)[0]
        else:
            raw_audio = acoustic_outs[0]

        return {"raw_audio": raw_audio}

    def postprocess(self, outputs: dict[str, Any]) -> tuple[np.ndarray, int]:
        raw_audio = outputs["raw_audio"]
        audio_flat = np.squeeze(np.asarray(raw_audio, dtype=np.float32))

        max_val = np.max(np.abs(audio_flat))
        if max_val > 0.0:
            audio_norm = audio_flat / max_val * 0.95
        else:
            audio_norm = audio_flat

        return audio_norm, self.sample_rate


# ==============================================================================
# WINDOWS NATIVE OFFLINE SAPI SPEECH ADAPTER (NATURAL SPOKEN VOICE)
# ==============================================================================
class WindowsSAPITTSAdapter(TTSModelAdapter):
    """
    High-quality offline speech synthesizer using native Windows SAPI Speech API.
    Produces natural, audible human speech audio in WAV format with zero network calls.
    """

    def __init__(self, sample_rate: int = config.TTS_SAMPLE_RATE):
        self.sample_rate = sample_rate

    def is_ready(self) -> bool:
        return True

    def prepare_inputs(self, text: str, language_code: str) -> dict[str, Any]:
        clean_text = text.replace('"', '').replace("'", "").strip()
        return {
            "text": clean_text,
            "language_code": language_code,
        }

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        text = inputs["text"]
        temp_dir = config.TEMP_AUDIO_DIR
        temp_dir.mkdir(parents=True, exist_ok=True)
        unique_id = f"sapi_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
        temp_wav = temp_dir / f"{unique_id}.wav"
        temp_vbs = temp_dir / f"{unique_id}.vbs"

        # Construct SAPI VBScript for zero-dependency offline speech file export
        escaped_text = text.replace('"', '""')
        escaped_wav_path = str(temp_wav.resolve()).replace('\\', '\\\\')

        vbs_code = (
            f'Dim Sapi, FileStream\n'
            f'Set Sapi = CreateObject("SAPI.SpVoice")\n'
            f'Set FileStream = CreateObject("SAPI.SpFileStream")\n'
            f'FileStream.Open "{escaped_wav_path}", 3, False\n'
            f'Set Sapi.AudioOutputStream = FileStream\n'
            f'Sapi.Rate = 0\n'
            f'Sapi.Volume = 100\n'
            f'Sapi.Speak "{escaped_text}"\n'
            f'FileStream.Close\n'
        )

        temp_vbs.write_text(vbs_code, encoding="utf-8")

        try:
            # Run CScript with 10s timeout
            subprocess.run(
                ["cscript", "//Nologo", str(temp_vbs)],
                check=True,
                capture_output=True,
                timeout=10,
            )

            if temp_wav.exists():
                audio, sr = sf.read(str(temp_wav), dtype="float32")
                # Clean up temporary VBS script
                if temp_vbs.exists():
                    temp_vbs.unlink(missing_ok=True)
                if temp_wav.exists():
                    temp_wav.unlink(missing_ok=True)

                # Ensure mono
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)

                return {"audio_array": audio, "sample_rate": sr}
            else:
                raise TTSError(f"SAPI voice output file {temp_wav} was not created.")

        except Exception as e:
            # Clean up temp files
            if temp_vbs.exists():
                temp_vbs.unlink(missing_ok=True)
            if temp_wav.exists():
                temp_wav.unlink(missing_ok=True)
            raise TTSError(f"Failed to generate speech via Windows SAPI: {e}")

    def postprocess(self, outputs: dict[str, Any]) -> tuple[np.ndarray, int]:
        audio = np.asarray(outputs["audio_array"], dtype=np.float32)
        sr = outputs.get("sample_rate", self.sample_rate)

        # Normalize safely
        max_val = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
        if max_val > 0.0:
            audio = (audio / max_val * 0.95).astype(np.float32)

        return audio, sr


# Alias for backward compatibility with existing tests and scripts
MockTTSAdapter = WindowsSAPITTSAdapter


# ==============================================================================
# OFFLINE TTS ENGINE CLASS
# ==============================================================================
class OfflineTTS:
    """
    High-level Offline Text-to-Speech Engine for the iTantra Receiver Pipeline.
    Manages text preprocessing, inference timing, postprocessing, audio writing,
    and anti-clipping validation.
    """

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        language_code: str = "en",
        adapter: Optional[TTSModelAdapter] = None,
    ):
        self.language_code = language_code
        self.set_language(language_code)

        if adapter is not None:
            self.adapter = adapter
        elif model_path is not None and Path(model_path).exists():
            self.adapter = IndicTTSFastPitchVitsONNXAdapter(model_path)
        else:
            lang_dir = config.TTS_MODELS_DIR / language_code
            if lang_dir.exists() and (lang_dir / "model.onnx").exists():
                self.adapter = IndicTTSFastPitchVitsONNXAdapter(lang_dir)
            else:
                # Use natural Windows SAPI offline speech adapter
                self.adapter = WindowsSAPITTSAdapter(sample_rate=config.TTS_SAMPLE_RATE)

        self._last_timing: dict[str, float] = {}

    @property
    def sample_rate(self) -> int:
        return config.TTS_SAMPLE_RATE

    def set_language(self, language_code: str):
        """Set and validate TTS synthesis language."""
        if language_code not in config.SUPPORTED_LANGUAGES:
            raise UnsupportedLanguageError(
                f"TTS Language '{language_code}' is not supported. "
                f"Supported: {config.SUPPORTED_LANGUAGES}"
            )
        self.language_code = language_code

    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesizes speech from input text.
        Returns mono float32 numpy array normalized and checked against clipping.
        """
        if not isinstance(text, str) or not text.strip():
            raise InvalidInputError("Text for TTS synthesis must be a non-empty string.")

        clean_text = text.strip()
        if len(clean_text) > config.TTS_MAX_TEXT_LENGTH:
            clean_text = clean_text[:config.TTS_MAX_TEXT_LENGTH]

        # 1. Preprocessing stage
        t0 = time.perf_counter()
        inputs = self.adapter.prepare_inputs(clean_text, self.language_code)
        t1 = time.perf_counter()
        preprocess_ms = (t1 - t0) * 1000.0

        # 2. Model inference stage
        outputs = self.adapter.infer(inputs)
        t2 = time.perf_counter()
        inference_ms = (t2 - t1) * 1000.0

        # 3. Postprocessing stage
        audio, sample_rate = self.adapter.postprocess(outputs)
        t3 = time.perf_counter()
        postprocess_ms = (t3 - t2) * 1000.0
        total_tts_ms = (t3 - t0) * 1000.0

        # Audio safety assertions
        if not np.isfinite(audio).all():
            raise TTSError("Synthesized audio contains non-finite values (NaN or Inf).")

        # Anti-clipping verification & safety normalization
        max_amplitude = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
        if max_amplitude > 1.0:
            audio = (audio / max_amplitude * 0.95).astype(np.float32)

        self._last_timing = {
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "postprocess_ms": postprocess_ms,
            "total_ms": total_tts_ms,
        }

        # Print latency breakdown to console
        print(
            f"[TTS] inference: {inference_ms:.2f} ms | "
            f"postprocess: {postprocess_ms:.2f} ms | "
            f"total: {total_tts_ms:.2f} ms"
        )

        return audio

    def get_last_timing(self) -> dict[str, float]:
        """Return the detailed latency breakdown of the most recent synthesis."""
        return dict(self._last_timing)

    def save_audio(self, audio: np.ndarray, output_path: str | Path) -> tuple[str, float]:
        """
        Saves float32 audio to a standard uncompressed WAV file.
        Returns (saved_path_str, write_latency_ms).
        """
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        sf.write(str(target_path), audio, self.sample_rate, subtype="PCM_16")
        t1 = time.perf_counter()
        write_ms = (t1 - t0) * 1000.0

        return str(target_path), write_ms
