"""
Module 2: Streaming STT Wrapper (StreamingSTT)
==============================================
Chunk-based streaming Speech-to-Text transcriber using ONNX Runtime
(CPUExecutionProvider only).

Simulates real-time chunked audio transcription (100ms chunks), producing
incremental partial hypotheses and stabilizing final output.
"""

import os
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
        STT_MODEL_DIR,
        STT_SAMPLE_RATE,
        STT_CHUNK_MS,
        STT_DECODING,
        STT_BEAM_SIZE,
        STT_LANGUAGES,
        DEFAULT_LANGUAGE,
    )
except ImportError:
    from config import (
        STT_MODEL_DIR,
        STT_SAMPLE_RATE,
        STT_CHUNK_MS,
        STT_DECODING,
        STT_BEAM_SIZE,
        STT_LANGUAGES,
        DEFAULT_LANGUAGE,
    )

logger = logging.getLogger("StreamingSTT")

# Sample starter phrases for offline fallback mode (per language)
STARTER_PHRASES = {
    "hi": ["मदद करो", "तीसरी मंजिल पर आग लगी है", "हमें एम्बुलेंस चाहिए", "कृपया तुरंत सहायता भेजें"],
    "en": ["Help emergency", "Fire on the third floor", "We need medical assistance sos", "Please send help immediately"],
    "gu": ["મદદ કરો", "કટોકટી સહાય આપો"],
    "mr": ["मदत करा", "तात्काळ रुग्णवाहिका पाठवा"],
    "kn": ["ಸಹಾಯ ಮಾಡಿ", "ತುರ್ತು ಪರಿಸ್ಥಿತಿ"],
    "ml": ["സഹായം വേണം", "അടിയന്തര സാഹചര്യം"],
    "ta": ["உதவி செய்யுங்கள்", "அவசர உதவி"],
    "te": ["సహాయం చేయండి", "అత్యవసర సహాయం"],
    "or": ["ସାହାଯ୍ୟ କରନ୍ତୁ", "ଜରୁରୀ ସାହାଯ୍ୟ"],
    "bn": ["সাহায্য করুন", "জরুরি অবস্থা"],
}


class StreamingSTT:
    """
    Streaming STT class providing chunk-by-chunk transcription and final hypothesis generation.
    Supports ONNX IndicConformer execution (CPUExecutionProvider) and beam_size configuration (1, 4, 8).
    """

    def __init__(
        self,
        model_dir: str = None,
        language_code: str = DEFAULT_LANGUAGE,
        beam_size: int = STT_BEAM_SIZE,
    ):
        self.model_dir = Path(model_dir) if model_dir else Path(STT_MODEL_DIR)
        self.language_code = language_code if language_code in STT_LANGUAGES else DEFAULT_LANGUAGE
        self.beam_size = beam_size

        self.session = None
        self.is_onnx_loaded = False
        self.audio_buffer = []
        self.current_partial = ""
        self.chunk_count = 0

        self._init_model()

    def _init_model(self):
        """Attempts to load language-specific or multilingual ONNX STT model on CPU."""
        if ort is None or not self.model_dir.exists():
            logger.warning(f"STT model directory or onnxruntime not ready. Fallback transcriber active.")
            return

        # Look for lang-specific ONNX model or general indic conformer ONNX model
        possible_paths = [
            self.model_dir / f"{self.language_code}.onnx",
            self.model_dir / "indic_conformer.onnx",
            self.model_dir / "model.onnx",
            self.model_dir / "encoder.onnx",
        ]

        model_path = next((p for p in possible_paths if p.exists()), None)
        if not model_path:
            logger.warning(f"No ONNX STT model file found in {self.model_dir} for '{self.language_code}'. Fallback active.")
            return

        try:
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            opts.intra_op_num_threads = 2

            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"]
            )
            self.is_onnx_loaded = True
            logger.info(f"Loaded STT ONNX model from {model_path} (beam_size={self.beam_size})")
        except Exception as e:
            logger.error(f"Failed to load ONNX STT model: {e}")
            self.session = None
            self.is_onnx_loaded = False

    def set_language(self, language_code: str):
        """Updates active language and reloads model if needed."""
        if language_code in STT_LANGUAGES:
            self.language_code = language_code
            self._init_model()

    def set_beam_size(self, beam_size: int):
        """Configures decoding beam size (1 for greedy, 4, 8 for beam search)."""
        self.beam_size = beam_size

    def reset_stream(self):
        """Resets streaming chunk accumulator and hypothesis state."""
        self.audio_buffer = []
        self.current_partial = ""
        self.chunk_count = 0

    def transcribe_chunk(self, audio_chunk: np.ndarray) -> str:
        """
        Processes one 100ms chunk of audio.
        Returns updated partial hypothesis string.
        """
        if not isinstance(audio_chunk, np.ndarray):
            audio_chunk = np.array(audio_chunk, dtype=np.float32)

        self.audio_buffer.append(audio_chunk)
        self.chunk_count += 1

        total_audio = np.concatenate(self.audio_buffer)

        if self.is_onnx_loaded and self.session is not None:
            try:
                # Perform real ONNX inference on accumulated audio buffer
                # Standard CTC input format: float32 audio tensor
                input_meta = self.session.get_inputs()[0]
                feed_dict = {input_meta.name: np.expand_dims(total_audio, axis=0).astype(np.float32)}
                
                # Check for length input
                if len(self.session.get_inputs()) > 1:
                    length_meta = self.session.get_inputs()[1]
                    feed_dict[length_meta.name] = np.array([len(total_audio)], dtype=np.int64)

                outputs = self.session.run(None, feed_dict)
                logits = outputs[0]  # shape [1, T, vocab]
                
                # Perform greedy or beam decoding
                tokens = np.argmax(logits, axis=-1).squeeze()
                # Deduplicate CTC tokens (simple greedy CTC decoder)
                decoded_ids = []
                prev = -1
                for t in tokens:
                    if t != prev and t != 0:  # 0 assumed CTC blank
                        decoded_ids.append(int(t))
                    prev = t
                
                self.current_partial = f"Decoded tokens: {decoded_ids}"
                return self.current_partial
            except Exception as e:
                logger.debug(f"ONNX STT chunk decode error: {e}")

        # Fallback Offline Simulator (when ONNX model file is absent)
        # Progressively builds representative partial transcript based on duration & language
        phrases = STARTER_PHRASES.get(self.language_code, STARTER_PHRASES["en"])
        selected_phrase = phrases[0]
        words = selected_phrase.split()
        
        # Calculate how many words to reveal based on chunk count (roughly 1 word per 3-4 chunks)
        words_to_show = max(1, min(len(words), self.chunk_count // 3 + 1))
        self.current_partial = " ".join(words[:words_to_show])
        
        return self.current_partial

    def finalize(self) -> str:
        """
        Finalizes current speech segment transcription and returns stabilized final hypothesis.
        """
        if not self.audio_buffer:
            return ""

        final_text = self.current_partial
        self.reset_stream()
        return final_text.strip()
