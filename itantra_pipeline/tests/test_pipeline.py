"""
Standalone test for pipeline.py (VoicePipeline)
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pipeline import VoicePipeline

def test_voice_pipeline():
    print("--- Running Test: VoicePipeline ---")
    pipeline = VoicePipeline()

    sr = 16000
    # Generate 1 sec silence, 1.5 sec sine wave, 1 sec silence
    silence1 = np.zeros(sr, dtype=np.float32)
    t = np.linspace(0, 1.5, int(1.5 * sr), endpoint=False)
    tone = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    silence2 = np.zeros(sr, dtype=np.float32)

    audio = np.concatenate([silence1, tone, silence2])

    result = pipeline.process_audio((sr, audio), language_code="hi")

    print("Pipeline Execution Result Keys:", list(result.keys()))
    print(f"VAD Latency: {result['vad_latency_ms']} ms")
    print(f"STT Latency: {result['stt_latency_ms']} ms")
    print(f"Classifier Latency: {result['classifier_latency_ms']} ms")
    print(f"Total Latency: {result['total_latency_ms']} ms")
    print(f"Final Transcript: '{result['final_transcript']}'")
    print(f"Emergency Flag: {result['emergency_result']}")

    assert result["total_latency_ms"] > 0, "Latency measurement should be non-zero!"
    assert "vad_segments" in result
    assert "emergency_result" in result

    print("--- Test VoicePipeline PASSED ---\n")

if __name__ == "__main__":
    test_voice_pipeline()
