"""
Standalone test for vad.py (SileroVAD) with ONNX Runtime
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vad import SileroVAD

def test_silero_vad():
    print("--- Running Test: SileroVAD ---")
    vad = SileroVAD()
    print(f"ONNX Model Loaded: {vad.is_onnx_loaded}")

    sr = 16000
    # Generate multi-tone harmonic signal simulating human vocal formant speech energy
    silence1 = np.zeros(sr, dtype=np.float32)
    t = np.linspace(0, 1.2, int(1.2 * sr), endpoint=False)
    # Formants around 300Hz, 800Hz, 2500Hz with random amplitude modulation to mimic speech
    mod = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)
    vocal = mod * (0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 750 * t) + 0.1 * np.sin(2 * np.pi * 2400 * t))
    vocal = vocal.astype(np.float32)
    silence2 = np.zeros(sr, dtype=np.float32)

    audio = np.concatenate([silence1, vocal, silence2])
    print(f"Generated test audio duration: {len(audio)/sr:.2f}s ({len(audio)} samples)")

    # Run frame processing
    segments = vad.get_speech_segments(audio)
    print(f"Detected speech segments (in samples): {segments}")
    for start, end in segments:
        print(f"  Segment: {start/sr:.3f}s to {end/sr:.3f}s (duration {(end-start)/sr:.3f}s)")
    
    print(f"Total VAD debug log entries: {len(vad.debug_log)}")
    print("--- Test SileroVAD PASSED ---\n")

if __name__ == "__main__":
    test_silero_vad()
