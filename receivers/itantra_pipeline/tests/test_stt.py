"""
Standalone test for stt.py (StreamingSTT)
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from stt import StreamingSTT

def test_streaming_stt():
    print("--- Running Test: StreamingSTT ---")
    stt = StreamingSTT(language_code="hi", beam_size=1)
    
    # 100ms chunk at 16kHz = 1600 samples
    chunk_samples = 1600
    dummy_chunk = np.random.randn(chunk_samples).astype(np.float32)

    print("Feeding 5 streaming chunks (100ms each)...")
    for i in range(5):
        partial = stt.transcribe_chunk(dummy_chunk)
        print(f"  Chunk {i+1} Partial Hypothesis: '{partial}'")

    final_text = stt.finalize()
    print(f"Final Transcript: '{final_text}'")
    assert len(final_text) > 0, "Final transcript should not be empty!"
    print("--- Test StreamingSTT PASSED ---\n")

if __name__ == "__main__":
    test_streaming_stt()
