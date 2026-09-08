"""
Standalone test for app.py (Gradio UI Handler)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app import run_pipeline, create_ui

def test_app_ui():
    print("--- Running Test: Gradio App UI Handler ---")
    app = create_ui()
    assert app is not None, "Gradio app instance creation failed!"

    sample_audio_path = Path(__file__).resolve().parent.parent / "sample_emergency_hi.wav"
    print(f"Testing run_pipeline with sample audio: {sample_audio_path}")

    badge, latency_md, matched_kw, final_tx, partial_log, plot_fig, debug_df = run_pipeline(
        audio_file=str(sample_audio_path),
        language_code="hi",
        decoding_mode="greedy (beam=1)",
        vad_thresh=0.5,
        vad_silence=200,
        vad_speech=250,
        vad_pad=30,
    )

    print("Gradio Outputs:")
    print("Badge HTML:", badge[:60] + "...")
    print("Matched Keywords:", matched_kw)
    print("Final Transcript:", final_tx)
    print("Partial Log:", partial_log[:60] if partial_log else "Empty")
    print("Matplotlib Plot Object:", plot_fig)
    print(f"VAD Debug DataFrame Rows: {len(debug_df)}")

    assert plot_fig is not None, "Matplotlib plot figure should be generated!"
    print("--- Test Gradio App UI PASSED ---\n")

if __name__ == "__main__":
    test_app_ui()
