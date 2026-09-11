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

    # 1. Test Direct PTT Mode (Real Speech Audio with Live Incremental Streaming STT)
    ptt_audio_path = Path(__file__).resolve().parent / "audio" / "non_wake_normal.wav"
    print(f"Testing run_pipeline Direct PTT Mode with: {ptt_audio_path}")

    ptt_yields = []
    for step_output in run_pipeline(
        audio_file=str(ptt_audio_path),
        language_code="hi",
        decoding_mode="greedy (beam=1)",
        vad_thresh=0.5,
        vad_silence=200,
        vad_speech=250,
        vad_pad=30,
        enable_wake_word=False,
    ):
        ptt_yields.append(step_output)

    print(f"Direct PTT yielded {len(ptt_yields)} real-time UI streaming updates.")
    assert len(ptt_yields) >= 2, "PTT mode should yield initial state, intermediate partials, and final state!"

    badge, state_html, latency_md, matched_kw, final_tx, partial_log, tx_info, plot_fig, debug_df = ptt_yields[-1]

    print("Gradio Direct PTT Final Outputs:")
    print("Badge HTML:", badge[:60] + "...")
    print("State Machine HTML:", state_html[:60] + "...")
    print("Matched Keywords:", matched_kw)
    print("Final Transcript:", final_tx)
    print("Transmission Info:", tx_info[:60] + "...")
    print("Partial Log:\n", partial_log[:100] + "..." if len(partial_log) > 100 else partial_log)
    print("Matplotlib Plot Object:", plot_fig)
    print(f"VAD Debug DataFrame Rows: {len(debug_df)}")

    assert plot_fig is not None, "Matplotlib plot figure should be generated at final step!"
    assert len(final_tx) > 0, "PTT mode should produce a non-empty transcript for real speech audio!"
    assert "NORMAL" in badge, "Normal speech should produce NORMAL badge in PTT mode"

    # 2. Test Voice Keyword Mode (Wake-word 'Hey iTantra' -> Command Capture -> STT)
    wake_audio_path = Path(__file__).resolve().parent / "audio" / "wake_emergency.wav"
    print(f"\nTesting run_pipeline Voice Keyword Detection Mode with: {wake_audio_path}")

    wake_yields = []
    for step_output in run_pipeline(
        audio_file=str(wake_audio_path),
        language_code="hi",
        decoding_mode="greedy (beam=1)",
        vad_thresh=0.5,
        vad_silence=200,
        vad_speech=250,
        vad_pad=30,
        enable_wake_word=True,
    ):
        wake_yields.append(step_output)

    print(f"Voice Keyword mode yielded {len(wake_yields)} real-time UI streaming updates.")
    assert len(wake_yields) >= 2, "Voice Keyword mode should yield real-time streaming updates!"

    w_badge, w_state, w_latency, w_matched, w_tx, w_partial, w_txinfo, w_plot, w_df = wake_yields[-1]
    print("Wake-word Badge HTML:", w_badge[:60] + "...")
    print("Wake-word Matched Keywords:", w_matched)
    print("Wake-word Final Transcript:", w_tx)
    print("Wake-word Transmission:", w_txinfo[:60] + "...")

    assert "EMERGENCY" in w_badge, "Wake emergency audio should trigger EMERGENCY badge"
    assert len(w_matched) > 0, "Keywords should be matched in wake emergency audio"
    assert "फायर" in w_tx or "इमरजेंसी" in w_tx or "हेल्थ" in w_tx, "Recognized transcript should match audio content!"
    print("--- Test Gradio App UI PASSED ---\n")

if __name__ == "__main__":
    test_app_ui()
