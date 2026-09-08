"""
Module 5: Gradio UI Entrypoint (app.py)
=======================================
Fully offline local desktop web interface for testing and tuning iTantra's voice pipeline:
Silero VAD -> Streaming STT -> Emergency Classifier.

Launches on localhost (share=False).
"""

import sys
import logging
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server rendering
import matplotlib.pyplot as plt

import gradio as gr

try:
    from .config import (
        VAD_THRESHOLD,
        VAD_MIN_SILENCE_MS,
        VAD_MIN_SPEECH_MS,
        VAD_SPEECH_PAD_MS,
        STT_BEAM_SIZE,
        STT_LANGUAGES,
        LANGUAGE_NAMES,
        DEFAULT_LANGUAGE,
    )
    from .pipeline import VoicePipeline
except ImportError:
    from config import (
        VAD_THRESHOLD,
        VAD_MIN_SILENCE_MS,
        VAD_MIN_SPEECH_MS,
        VAD_SPEECH_PAD_MS,
        STT_BEAM_SIZE,
        STT_LANGUAGES,
        LANGUAGE_NAMES,
        DEFAULT_LANGUAGE,
    )
    from pipeline import VoicePipeline

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger("iTantraApp")

# Instantiate Pipeline Orchestrator globally
pipeline = VoicePipeline()


def plot_speech_waveform(audio: np.ndarray, sr: int, speech_segments: list[tuple[int, int]]):
    """Generates a Matplotlib plot showing audio waveform with highlighted speech regions."""
    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=100)
    
    time_axis = np.linspace(0, len(audio) / sr, len(audio))
    ax.plot(time_axis, audio, color="#2b5c8f", alpha=0.6, linewidth=0.8, label="Audio Signal")

    # Highlight speech segments
    for idx, (start_sample, end_sample) in enumerate(speech_segments):
        start_t = start_sample / float(sr)
        end_t = end_sample / float(sr)
        label = "Speech Region" if idx == 0 else ""
        ax.axvspan(start_t, end_t, color="#4CAF50", alpha=0.35, label=label)
        ax.axvline(start_t, color="#2E7D32", linestyle="--", linewidth=1.0)
        ax.axvline(end_t, color="#C62828", linestyle="--", linewidth=1.0)

    ax.set_title("Silero VAD Speech Segmentation Waveform", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Amplitude", fontsize=10)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig


def run_pipeline(
    audio_file,
    language_code,
    decoding_mode,
    vad_thresh,
    vad_silence,
    vad_speech,
    vad_pad,
):
    """Gradio handler function running full voice pipeline and formatting UI outputs."""
    if audio_file is None:
        return (
            "<div style='padding: 10px; background: #eee; text-align: center;'>No audio uploaded</div>",
            "Please upload an audio file (.wav, .mp3, .m4a)",
            {},
            "",
            "",
            None,
            [],
        )

    # Parse beam size option
    beam_size = 1
    if "beam_size=4" in decoding_mode:
        beam_size = 4
    elif "beam_size=8" in decoding_mode:
        beam_size = 8

    # Execute voice pipeline
    result = pipeline.process_audio(
        audio_source=audio_file,
        language_code=language_code,
        vad_threshold=float(vad_thresh),
        vad_min_silence_ms=int(vad_silence),
        vad_min_speech_ms=int(vad_speech),
        vad_speech_pad_ms=int(vad_pad),
        beam_size=beam_size,
    )

    emergency_res = result["emergency_result"]
    is_emergency = emergency_res.get("is_emergency", False)
    priority = emergency_res.get("priority", "P2")

    # Format Emergency Badge HTML
    if is_emergency:
        badge_html = (
            "<div style='background-color: #d9534f; color: white; padding: 14px; "
            "font-weight: bold; text-align: center; border-radius: 8px; font-size: 20px; "
            "box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>"
            f"🚨 EMERGENCY DETECTED ({priority})</div>"
        )
    else:
        badge_html = (
            "<div style='background-color: #5cb85c; color: white; padding: 14px; "
            "font-weight: bold; text-align: center; border-radius: 8px; font-size: 20px; "
            "box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>"
            f"✅ NORMAL ({priority})</div>"
        )

    # Format Latency Breakdown Table Markdown
    latency_markdown = f"""
### ⏱️ Latency & Performance Breakdown

| Stage | Processing Latency |
| :--- | :--- |
| **Silero VAD** | **{result['vad_latency_ms']:.2f} ms** |
| **Streaming STT** | **{result['stt_latency_ms']:.2f} ms** |
| **Emergency Classifier** | **{result['classifier_latency_ms']:.2f} ms** |
| **TOTAL PIPELINE** | **{result['total_latency_ms']:.2f} ms** |

* **Audio Duration**: {result['audio_duration_s']:.2f}s
* **Real-Time Factor (RTF)**: `{result['real_time_factor']:.4f}x` *(Processing Time / Audio Time)*
* **Speech Regions**: {len(result['vad_segments'])} detected segment(s)
"""

    # Format Partial Transcripts Log
    partial_log_lines = [
        f"[{p['timestamp_s']:.2f}s] Partial: {p['text']}" for p in result["partial_transcripts"]
    ]
    partial_log_str = "\n".join(partial_log_lines) if partial_log_lines else "No partials generated."

    # Generate Matplotlib Waveform Plot
    audio_data, sr = pipeline.load_audio(audio_file)
    plot_fig = plot_speech_waveform(audio_data, sr, result["vad_segments"])

    # Prepare VAD Debug Log for DataFrame display
    debug_log_data = [
        [
            entry["frame_idx"],
            entry["timestamp_s"],
            entry["probability"],
            "TRUE" if entry["triggered"] else "FALSE",
            entry["event"] or "",
        ]
        for entry in result["vad_debug_log"]
    ]

    return (
        badge_html,
        latency_markdown,
        emergency_res.get("matched_keywords", []),
        result["final_transcript"],
        partial_log_str,
        plot_fig,
        debug_log_data,
    )


def create_ui():
    """Constructs Gradio single-page testing interface."""
    with gr.Blocks(title="iTantra Voice Pipeline Benchmarking Tool") as app:
        gr.Markdown("# 🎙️ iTantra Voice Pipeline Testing & Prototyping Tool")
        gr.Markdown(
            "Local desktop benchmarker for **Silero VAD → Streaming STT → Emergency Classifier**. "
            "Uses CPU-only ONNX inference to benchmark low-end Android mobile deployment reality."
        )

        with gr.Row():
            # Left Input Column
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Input & Configuration")
                audio_input = gr.Audio(
                    type="filepath",
                    label="Upload Audio File (WAV / MP3 / M4A)",
                    sources=["upload", "microphone"],
                )
                
                lang_input = gr.Dropdown(
                    choices=STT_LANGUAGES,
                    value=DEFAULT_LANGUAGE,
                    label="Target Language (10 Indic/EN Codes)",
                )

                decoding_input = gr.Radio(
                    choices=["greedy (beam=1)", "beam_size=4", "beam_size=8"],
                    value="greedy (beam=1)",
                    label="STT Decoding Strategy",
                )

                with gr.Accordion("⚙️ Advanced: VAD Tuning Parameters", open=False):
                    vad_thresh_slider = gr.Slider(
                        minimum=0.1, maximum=0.9, value=VAD_THRESHOLD, step=0.05, label="VAD Threshold"
                    )
                    vad_silence_slider = gr.Slider(
                        minimum=100, maximum=500, value=VAD_MIN_SILENCE_MS, step=10, label="VAD Min Silence (ms)"
                    )
                    vad_speech_slider = gr.Slider(
                        minimum=100, maximum=500, value=VAD_MIN_SPEECH_MS, step=10, label="VAD Min Speech (ms)"
                    )
                    vad_pad_slider = gr.Slider(
                        minimum=0, maximum=100, value=VAD_SPEECH_PAD_MS, step=5, label="VAD Speech Pad (ms)"
                    )

                process_btn = gr.Button("⚡ Process Voice Pipeline", variant="primary", size="lg")

            # Right Output Column
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Pipeline Results & Classification")
                badge_output = gr.HTML(
                    value="<div style='padding: 14px; background: #eceff1; text-align: center; border-radius: 8px; font-weight: bold;'>Awaiting Audio Input</div>"
                )
                matched_keywords_output = gr.JSON(label="Matched Emergency / Location Keywords")
                final_transcript_output = gr.Textbox(label="Final STT Transcript", lines=2)
                partial_log_output = gr.Textbox(
                    label="Streaming Partial Hypotheses (Append-Only Log)", lines=5
                )
                latency_markdown_output = gr.Markdown()

        with gr.Row():
            plot_output = gr.Plot(label="Silero VAD Speech Segmentation Visualization")

        with gr.Accordion("🔍 VAD Frame-by-Frame Debug Log", open=False):
            vad_log_output = gr.Dataframe(
                headers=["Frame Index", "Timestamp (s)", "Probability", "Speech Triggered", "Event Transition"],
                datatype=["number", "number", "number", "str", "str"],
                label="Sequential VAD Decisions",
            )

        process_btn.click(
            fn=run_pipeline,
            inputs=[
                audio_input,
                lang_input,
                decoding_input,
                vad_thresh_slider,
                vad_silence_slider,
                vad_speech_slider,
                vad_pad_slider,
            ],
            outputs=[
                badge_output,
                latency_markdown_output,
                matched_keywords_output,
                final_transcript_output,
                partial_log_output,
                plot_output,
                vad_log_output,
            ],
        )

    return app


if __name__ == "__main__":
    app = create_ui()
    # Launch locally on share=False
    logger.info("Launching Gradio UI locally on http://127.0.0.1:7860...")
    app.launch(share=False, server_name="127.0.0.1", server_port=7860)
