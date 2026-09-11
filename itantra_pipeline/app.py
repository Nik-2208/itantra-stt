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
        DEFAULT_SAMPLE_RATE,
        VAD_THRESHOLD,
        VAD_MIN_SILENCE_MS,
        VAD_MIN_SPEECH_MS,
        VAD_SPEECH_PAD_MS,
        STT_BEAM_SIZE,
        STT_LANGUAGES,
        LANGUAGE_NAMES,
        DEFAULT_LANGUAGE,
        WAKE_WORD_THRESHOLD,
        COMMAND_TIMEOUT_MS,
        COMMAND_SILENCE_TIMEOUT_MS,
    )
    from .pipeline import VoicePipeline, StreamPipelineEventType
except ImportError:
    from config import (
        DEFAULT_SAMPLE_RATE,
        VAD_THRESHOLD,
        VAD_MIN_SILENCE_MS,
        VAD_MIN_SPEECH_MS,
        VAD_SPEECH_PAD_MS,
        STT_BEAM_SIZE,
        STT_LANGUAGES,
        LANGUAGE_NAMES,
        DEFAULT_LANGUAGE,
        WAKE_WORD_THRESHOLD,
        COMMAND_TIMEOUT_MS,
        COMMAND_SILENCE_TIMEOUT_MS,
    )
    from pipeline import VoicePipeline, StreamPipelineEventType

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


def format_state_machine_html(
    state_transitions: list,
    final_decision: str,
    wake_keyword: str = None,
    enable_wake_word: bool = False,
) -> str:
    """Renders a responsive visual stepper for PTT vs Voice Keyword mode."""
    states_hit = {t.get("state") for t in state_transitions if t.get("state")}

    if enable_wake_word:
        stages = [
            ("IDLE_LISTENING", "🎧 1. Idle Listening", "Lightweight KWS dormant loop (<0.5ms)"),
            ("WAKE_DETECTED", f"⚡ 2. Wake Detected: {wake_keyword or 'Yes'}", "Voice trigger verified"),
            ("COMMAND_CAPTURE", "🎙️ 3. Command Capture", "Silero VAD speech boundary tracking"),
            ("PROCESSING", "⚙️ 4. Incremental STT", "IndicConformer streaming inference"),
            (
                "DECISION",
                "🚨 5. EMERGENCY (P0)" if final_decision == "EMERGENCY" else ("✅ 5. NORMAL (P2)" if final_decision == "NORMAL" else "📊 5. Classification"),
                "Keyword safety & priority routing",
            ),
            ("TRANSMITTING", "📡 6. Transmitting", "Low-bitrate text payload dispatched"),
        ]
    else:
        stages = [
            ("COMMAND_CAPTURE", "🎙️ 1. Push-to-Talk Active", "Mic speech capture & VAD boundary"),
            ("PROCESSING", "⚙️ 2. Real-Time Streaming STT", "Live incremental partial hypotheses"),
            (
                "DECISION",
                "🚨 3. EMERGENCY (P0)" if final_decision == "EMERGENCY" else ("✅ 3. NORMAL (P2)" if final_decision == "NORMAL" else "📊 3. Classification"),
                "Offline keyword safety classifier",
            ),
            ("TRANSMITTING", "📡 4. Mesh Transmitted", "Zero-raw-audio low-bitrate packet"),
        ]

    html = "<div style='display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; justify-content: space-between;'>"
    for st_id, title, desc in stages:
        if enable_wake_word:
            is_active = (st_id in states_hit) or (st_id == "DECISION" and ("EMERGENCY" in states_hit or "NORMAL" in states_hit))
        else:
            # In PTT mode:
            if st_id == "COMMAND_CAPTURE":
                is_active = True
            elif st_id == "PROCESSING":
                is_active = "PROCESSING" in states_hit or len(states_hit) > 0
            elif st_id == "DECISION":
                is_active = ("EMERGENCY" in states_hit or "NORMAL" in states_hit or final_decision in ("EMERGENCY", "NORMAL"))
            elif st_id == "TRANSMITTING":
                is_active = "TRANSMITTING" in states_hit or final_decision in ("EMERGENCY", "NORMAL")
            else:
                is_active = False

        if is_active:
            if st_id == "DECISION" and final_decision == "EMERGENCY":
                bg = "linear-gradient(135deg, #d9534f, #c9302c)"
            elif is_active and st_id == "DECISION":
                bg = "linear-gradient(135deg, #5cb85c, #449d44)"
            elif st_id in ("WAKE_DETECTED", "COMMAND_CAPTURE"):
                bg = "linear-gradient(135deg, #f0ad4e, #ec971f)" if enable_wake_word else "linear-gradient(135deg, #337ab7, #286090)"
            else:
                bg = "linear-gradient(135deg, #337ab7, #286090)"
            border = "2px solid rgba(255,255,255,0.4)"
            color = "#ffffff"
            shadow = "0 3px 8px rgba(0,0,0,0.18)"
        else:
            bg = "#f5f5f5"
            border = "1px solid #e0e0e0"
            color = "#888888"
            shadow = "none"

        html += f"""
        <div style='flex: 1; min-width: 140px; background: {bg}; border: {border}; color: {color};
                    padding: 10px; border-radius: 8px; box-shadow: {shadow}; font-family: sans-serif;'>
            <div style='font-size: 13px; font-weight: bold;'>{title}</div>
            <div style='font-size: 11px; opacity: 0.9; margin-top: 2px;'>{desc}</div>
        </div>
        """
    html += "</div>"
    return html


def run_pipeline(
    audio_file,
    language_code,
    decoding_mode,
    vad_thresh,
    vad_silence,
    vad_speech,
    vad_pad,
    enable_wake_word=False,
    wake_thresh=WAKE_WORD_THRESHOLD,
    cmd_timeout=COMMAND_TIMEOUT_MS,
):
    """Gradio generator handler running streaming voice pipeline with live partial & final UI updates."""
    if audio_file is None:
        yield (
            "<div style='padding: 10px; background: #eee; text-align: center;'>No audio uploaded</div>",
            "<div style='padding: 8px; background: #fafafa; text-align: center;'>Awaiting audio stream</div>",
            "Please upload or record an audio file (.wav, .mp3, .m4a)",
            {},
            "",
            "",
            "Awaiting transmission...",
            None,
            [],
        )
        return

    # Parse beam size option
    beam_size = 1
    if "beam_size=4" in decoding_mode:
        beam_size = 4
    elif "beam_size=8" in decoding_mode:
        beam_size = 8

    try:
        audio_data, sr = pipeline.load_audio(audio_file, target_sr=DEFAULT_SAMPLE_RATE)
        frame_size = pipeline.vad.window_samples

        def frame_generator():
            for i in range(0, len(audio_data), frame_size):
                yield audio_data[i : i + frame_size]

        stream = pipeline.process_stream(
            frame_generator=frame_generator(),
            language_code=language_code,
            vad_threshold=float(vad_thresh),
            vad_min_silence_ms=int(vad_silence),
            vad_min_speech_ms=int(vad_speech),
            vad_speech_pad_ms=int(vad_pad),
            beam_size=beam_size,
            enable_wake_word=bool(enable_wake_word),
            wake_threshold=float(wake_thresh),
            command_timeout_ms=int(cmd_timeout),
        )

        state_transitions = []
        partial_transcripts = []
        final_transcripts = []
        emergency_res = {"is_emergency": False, "matched_keywords": [], "priority": "P2"}
        early_emergency = None
        wake_word_event = None
        transmission_event = None
        current_live_transcript = ""
        decision = "NORMAL"

        if enable_wake_word:
            current_badge = (
                "<div style='background: linear-gradient(135deg, #4a90e2, #357abd); color: white; padding: 14px; "
                "font-weight: bold; text-align: center; border-radius: 8px; font-size: 18px; "
                "box-shadow: 0 4px 6px rgba(0,0,0,0.15);'>"
                "🎧 IDLE LISTENING (Awaiting 'Hey iTantra'...)</div>"
            )
            initial_partial_log = "[Idle Loop] Low-power KWS active. Say 'Hey iTantra' to trigger capture."
        else:
            current_badge = (
                "<div style='background: linear-gradient(135deg, #337ab7, #286090); color: white; padding: 14px; "
                "font-weight: bold; text-align: center; border-radius: 8px; font-size: 18px; "
                "box-shadow: 0 4px 6px rgba(0,0,0,0.15);'>"
                "🎙️ PUSH-TO-TALK ACTIVE (Streaming STT...)</div>"
            )
            initial_partial_log = "[PTT Active] Streaming audio frames -> Silero VAD + IndicConformer STT..."

        state_html = format_state_machine_html(
            state_transitions,
            final_decision="NORMAL",
            wake_keyword=None,
            enable_wake_word=enable_wake_word,
        )

        # Initial yield immediately as stream starts
        yield (
            current_badge,
            state_html,
            "### ⏳ Processing speech stream in real time...",
            [],
            current_live_transcript,
            initial_partial_log,
            "Preparing mesh transmission...",
            None,
            [],
        )

        metrics = None
        try:
            while True:
                event = next(stream)

                if event.event_type == StreamPipelineEventType.STATE_CHANGED:
                    state_transitions.append({
                        "timestamp_s": event.timestamp_s,
                        "state": event.data.get("to_state"),
                        "data": event.data,
                    })
                    to_state = event.data.get("to_state")
                    if to_state == "WAKE_DETECTED":
                        current_badge = (
                            "<div style='background: linear-gradient(135deg, #f0ad4e, #ec971f); color: white; padding: 14px; "
                            "font-weight: bold; text-align: center; border-radius: 8px; font-size: 18px; "
                            "box-shadow: 0 4px 6px rgba(0,0,0,0.15);'>"
                            f"⚡ WAKE DETECTED: '{event.data.get('keyword', 'HEY I')}' (Starting Command Capture)</div>"
                        )
                    elif to_state == "COMMAND_CAPTURE":
                        current_badge = (
                            "<div style='background: linear-gradient(135deg, #337ab7, #286090); color: white; padding: 14px; "
                            "font-weight: bold; text-align: center; border-radius: 8px; font-size: 18px; "
                            "box-shadow: 0 4px 6px rgba(0,0,0,0.15);'>"
                            "🎙️ COMMAND CAPTURE (Streaming Speech...)</div>"
                        )

                elif event.event_type == StreamPipelineEventType.WAKE_WORD_DETECTED:
                    wake_word_event = event.data

                elif event.event_type == StreamPipelineEventType.PARTIAL_TRANSCRIPT:
                    p_text = event.data.get("partial_transcript", "")
                    chunk_lat = event.data.get("chunk_latency_ms", 0.0)
                    partial_transcripts.append({
                        "timestamp_s": event.timestamp_s,
                        "text": p_text,
                        "chunk_latency_ms": chunk_lat,
                    })
                    if p_text:
                        current_live_transcript = p_text

                    p_lines = [
                        f"[{p['timestamp_s']:.2f}s] Partial: {p['text']} ({p.get('chunk_latency_ms', 0):.1f}ms)"
                        for p in partial_transcripts
                        if p.get("text")
                    ]
                    partial_log_str = "\n".join(p_lines) if p_lines else "Streaming audio..."

                    # Live UI partial hypothesis yield
                    yield (
                        current_badge,
                        format_state_machine_html(
                            state_transitions,
                            decision,
                            wake_keyword=(wake_word_event.get("keyword") if wake_word_event else None),
                            enable_wake_word=enable_wake_word,
                        ),
                        "### ⚡ Streaming partial hypotheses in real time...",
                        emergency_res.get("matched_keywords", []),
                        current_live_transcript,
                        partial_log_str,
                        f"Payload: '{current_live_transcript}'",
                        None,
                        [],
                    )

                elif event.event_type == StreamPipelineEventType.EMERGENCY_ALERT:
                    early_emergency = event.data
                    emergency_res = event.data
                    decision = "EMERGENCY"

                elif event.event_type == StreamPipelineEventType.FINAL_TRANSCRIPT:
                    f_text = event.data.get("final_transcript", "")
                    if f_text:
                        final_transcripts.append(f_text)
                    emergency_res = event.data.get("emergency_result", {})
                    decision = event.data.get("decision", "NORMAL")
                    current_live_transcript = f_text

                elif event.event_type == StreamPipelineEventType.TRANSMISSION:
                    transmission_event = event.data

        except StopIteration as e:
            metrics = e.value

        final_transcript = " ".join(final_transcripts).strip() if final_transcripts else current_live_transcript.strip()
        if not emergency_res or "is_emergency" not in emergency_res:
            pipeline.classifier.load_language(language_code)
            emergency_res = pipeline.classifier.classify(final_transcript)

        if early_emergency and early_emergency.get("is_emergency", False):
            emergency_res["is_emergency"] = True
            emergency_res["priority"] = "P0"
            emergency_res["matched_keywords"] = list(dict.fromkeys(
                emergency_res.get("matched_keywords", []) + early_emergency.get("matched_keywords", [])
            ))

        is_emergency = emergency_res.get("is_emergency", False)
        priority = emergency_res.get("priority", "P0" if is_emergency else "P2")
        decision = "EMERGENCY" if is_emergency else "NORMAL"

        # Final Badge & Output Presentation
        if enable_wake_word and wake_word_event is None:
            badge_html = (
                "<div style='background: linear-gradient(135deg, #4a90e2, #357abd); color: white; padding: 14px; "
                "font-weight: bold; text-align: center; border-radius: 8px; font-size: 18px; "
                "box-shadow: 0 4px 6px rgba(0,0,0,0.15);'>"
                "🎧 IDLE LISTENING (No Wake-Word Triggered)</div>"
            )
            final_transcript_display = (
                "[Awaiting Wake-Word] No wake phrase ('Hey iTantra') detected. "
                "In Voice Keyword mode, say 'Hey iTantra' before speaking, or uncheck 'Enable Voice Keyword Detection' for Direct PTT speech."
            )
            partial_log_str = (
                "[Dormant Loop] Lightweight KWS (<0.5ms) active in IDLE_LISTENING. "
                "VAD and STT are dormant to preserve battery until 'Hey iTantra' is triggered."
            )
            tx_info = "Awaiting trigger (No transmission in IDLE state)"
        else:
            if is_emergency:
                badge_html = (
                    "<div style='background: linear-gradient(135deg, #d9534f, #c9302c); color: white; padding: 14px; "
                    "font-weight: bold; text-align: center; border-radius: 8px; font-size: 20px; "
                    "box-shadow: 0 4px 6px rgba(0,0,0,0.15);'>"
                    f"🚨 EMERGENCY DETECTED ({priority})</div>"
                )
            else:
                badge_html = (
                    "<div style='background: linear-gradient(135deg, #5cb85c, #449d44); color: white; padding: 14px; "
                    "font-weight: bold; text-align: center; border-radius: 8px; font-size: 20px; "
                    "box-shadow: 0 4px 6px rgba(0,0,0,0.15);'>"
                    f"✅ NORMAL ({priority})</div>"
                )
            final_transcript_display = final_transcript
            p_lines = [
                f"[{p['timestamp_s']:.2f}s] Partial: {p['text']} (chunk latency: {p.get('chunk_latency_ms', 0):.2f}ms)"
                for p in partial_transcripts
                if p.get("text")
            ]
            partial_log_str = "\n".join(p_lines) if p_lines else ("Final: " + final_transcript if final_transcript else "No speech detected in audio.")
            if transmission_event:
                tx_info = (
                    f"Channel: {transmission_event.get('channel', 'STANDARD_COMM_CHANNEL')} | Priority: {transmission_event.get('priority', 'P2')}\n"
                    f"Payload Dispatched: '{transmission_event.get('payload', '')}'\n"
                    f"Timestamp: {transmission_event.get('timestamp_s', 0.0):.2f}s (Low-bitrate text mesh ready)"
                )
            else:
                tx_info = f"Channel: {'SOS_PRIORITY_CHANNEL' if is_emergency else 'STANDARD_COMM_CHANNEL'} | Priority: {priority}\nPayload: '{final_transcript}'"

        state_html = format_state_machine_html(
            state_transitions,
            final_decision=decision,
            wake_keyword=(wake_word_event.get("keyword") if wake_word_event else None),
            enable_wake_word=enable_wake_word,
        )

        if metrics:
            ftl_str = f"{metrics.first_token_latency_ms:.2f} ms" if metrics.first_token_latency_ms is not None else "N/A"
            eoul_str = f"{metrics.endpointing_latency_ms:.2f} ms" if metrics.endpointing_latency_ms is not None else "N/A"
            avg_chunk = metrics.avg_chunk_latency_ms
            p95_chunk = metrics.p95_chunk_latency_ms
            w_lat_str = f"{metrics.wake_detection_latency_ms:.2f} ms" if metrics.wake_detection_latency_ms is not None else "N/A"
            w2c_str = f"{metrics.wake_to_capture_latency_ms:.3f} ms" if metrics.wake_to_capture_latency_ms is not None else "N/A"
            tot_w2d_str = f"{metrics.total_wake_to_decision_latency_ms:.2f} ms" if metrics.total_wake_to_decision_latency_ms is not None else "N/A"
            cpu_str = f"{metrics.cpu_percent:.1f}%" if metrics.cpu_percent is not None else "N/A"
            ram_str = f"{metrics.ram_rss_mb:.2f} MB" if metrics.ram_rss_mb is not None else "N/A"
            tot_lat_str = f"{metrics.total_latency_ms:.2f} ms"
            dur_str = f"{metrics.total_audio_duration_s:.2f}s"
            rtf_str = f"{metrics.rtf:.4f}x"
            vad_lat_str = f"{metrics.vad_latency_ms:.2f} ms"
            stt_lat_str = f"{metrics.stt_latency_ms:.2f} ms"
            cls_lat_str = f"{metrics.classifier_latency_ms:.2f} ms"
        else:
            ftl_str = eoul_str = w_lat_str = w2c_str = tot_w2d_str = cpu_str = ram_str = tot_lat_str = dur_str = rtf_str = vad_lat_str = stt_lat_str = cls_lat_str = "N/A"
            avg_chunk = p95_chunk = 0.0

        latency_markdown = f"""
### ⏱️ Streaming Latency & Performance Breakdown

| Pipeline Stage / Metric | Measured Latency | Target SLA |
| :--- | :--- | :--- |
| **Wake-Word Detection (KWS)** | **{w_lat_str}** | `< 25 ms` |
| **Wake ➔ Capture Switch** | **{w2c_str}** | `< 2 ms` |
| **Silero VAD (Frame-by-Frame)** | **{vad_lat_str}** | `< 20 ms` |
| **Streaming STT (Incremental)** | **{stt_lat_str}** | `< 600 ms` |
| **Emergency Classifier** | **{cls_lat_str}** | `< 5 ms` |
| **TOTAL PIPELINE LATENCY** | **{tot_lat_str}** | — |
| **First-Token Latency (FTL)** | **{ftl_str}** | `< 50 ms` |
| **Total Wake ➔ Decision Latency** | **{tot_w2d_str}** | `< 1000 ms` |
| **Avg Chunk Latency (100ms)** | **{avg_chunk:.2f} ms** *(p95: {p95_chunk:.2f} ms)* | `< 30 ms` |
| **Endpointing Latency (EOUL)** | **{eoul_str}** | `< 1500 ms` |

* **Audio Duration**: `{dur_str}` | **RTF**: `{rtf_str}`
* **Process CPU Utilization**: `{cpu_str}` | **Process RAM (Max RSS)**: `{ram_str}`
* **Operation Mode**: `{'⚡ VOICE KEYWORD DETECTION (KWS)' if enable_wake_word else '🎙️ DIRECT PUSH-TO-TALK (PTT)'}`
"""

        # Generate Matplotlib Waveform Plot
        vad_segments = pipeline.vad.get_speech_segments(
            audio_data,
            threshold=float(vad_thresh),
            min_speech_ms=int(vad_speech),
            min_silence_ms=int(vad_silence),
            speech_pad_ms=int(vad_pad),
        )
        plot_fig = plot_speech_waveform(audio_data, sr, vad_segments)

        # Prepare VAD Debug Log for DataFrame display
        debug_log_data = [
            [
                entry.get("frame_idx", 0),
                entry.get("timestamp_s", 0.0),
                entry.get("probability", 0.0),
                "TRUE" if entry.get("triggered", False) else "FALSE",
                entry.get("event") or "",
            ]
            for entry in pipeline.vad.debug_log
        ]

        # Final complete yield with plot, metrics, and logs
        yield (
            badge_html,
            state_html,
            latency_markdown,
            emergency_res.get("matched_keywords", []),
            final_transcript_display,
            partial_log_str,
            tx_info,
            plot_fig,
            debug_log_data,
        )

    except Exception as e:
        logger.error(f"Error running pipeline: {e}", exc_info=True)
        error_badge = (
            "<div style='background-color: #f0ad4e; color: white; padding: 14px; "
            "font-weight: bold; text-align: center; border-radius: 8px; font-size: 18px;'>"
            f"⚠️ PROCESSING ERROR: {str(e)[:80]}</div>"
        )
        error_md = f"### ❌ Pipeline Error\n\n```\n{e}\n```\nPlease check the audio format or server logs."
        yield (
            error_badge,
            "<div style='padding: 8px; background: #ffebee; color: #c62828;'>Pipeline execution failed</div>",
            error_md,
            [],
            f"Error: {e}",
            f"Failed to process stream: {e}",
            f"Transmission failed: {e}",
            None,
            [],
        )


def create_ui():
    """Constructs Gradio single-page testing interface."""
    with gr.Blocks(title="iTantra Voice Pipeline Benchmarking Tool") as app:
        gr.Markdown("# 🎙️ iTantra Voice Pipeline Testing & Prototyping Tool")
        gr.Markdown(
            "Local desktop benchmarker for **Push-to-Talk / Offline Voice Keyword ➔ Silero VAD ➔ Streaming STT ➔ Emergency Classifier**. "
            "Uses CPU-only ONNX inference to benchmark low-end Android mobile deployment reality."
        )

        with gr.Row():
            # Left Input Column
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Input & Configuration")
                audio_input = gr.Audio(
                    type="filepath",
                    label="Push-to-Talk / Audio Stream (Upload WAV/MP3 or Record Live Mic)",
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

                enable_wake_word_cb = gr.Checkbox(
                    label="Enable Voice Keyword Detection ('Hey iTantra' / KWS)",
                    value=False,
                    info="When enabled, runs low-power offline voice trigger. When disabled (default), Push-to-Talk mode is active.",
                )

                with gr.Accordion("⚡ Voice Keyword / Hands-Free Settings", open=False):
                    wake_thresh_slider = gr.Slider(
                        minimum=0.01, maximum=0.50, value=WAKE_WORD_THRESHOLD, step=0.01,
                        label="Wake-Word Sensitivity Threshold (Lower = More Sensitive)"
                    )
                    cmd_timeout_slider = gr.Slider(
                        minimum=1000, maximum=10000, value=COMMAND_TIMEOUT_MS, step=500,
                        label="Command Capture Timeout (ms)"
                    )

                with gr.Accordion("⚙️ Advanced: VAD & STT Tuning Parameters", open=False):
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

                process_btn = gr.Button("⚡ Start Streaming Pipeline", variant="primary", size="lg")

            # Right Output Column
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Pipeline Results & Classification")
                badge_output = gr.HTML(
                    value="<div style='padding: 14px; background: #eceff1; text-align: center; border-radius: 8px; font-weight: bold;'>Awaiting Audio Input</div>"
                )
                state_machine_output = gr.HTML(
                    value="<div style='padding: 8px; background: #fafafa; text-align: center; color: #777;'>State Machine: Ready</div>"
                )
                matched_keywords_output = gr.JSON(label="Matched Emergency / Location Keywords")
                final_transcript_output = gr.Textbox(label="Live & Final STT Transcript", lines=2)
                transmission_output = gr.Textbox(label="Dispatched Mesh Transport Packet", lines=2)
                partial_log_output = gr.Textbox(
                    label="Streaming Partial Hypotheses (Live Incremental Log)", lines=4
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
                enable_wake_word_cb,
                wake_thresh_slider,
                cmd_timeout_slider,
            ],
            outputs=[
                badge_output,
                state_machine_output,
                latency_markdown_output,
                matched_keywords_output,
                final_transcript_output,
                partial_log_output,
                transmission_output,
                plot_output,
                vad_log_output,
            ],
        )

    return app


if __name__ == "__main__":
    app = create_ui()
    logger.info("Launching Gradio UI locally on http://127.0.0.1:7860...")
    app.launch(share=False, server_name="127.0.0.1", server_port=7860)

