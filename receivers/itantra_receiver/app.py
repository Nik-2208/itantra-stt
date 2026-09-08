"""
iTantra Receiver Pipeline - Interactive Prototyping & Benchmarking UI (app.py)
==============================================================================
Minimal, zero-network, local desktop testing UI using Gradio.
Launches strictly with share=False and CPU-only inference.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import gradio as gr
import numpy as np

import config
from benchmark import PipelineBenchmark
from formatter import ReceiverFormatter
from pipeline import ReceiverPipeline
from translation import OnDemandTranslator
from tts import OfflineTTS

# Initialize shared receiver pipeline
pipeline = ReceiverPipeline(
    translator=OnDemandTranslator(),
    tts=OfflineTTS(),
    formatter=ReceiverFormatter(),
    output_dir=config.TEMP_AUDIO_DIR,
)

# Global debug log accumulator
debug_log_history: list[str] = []


def add_log_entry(msg: str):
    """Appends timestamped log entry."""
    now_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    entry = f"[{now_str}] {msg}"
    debug_log_history.append(entry)
    # Keep last 100 entries
    if len(debug_log_history) > 100:
        debug_log_history.pop(0)


def process_translate_and_speak(
    transcript: str,
    source_lang: str,
    target_lang: str,
    beam_choice: str,
    is_emergency: bool,
    emergency_keywords_str: str,
):
    """
    Executes on-demand translation, output formatting, and TTS speech synthesis.
    """
    if not transcript or not transcript.strip():
        return (
            "",  # translated text
            None,  # audio path
            "0.00 ms",  # audio duration
            "Error: Transcript cannot be empty.",  # formatted output
            "{}",  # json output
            "N/A",  # latency summary
            "\n".join(debug_log_history),
        )

    # Parse beam size
    beam_size = 1
    if "4" in beam_choice:
        beam_size = 4
    elif "8" in beam_choice:
        beam_size = 8

    pipeline.translator.set_beam_size(beam_size)

    # Prepare emergency metadata
    emergency_meta = None
    if is_emergency:
        keywords = [k.strip() for k in emergency_keywords_str.split(",") if k.strip()]
        emergency_meta = {
            "is_emergency": True,
            "priority": "P1",
            "matched_keywords": keywords,
        }

    add_log_entry("Pipeline execution triggered by user.")
    add_log_entry(f"Source Language: {source_lang} | Target Language: {target_lang} | Beam: {beam_size}")
    add_log_entry("Translation started...")

    try:
        result = pipeline.translate_and_speak(
            text=transcript,
            source_language=source_lang,
            target_language=target_lang,
            emergency_result=emergency_meta,
        )

        timing = result["timing"]
        bench = result["benchmark"]
        audio_info = result["audio"]
        formatted = result["formatted_output"]

        add_log_entry(f"Translation completed in {timing['translation_latency_ms']:.2f} ms")
        add_log_entry(f"Formatting completed in {timing['formatter_latency_ms']:.2f} ms")
        add_log_entry(f"TTS completed in {timing['tts_latency_ms']:.2f} ms")
        add_log_entry(f"Audio written to {audio_info['path']} in {timing['audio_write_latency_ms']:.2f} ms")
        add_log_entry(f"Total pipeline completed in {timing['total_latency_ms']:.2f} ms")

        # Format human-readable summary card
        summary_card = formatted["display"]["formatted_summary"]
        json_output = json.dumps(formatted, indent=2, ensure_ascii=False)

        latency_table = (
            f"| Stage | Latency |\n"
            f"|---|---|\n"
            f"| Translation | **{timing['translation_latency_ms']:.2f} ms** |\n"
            f"| Formatting | **{timing['formatter_latency_ms']:.2f} ms** |\n"
            f"| TTS Synthesis | **{timing['tts_latency_ms']:.2f} ms** |\n"
            f"| Audio I/O | **{timing['audio_write_latency_ms']:.2f} ms** |\n"
            f"| **TOTAL PIPELINE** | **{timing['total_latency_ms']:.2f} ms** |\n\n"
            f"- **Audio Duration:** `{audio_info['duration_ms']:.2f} ms`\n"
            f"- **TTS Real-Time Factor (RTF):** `{bench['tts_rtf']:.4f}`\n"
            f"- **End-to-End RTF:** `{bench['end_to_end_rtf']:.4f}`"
        )

        return (
            result["translated_text"],
            audio_info["path"],
            f"{audio_info['duration_ms']:.2f} ms",
            summary_card,
            json_output,
            latency_table,
            "\n".join(debug_log_history),
        )

    except Exception as e:
        err_msg = f"Error during pipeline execution: {str(e)}"
        add_log_entry(f"ERROR: {err_msg}")
        return (
            "",
            None,
            "0.00 ms",
            err_msg,
            "{}",
            "N/A",
            "\n".join(debug_log_history),
        )


def run_benchmark_ui(
    custom_text: str,
    source_lang: str,
    target_lang: str,
    warmup_runs: int,
    measured_runs: int,
):
    """
    Executes benchmark harness from UI and generates report downloads.
    """
    cases = []
    # If custom text provided, run single custom case
    if custom_text and custom_text.strip():
        cases = [
            {
                "id": "ui_custom_case",
                "category": "custom",
                "source_language": source_lang,
                "target_language": target_lang,
                "text": custom_text.strip(),
            }
        ]
    else:
        # Load standard benchmark cases
        cases_path = config.BASE_DIR / "benchmark_cases.json"
        if cases_path.exists():
            with open(cases_path, "r", encoding="utf-8") as f:
                cases = json.load(f)
        else:
            cases = [
                {
                    "id": "default_short_hi",
                    "category": "short",
                    "source_language": "hi",
                    "target_language": "en",
                    "text": "मदद कीजिए।",
                }
            ]

    benchmark_runner = PipelineBenchmark(pipeline=pipeline)
    report_payload = benchmark_runner.run(
        test_cases=cases,
        warmup_runs=int(warmup_runs),
        measured_runs=int(measured_runs),
    )
    json_path, md_path = benchmark_runner.export_reports(report_payload)

    summary = report_payload["overall_summary"]
    table_md = (
        f"### Benchmark Aggregate Results ({len(cases)} Cases, {measured_runs} Runs each)\n\n"
        f"| Stage | Mean (ms) | Median (ms) | P95 (ms) | StdDev (ms) |\n"
        f"|---|---|---|---|---|\n"
        f"| Translation | {summary['translation']['mean']} | {summary['translation']['median']} | {summary['translation']['p95']} | {summary['translation']['std_dev']} |\n"
        f"| Formatting | {summary['formatter']['mean']} | {summary['formatter']['median']} | {summary['formatter']['p95']} | {summary['formatter']['std_dev']} |\n"
        f"| TTS Synthesis | {summary['tts']['mean']} | {summary['tts']['median']} | {summary['tts']['p95']} | {summary['tts']['std_dev']} |\n"
        f"| Audio Write | {summary['audio_io']['mean']} | {summary['audio_io']['median']} | {summary['audio_io']['p95']} | {summary['audio_io']['std_dev']} |\n"
        f"| **TOTAL PIPELINE** | **{summary['total_pipeline']['mean']}** | **{summary['total_pipeline']['median']}** | **{summary['total_pipeline']['p95']}** | **{summary['total_pipeline']['std_dev']}** |\n\n"
        f"- **Mean TTS RTF:** `{summary['mean_tts_rtf']}`\n"
        f"- **Mean E2E RTF:** `{summary['mean_e2e_rtf']}`"
    )

    return table_md, str(json_path), str(md_path)


# ==============================================================================
# GRADIO INTERFACE LAYOUT
# ==============================================================================
def create_ui():
    lang_choices = config.SUPPORTED_LANGUAGES
    beam_choices = ["Greedy / Beam 1", "Beam 4", "Beam 8"]

    with gr.Blocks(title="iTantra Receiver Pipeline") as demo:
        gr.Markdown(
            "# iTantra Receiver Pipeline: On-Demand Translation & TTS Lab\n"
            "*Fully Offline CPU-Only Prototyping, Benchmarking, and Tuning Environment*"
        )

        with gr.Tab("On-Demand Receiver Pipeline"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Incoming Transcript Input")
                    transcript_input = gr.Textbox(
                        label="Incoming Source Transcript",
                        placeholder="Enter incoming sender transcript (e.g., मदद कीजिए।)",
                        lines=4,
                        value="मदद कीजिए।",
                    )
                    with gr.Row():
                        source_lang = gr.Dropdown(
                            label="Source Language",
                            choices=lang_choices,
                            value="hi",
                        )
                        target_lang = gr.Dropdown(
                            label="Target Language",
                            choices=lang_choices,
                            value="en",
                        )
                    with gr.Row():
                        beam_dropdown = gr.Dropdown(
                            label="Translation Decoding",
                            choices=beam_choices,
                            value="Greedy / Beam 1",
                        )

                    with gr.Accordion("Emergency Metadata (Optional)", open=False):
                        is_emergency_chk = gr.Checkbox(label="Is Emergency", value=False)
                        emergency_keywords_box = gr.Textbox(
                            label="Matched Emergency Keywords (comma separated)",
                            value="help, police, emergency",
                        )

                    translate_btn = gr.Button("Translate + Speak", variant="primary", size="lg")

                with gr.Column(scale=1):
                    gr.Markdown("### 2. Receiver Synthesized Speech & Text")
                    translated_output = gr.Textbox(label="Translated Text", lines=3)
                    audio_output = gr.Audio(label="Synthesized Audio Output", type="filepath")
                    audio_duration_display = gr.Textbox(label="Audio Duration", interactive=False)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 3. Structured Receiver Output")
                    summary_display = gr.Textbox(label="Receiver UI Display Card", lines=8, interactive=False)
                    with gr.Accordion("Structured JSON Payload (Android Contract)", open=False):
                        json_display = gr.Code(label="JSON Payload", language="json")

                with gr.Column(scale=1):
                    gr.Markdown("### 4. Stage Latencies & Real-Time Factor (RTF)")
                    latency_display = gr.Markdown(value="*Run pipeline to view stage-by-stage timings.*")

            with gr.Accordion("Debug / Benchmark Event Log", open=False):
                debug_log_box = gr.Textbox(
                    label="Append-Only Event Log",
                    lines=10,
                    interactive=False,
                    value="[00:00:00.000] Pipeline initialized and ready.",
                )

            # Connect Pipeline Trigger
            translate_btn.click(
                fn=process_translate_and_speak,
                inputs=[
                    transcript_input,
                    source_lang,
                    target_lang,
                    beam_dropdown,
                    is_emergency_chk,
                    emergency_keywords_box,
                ],
                outputs=[
                    translated_output,
                    audio_output,
                    audio_duration_display,
                    summary_display,
                    json_display,
                    latency_display,
                    debug_log_box,
                ],
            )

        with gr.Tab("Benchmarking & Performance Harness"):
            gr.Markdown("### Comprehensive Multi-Run Latency & RTF Benchmarking")
            with gr.Row():
                with gr.Column():
                    bench_custom_text = gr.Textbox(
                        label="Custom Benchmark Text (Leave blank to use all benchmark_cases.json)",
                        placeholder="Optional custom test phrase...",
                        lines=2,
                    )
                    with gr.Row():
                        bench_src = gr.Dropdown(label="Source Language", choices=lang_choices, value="hi")
                        bench_tgt = gr.Dropdown(label="Target Language", choices=lang_choices, value="en")
                    with gr.Row():
                        bench_warmup = gr.Slider(minimum=0, maximum=5, value=1, step=1, label="Warmup Runs")
                        bench_measured = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Measured Runs")

                    run_bench_btn = gr.Button("Run Benchmark", variant="primary")

                with gr.Column():
                    bench_results_display = gr.Markdown(value="*Click 'Run Benchmark' to execute runs.*")
                    with gr.Row():
                        json_file_down = gr.File(label="Download JSON Report")
                        md_file_down = gr.File(label="Download Markdown Report")

            run_bench_btn.click(
                fn=run_benchmark_ui,
                inputs=[bench_custom_text, bench_src, bench_tgt, bench_warmup, bench_measured],
                outputs=[bench_results_display, json_file_down, md_file_down],
            )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    # Share strictly False for offline security
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
