"""
iTantra Receiver Pipeline - Interactive Prototyping & Benchmarking UI (app.py)
==============================================================================
Minimal, zero-network, local desktop testing UI using Gradio.
Launches strictly with share=False and CPU-only execution.
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

# Append-only debug log history
debug_log_history: list[str] = []


def add_log_entry(msg: str):
    """Appends timestamped log entry."""
    now_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    entry = f"[{now_str}] {msg}"
    debug_log_history.append(entry)
    if len(debug_log_history) > 150:
        debug_log_history.pop(0)


def get_voices_for_lang(target_lang: str) -> list[tuple[str, str]]:
    """Returns list of (display_label, voice_id) for the target language."""
    voices = config.VOICE_CONFIGS.get(target_lang, [])
    if not voices:
        return [("Default (Model-native)", "default")]
    return [
        (f"{v['name']} [{v['accent_name']}]", v["voice_id"]) for v in voices
    ]


def process_translate_and_speak(
    transcript: str,
    source_lang: str,
    target_lang: str,
    voice_choice_id: str,
    beam_choice: str,
    is_emergency: bool,
    emergency_keywords_str: str,
):
    """
    Executes on-demand translation, quality validation, output formatting,
    and speech synthesis.
    """
    if not transcript or not transcript.strip():
        return (
            "",  # translated text
            "⚠️ VALIDATION FAILED: Empty transcript",  # validation status
            "N/A",  # voice & accent info
            None,  # audio path
            "0.00 ms",  # audio duration
            "| Metric | Value |\n|---|---|\n| RMS | 0.00 |\n| Peak | 0.00 |\n| Clipping | 0 |",  # audio metrics
            "Error: Transcript cannot be empty.",  # summary card
            "{}",  # json payload
            "N/A",  # latency table
            "\n".join(debug_log_history),  # debug log
        )

    # Parse beam size
    beam_size = 1
    if "4" in beam_choice:
        beam_size = 4
    elif "8" in beam_choice:
        beam_size = 8

    pipeline.translator.set_beam_size(beam_size)

    # Emergency metadata
    emergency_meta = None
    if is_emergency:
        keywords = [k.strip() for k in emergency_keywords_str.split(",") if k.strip()]
        emergency_meta = {
            "is_emergency": True,
            "priority": "P1",
            "matched_keywords": keywords,
        }

    add_log_entry("Pipeline execution triggered by user.")
    add_log_entry(f"Source: {source_lang} -> Target: {target_lang} | Beam: {beam_size}")
    add_log_entry("Translation started...")

    try:
        result = pipeline.translate_and_speak(
            text=transcript,
            source_language=source_lang,
            target_language=target_lang,
            voice_id=voice_choice_id,
            is_final=True,
            emergency_result=emergency_meta,
        )

        timing = result["timing"]
        bench = result["benchmark"]
        audio_info = result.get("audio")
        tts_info = result.get("tts")
        trans_info = result.get("translation", {})
        formatted = result["formatted_output"]
        val_info = trans_info.get("validation", {"valid": True, "warnings": []})
        trans_success = trans_info.get("success", False)
        error_code = trans_info.get("error")
        leak_info = trans_info.get("leakage_info", {})

        add_log_entry(f"Translation completed in {timing['translation_ms']:.2f} ms")
        add_log_entry(f"Translation status: {'SUCCESS' if trans_success else f'FAILED ({error_code})'}")
        if trans_success and tts_info:
            add_log_entry(f"TTS completed in {timing['tts_latency_ms']:.2f} ms")
            add_log_entry(f"Audio written in {timing['audio_write_ms']:.2f} ms")
        else:
            add_log_entry(f"TTS execution: SKIPPED (Translation was not successful)")
        add_log_entry(f"Total pipeline completed in {timing['total_ms']:.2f} ms")

        # Format Validation & Diagnostic Display Card
        if not trans_success:
            if error_code == "SOURCE_LEAKAGE":
                val_card = (
                    f"### ❌ TRANSLATION FAILED\n\n"
                    f"**Reason:** `SOURCE_LEAKAGE`\n\n"
                    f"**Source Language:** `{source_lang}` | **Target Language:** `{target_lang}`\n\n"
                    f"**Source Text:** `{transcript}`\n\n"
                    f"**Model Raw Output:** `{trans_info.get('raw_output', 'N/A')}`\n\n"
                    f"**Action:** Translation was rejected and was **NOT** sent to TTS.\n\n"
                    f"> **Diagnostics:**\n"
                    f"> - Exact Match: `{leak_info.get('exact_match', False)}`\n"
                    f"> - Token Overlap: `{leak_info.get('token_overlap', 0.0):.2f}`\n"
                    f"> - Source Script Ratio: `{leak_info.get('source_script_ratio', 0.0):.2f}`\n"
                    f"> - Target Script Ratio: `{leak_info.get('target_script_ratio', 0.0):.2f}`\n"
                    f"> - Reason: *{leak_info.get('reason', 'Untranslated source text detected in target.')}*"
                )
            elif error_code == "UNSUPPORTED_LANGUAGE_PAIR":
                val_card = (
                    f"### ❌ TRANSLATION FAILED\n\n"
                    f"**Reason:** `UNSUPPORTED_LANGUAGE_PAIR` (`{source_lang}` → `{target_lang}`)\n\n"
                    f"**Action:** Language pair not supported by local model. No source fallback allowed. TTS skipped."
                )
            else:
                val_card = (
                    f"### ❌ TRANSLATION FAILED\n\n"
                    f"**Reason:** `{error_code or 'UNKNOWN_ERROR'}`\n\n"
                    f"**Action:** Translation rejected. TTS skipped."
                )
        elif val_info.get("valid", True):
            same_lang_note = " *(Same language passthrough)*" if "SOURCE_AND_TARGET_LANGUAGE_IDENTICAL" in trans_info.get("warnings", []) else ""
            val_card = f"✅ **VALID** (All quality & script checks passed){same_lang_note}"
        else:
            warnings_str = "\n".join([f"- ⚠️ {w}" for w in val_info.get("warnings", [])])
            val_card = f"⚠️ **QUALITY WARNINGS DETECTED:**\n{warnings_str}"

        # Voice & Accent Info
        if tts_info:
            voice_card = (
                f"**Voice ID:** `{tts_info['voice_id']}`\n\n"
                f"**Accent ID:** `{tts_info['accent_id']}`\n\n"
                f"**Speaker:** `{tts_info['speaker_id']}`\n\n"
                f"**Sample Rate:** `{tts_info['sample_rate']} Hz`"
            )
            audio_path = audio_info.get("path") if audio_info else None
            audio_duration_str = f"{audio_info.get('duration_ms', 0.0):.2f} ms" if audio_info else "0.00 ms"
            audio_metrics_card = (
                f"| Audio Metric | Value |\n"
                f"|---|---|\n"
                f"| Duration | **{tts_info['duration_ms']:.2f} ms** |\n"
                f"| Peak Amplitude | **{tts_info['peak_amplitude']:.4f}** |\n"
                f"| RMS Energy | **{tts_info['rms']:.4f}** |\n"
                f"| Clipping Samples | **{tts_info['clipping_samples']}** |\n"
                f"| TTS RTF | **{tts_info['rtf']:.4f}** |"
            )
        else:
            voice_card = "*(TTS skipped due to translation failure or empty output)*"
            audio_path = None
            audio_duration_str = "N/A (Skipped)"
            audio_metrics_card = "*TTS was not executed for this run.*"

        # Latency breakdown table
        tts_inf_str = f"`{timing['tts_inference_ms']:.2f} ms`" if tts_info else "*Skipped*"
        tts_pre_str = f"`{timing['tts_preprocess_ms']:.2f} ms`" if tts_info else "*Skipped*"
        tts_post_str = f"`{timing['tts_postprocess_ms']:.2f} ms`" if tts_info else "*Skipped*"
        io_str = f"`{timing['audio_write_ms']:.2f} ms`" if tts_info else "*Skipped*"
        e2e_rtf_str = f"`{bench['end_to_end_rtf']:.4f}`" if bench.get('end_to_end_rtf') is not None else "*N/A*"

        latency_table = (
            f"| Stage | Latency |\n"
            f"|---|---|\n"
            f"| 1. Text Normalization | `{timing['normalization_ms']:.2f} ms` |\n"
            f"| 2. Word-Safe Segmentation | `{timing['segmentation_ms']:.2f} ms` |\n"
            f"| 3. Tokenization | `{timing['tokenization_ms']:.2f} ms` |\n"
            f"| 4. Translation Inference | **`{timing['translation_ms']:.2f} ms`** |\n"
            f"| 5. Detokenization | `{timing['detokenization_ms']:.2f} ms` |\n"
            f"| 6. Translation Validation | `{timing['validation_ms']:.2f} ms` |\n"
            f"| 7. TTS Preprocessing | {tts_pre_str} |\n"
            f"| 8. TTS Model Inference | **{tts_inf_str}** |\n"
            f"| 9. TTS Postprocessing | {tts_post_str} |\n"
            f"| 10. Receiver Formatting | `{timing['formatting_ms']:.2f} ms` |\n"
            f"| 11. Audio WAV Write (I/O) | {io_str} |\n"
            f"| **TOTAL PIPELINE** | **`{timing['total_ms']:.2f} ms`** |\n\n"
            f"- **End-to-End RTF:** {e2e_rtf_str}"
        )

        summary_card = formatted["display"]["formatted_summary"]
        json_output = json.dumps(result, indent=2, ensure_ascii=False)

        return (
            trans_info.get("translated_text") or "",
            val_card,
            voice_card,
            audio_path,
            audio_duration_str,
            audio_metrics_card,
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
            f"❌ ERROR: {str(e)}",
            "N/A",
            None,
            "0.00 ms",
            "N/A",
            err_msg,
            "{}",
            "N/A",
            "\n".join(debug_log_history),
        )


def run_benchmark_ui(
    custom_text: str,
    source_lang: str,
    target_lang: str,
    use_eval_suite: bool,
    warmup_runs: int,
    measured_runs: int,
    cold_start: bool,
):
    """
    Executes benchmark harness from UI and generates report downloads.
    """
    cases = []
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
    elif use_eval_suite:
        eval_path = config.BASE_DIR / "evaluation_cases.json"
        if eval_path.exists():
            with open(eval_path, "r", encoding="utf-8") as f:
                cases = json.load(f)
    else:
        cases_path = config.BASE_DIR / "benchmark_cases.json"
        if cases_path.exists():
            with open(cases_path, "r", encoding="utf-8") as f:
                cases = json.load(f)

    if not cases:
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
        cold_start=cold_start,
    )
    json_path, md_path = benchmark_runner.export_reports(report_payload)

    sum_ = report_payload["overall_summary"]
    table_md = (
        f"### Benchmark Aggregate Results ({len(cases)} Cases, {measured_runs} Runs each)\n\n"
        f"| Pipeline Stage | Mean (ms) | Median (ms) | P95 (ms) | StdDev (ms) |\n"
        f"|---|---|---|---|---|\n"
        f"| 1. Normalization | {sum_['normalization']['mean']} | {sum_['normalization']['median']} | {sum_['normalization']['p95']} | {sum_['normalization']['std_dev']} |\n"
        f"| 2. Segmentation | {sum_['segmentation']['mean']} | {sum_['segmentation']['median']} | {sum_['segmentation']['p95']} | {sum_['segmentation']['std_dev']} |\n"
        f"| 3. Tokenization | {sum_['tokenization']['mean']} | {sum_['tokenization']['median']} | {sum_['tokenization']['p95']} | {sum_['tokenization']['std_dev']} |\n"
        f"| 4. Translation Inference | **{sum_['translation_inference']['mean']}** | **{sum_['translation_inference']['median']}** | **{sum_['translation_inference']['p95']}** | {sum_['translation_inference']['std_dev']} |\n"
        f"| 5. Detokenization | {sum_['detokenization']['mean']} | {sum_['detokenization']['median']} | {sum_['detokenization']['p95']} | {sum_['detokenization']['std_dev']} |\n"
        f"| 6. Validation | {sum_['validation']['mean']} | {sum_['validation']['median']} | {sum_['validation']['p95']} | {sum_['validation']['std_dev']} |\n"
        f"| 7. TTS Preprocess | {sum_['tts_preprocess']['mean']} | {sum_['tts_preprocess']['median']} | {sum_['tts_preprocess']['p95']} | {sum_['tts_preprocess']['std_dev']} |\n"
        f"| 8. TTS Model Inference | **{sum_['tts_inference']['mean']}** | **{sum_['tts_inference']['median']}** | **{sum_['tts_inference']['p95']}** | {sum_['tts_inference']['std_dev']} |\n"
        f"| 9. TTS Postprocess | {sum_['tts_postprocess']['mean']} | {sum_['tts_postprocess']['median']} | {sum_['tts_postprocess']['p95']} | {sum_['tts_postprocess']['std_dev']} |\n"
        f"| 10. Receiver Formatting | {sum_['formatting']['mean']} | {sum_['formatting']['median']} | {sum_['formatting']['p95']} | {sum_['formatting']['std_dev']} |\n"
        f"| 11. Audio File Write (I/O) | {sum_['audio_write_io']['mean']} | {sum_['audio_write_io']['median']} | {sum_['audio_write_io']['p95']} | {sum_['audio_write_io']['std_dev']} |\n"
        f"| **TOTAL PIPELINE** | **{sum_['total_pipeline']['mean']}** | **{sum_['total_pipeline']['median']}** | **{sum_['total_pipeline']['p95']}** | **{sum_['total_pipeline']['std_dev']}** |\n\n"
        f"- **Mean TTS RTF:** `{sum_['mean_tts_rtf']}`"
    )

    return table_md, str(json_path), str(md_path)


# ==============================================================================
# GRADIO INTERFACE CREATION (Section 29 - 34)
# ==============================================================================
def create_ui():
    lang_choices = config.SUPPORTED_LANGUAGES
    beam_choices = ["Greedy / Beam 1", "Beam 4", "Beam 8"]
    initial_voices = get_voices_for_lang("en")

    with gr.Blocks(title="iTantra Receiver Pipeline") as demo:
        gr.Markdown(
            "# iTantra Receiver Pipeline: On-Demand Translation & TTS Lab\n"
            "*Zero-Network CPU-Only Prototyping, Benchmarking, and Tuning Environment*"
        )

        with gr.Tab("On-Demand Receiver Pipeline"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Incoming Transcript Input")
                    transcript_input = gr.Textbox(
                        label="Incoming Source Transcript (Final Utterance)",
                        placeholder="Enter finalized sender transcript (e.g., मदद कीजिए।)",
                        lines=3,
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
                        voice_dropdown = gr.Dropdown(
                            label="Target Voice & Accent",
                            choices=[(label, vid) for label, vid in initial_voices],
                            value=initial_voices[0][1],
                        )
                        beam_dropdown = gr.Dropdown(
                            label="Translation Decoding",
                            choices=beam_choices,
                            value="Greedy / Beam 1",
                        )

                    # Dynamic update of voice options when target language changes
                    def update_voice_options(tgt):
                        v_choices = get_voices_for_lang(tgt)
                        return gr.Dropdown(choices=v_choices, value=v_choices[0][1])

                    target_lang.change(
                        fn=update_voice_options,
                        inputs=[target_lang],
                        outputs=[voice_dropdown],
                    )

                    with gr.Accordion("Emergency Metadata (Optional)", open=False):
                        is_emergency_chk = gr.Checkbox(label="Is Emergency", value=False)
                        emergency_keywords_box = gr.Textbox(
                            label="Matched Emergency Keywords",
                            value="help, police, emergency",
                        )

                    translate_btn = gr.Button("Translate + Synthesize Speech", variant="primary", size="lg")

                with gr.Column(scale=1):
                    gr.Markdown("### 2. Translation & Quality Validation")
                    translated_output = gr.Textbox(label="Translated Text (Sequence-Level)", lines=2)
                    validation_card = gr.Markdown(label="Translation Quality Validation", value="*Validation pending.*")
                    voice_accent_card = gr.Markdown(label="Active Voice & Accent", value="*Voice info pending.*")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 3. Synthesized Speech & Audio Quality")
                    audio_output = gr.Audio(label="Synthesized Audio (float32 Mono)", type="filepath")
                    audio_duration_display = gr.Textbox(label="Audio Duration", interactive=False)
                    audio_quality_display = gr.Markdown(value="*Audio metrics pending.*")

                with gr.Column(scale=1):
                    gr.Markdown("### 4. Stage Latencies & RTF")
                    latency_display = gr.Markdown(value="*Run pipeline to view micro-stage latency breakdown.*")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 5. Structured Receiver Output (Android Contract)")
                    summary_display = gr.Textbox(label="Receiver UI Display Card", lines=6, interactive=False)
                    with gr.Accordion("Structured JSON Payload (Section 42)", open=False):
                        json_display = gr.Code(label="JSON Payload", language="json")

                with gr.Column(scale=1):
                    gr.Markdown("### 6. Append-Only Debug Log")
                    debug_log_box = gr.Textbox(
                        label="Event Log",
                        lines=8,
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
                    voice_dropdown,
                    beam_dropdown,
                    is_emergency_chk,
                    emergency_keywords_box,
                ],
                outputs=[
                    translated_output,
                    validation_card,
                    voice_accent_card,
                    audio_output,
                    audio_duration_display,
                    audio_quality_display,
                    summary_display,
                    json_display,
                    latency_display,
                    debug_log_box,
                ],
            )

        with gr.Tab("Benchmarking & Reference Evaluation"):
            gr.Markdown("### Multi-Run Benchmarking & Reference Translation Evaluation")
            with gr.Row():
                with gr.Column():
                    bench_custom_text = gr.Textbox(
                        label="Custom Benchmark Text (Leave blank to use suite)",
                        placeholder="Optional custom test phrase...",
                        lines=2,
                    )
                    with gr.Row():
                        bench_src = gr.Dropdown(label="Source Language", choices=lang_choices, value="hi")
                        bench_tgt = gr.Dropdown(label="Target Language", choices=lang_choices, value="en")
                    with gr.Row():
                        bench_use_eval = gr.Checkbox(label="Use evaluation_cases.json (Reference Translations)", value=True)
                        bench_cold_start = gr.Checkbox(label="Cold Start Mode (Include Model Load)", value=False)
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
                inputs=[
                    bench_custom_text,
                    bench_src,
                    bench_tgt,
                    bench_use_eval,
                    bench_warmup,
                    bench_measured,
                    bench_cold_start,
                ],
                outputs=[bench_results_display, json_file_down, md_file_down],
            )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    # Share strictly False for offline security
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
