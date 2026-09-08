"""
iTantra Receiver Pipeline - Benchmarking Harness (benchmark.py)
================================================================
Comprehensive, reproducible benchmark harness measuring stage-by-stage
and end-to-end latency, statistical distributions (min, max, mean, median, P95, std),
audio quality metrics, and Real-Time Factor (RTF).

Architecture Rules:
- Explicit separation of Cold-start vs Warm inference benchmarking.
- Separation of core model latency from file I/O and UI rendering overhead.
- Measurement of every micro-stage with time.perf_counter().
- Translation quality warning recording for every measured case.
- Reference evaluation comparison against evaluation_cases.json.
- Export to structured JSON and human-readable Markdown reports in reports/.
- CPUExecutionProvider provider tracking and thread logging.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

import config
from formatter import ReceiverFormatter
from pipeline import ReceiverPipeline
from translation import OnDemandTranslator, MockTranslationAdapter
from tts import OfflineTTS, MockTTSAdapter


# ==============================================================================
# STATISTICAL CALCULATION UTILITIES (Section 27)
# ==============================================================================
def calculate_statistics(values: list[float]) -> dict[str, float]:
    """
    Computes min, max, mean, median, p95 (nearest rank), and standard deviation.
    """
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "std_dev": 0.0,
        }

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    min_val = float(sorted_vals[0])
    max_val = float(sorted_vals[-1])
    mean_val = float(sum(sorted_vals) / n)

    # Median
    if n % 2 == 1:
        median_val = float(sorted_vals[n // 2])
    else:
        median_val = float((sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0)

    # 95th Percentile (nearest rank method)
    p95_idx = int(math.ceil(0.95 * n)) - 1
    p95_idx = max(0, min(p95_idx, n - 1))
    p95_val = float(sorted_vals[p95_idx])

    # Standard Deviation (sample)
    if n > 1:
        variance = sum((x - mean_val) ** 2 for x in sorted_vals) / (n - 1)
        std_dev = float(math.sqrt(variance))
    else:
        std_dev = 0.0

    return {
        "count": n,
        "min": round(min_val, 2),
        "max": round(max_val, 2),
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
        "p95": round(p95_val, 2),
        "std_dev": round(std_dev, 2),
    }


def compute_file_sha256(path: Path) -> str:
    """Computes SHA-256 checksum of a file, or returns 'unknown'."""
    if not path.is_file():
        return "unknown"
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ==============================================================================
# BENCHMARK HARNESS CLASS
# ==============================================================================
class PipelineBenchmark:
    """
    Executes cold/warm benchmarking, reference evaluations, and report generation.
    """

    def __init__(self, pipeline: Optional[ReceiverPipeline] = None):
        if pipeline is not None:
            self.pipeline = pipeline
        else:
            self.pipeline = ReceiverPipeline(
                translator=OnDemandTranslator(adapter=MockTranslationAdapter()),
                tts=OfflineTTS(adapter=MockTTSAdapter()),
                formatter=ReceiverFormatter(),
            )

    def _get_system_metadata(self) -> dict[str, Any]:
        """Collects environment metadata for reproducibility (Section 28)."""
        cpu_name = platform.processor() or platform.machine() or "unknown"
        ort_version = ort.__version__ if ort is not None else "unknown"

        # Check model hashes
        trans_model_path = config.TRANSLATION_MODELS_DIR / "model.onnx"
        tts_model_path = config.TTS_MODELS_DIR / "model.onnx"

        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version.split()[0],
            "os": f"{platform.system()} {platform.release()}",
            "cpu": cpu_name,
            "onnxruntime_version": ort_version,
            "providers": config.ONNX_PROVIDERS,
            "translation_model_sha256": compute_file_sha256(trans_model_path),
            "tts_model_sha256": compute_file_sha256(tts_model_path),
            "thread_configuration": {
                "onnx_intra_op_threads": config.ONNX_INTRA_OP_THREADS,
                "onnx_inter_op_threads": config.ONNX_INTER_OP_THREADS,
                "translation_num_threads": config.TRANSLATION_NUM_THREADS,
                "tts_num_threads": config.TTS_NUM_THREADS,
            },
            "context_settings": {
                "use_context": config.TRANSLATION_USE_CONTEXT,
                "context_sentence_count": config.TRANSLATION_CONTEXT_SENTENCE_COUNT,
            },
            "sample_rate_hz": config.TTS_SAMPLE_RATE,
        }

    def run(
        self,
        test_cases: list[dict[str, Any]],
        warmup_runs: int = config.BENCHMARK_WARMUP_RUNS,
        measured_runs: int = config.BENCHMARK_MEASURED_RUNS,
        cold_start: bool = False,
    ) -> dict[str, Any]:
        """
        Runs comprehensive benchmark across test cases.
        """
        if not test_cases:
            raise ValueError("Test cases list cannot be empty.")

        print("==================================================")
        print("Starting iTantra Receiver Pipeline Benchmark")
        print(f"Total Test Cases: {len(test_cases)} | Warmup Runs: {warmup_runs} | Measured Runs: {measured_runs}")
        print(f"Mode: {'Cold Start (Fresh Pipeline per Run)' if cold_start else 'Warm Inference'}")
        print("==================================================")

        system_meta = self._get_system_metadata()
        case_results = []

        all_norm: list[float] = []
        all_seg: list[float] = []
        all_tok: list[float] = []
        all_trans: list[float] = []
        all_detok: list[float] = []
        all_val: list[float] = []
        all_tts_pre: list[float] = []
        all_tts_inf: list[float] = []
        all_tts_post: list[float] = []
        all_form: list[float] = []
        all_io: list[float] = []
        all_total: list[float] = []
        all_tts_rtfs: list[float] = []

        total_cases_count = len(test_cases)
        total_translation_runs = 0
        successful_translation_runs = 0
        source_leakage_runs = 0
        unsupported_pair_runs = 0
        decoder_failure_runs = 0
        empty_output_runs = 0

        for case_idx, case in enumerate(test_cases, start=1):
            case_id = case.get("id", f"case_{case_idx}")
            category = case.get("category", "general")
            src_lang = case.get("source_language", "hi")
            tgt_lang = case.get("target_language", "en")
            voice_id = case.get("voice_id", None)
            text = case.get("source", case.get("text", ""))
            ref_translation = case.get("reference", "")

            print(f"\n[{case_idx}/{len(test_cases)}] Case '{case_id}' ({category}: {src_lang} -> {tgt_lang})")

            # Cold start initialization if requested
            if cold_start:
                t_cold_0 = time.perf_counter()
                pipeline_instance = ReceiverPipeline()
                t_cold_1 = time.perf_counter()
                cold_init_ms = (t_cold_1 - t_cold_0) * 1000.0
            else:
                pipeline_instance = self.pipeline
                cold_init_ms = 0.0

            # Warmup runs (strictly excluded from measured distribution)
            if not cold_start:
                for w in range(warmup_runs):
                    pipeline_instance.translate_and_speak(
                        text, src_lang, tgt_lang, voice_id=voice_id, is_final=True
                    )

            # Measured runs
            case_norm: list[float] = []
            case_seg: list[float] = []
            case_tok: list[float] = []
            case_trans: list[float] = []
            case_detok: list[float] = []
            case_val: list[float] = []
            case_tts_pre: list[float] = []
            case_tts_inf: list[float] = []
            case_tts_post: list[float] = []
            case_form: list[float] = []
            case_io: list[float] = []
            case_total: list[float] = []
            case_rtf: list[float] = []
            last_res = None

            for m in range(measured_runs):
                total_translation_runs += 1
                res = pipeline_instance.translate_and_speak(
                    text, src_lang, tgt_lang, voice_id=voice_id, is_final=True
                )
                last_res = res
                timing = res["timing"]
                trans_info = res.get("translation", {})
                tts_info = res.get("tts")

                if trans_info.get("success", False):
                    successful_translation_runs += 1
                else:
                    err = trans_info.get("error")
                    if err == "SOURCE_LEAKAGE":
                        source_leakage_runs += 1
                    elif err == "UNSUPPORTED_LANGUAGE_PAIR":
                        unsupported_pair_runs += 1
                    elif err == "EMPTY_OUTPUT":
                        empty_output_runs += 1
                    elif err == "DECODER_ERROR":
                        decoder_failure_runs += 1

                case_norm.append(timing["normalization_ms"])
                case_seg.append(timing["segmentation_ms"])
                case_tok.append(timing["tokenization_ms"])
                case_trans.append(timing["translation_ms"])
                case_detok.append(timing["detokenization_ms"])
                case_val.append(timing["validation_ms"])
                
                # Only record TTS timings if TTS actually ran (Section 30 requirement)
                if tts_info is not None:
                    case_tts_pre.append(timing["tts_preprocess_ms"])
                    case_tts_inf.append(timing["tts_inference_ms"])
                    case_tts_post.append(timing["tts_postprocess_ms"])
                    case_io.append(timing["audio_write_ms"])
                    case_rtf.append(tts_info.get("rtf", 0.0))

                case_form.append(timing["formatting_ms"])
                case_total.append(timing["total_ms"])

            # Accumulate overall distributions
            all_norm.extend(case_norm)
            all_seg.extend(case_seg)
            all_tok.extend(case_tok)
            all_trans.extend(case_trans)
            all_detok.extend(case_detok)
            all_val.extend(case_val)
            all_tts_pre.extend(case_tts_pre)
            all_tts_inf.extend(case_tts_inf)
            all_tts_post.extend(case_tts_post)
            all_form.extend(case_form)
            all_io.extend(case_io)
            all_total.extend(case_total)
            all_tts_rtfs.extend(case_rtf)

            # Quality + Latency record for this case (Section 25)
            last_tts = last_res.get("tts") if last_res and last_res.get("tts") else {}
            last_trans = last_res.get("translation") if last_res else {}

            case_stats = {
                "id": case_id,
                "category": category,
                "source_language": src_lang,
                "target_language": tgt_lang,
                "source_text": text,
                "reference_translation": ref_translation if ref_translation else "N/A",
                "translation_success": last_trans.get("success", False),
                "translation_error": last_trans.get("error"),
                "translated_text": last_trans.get("translated_text", "") or "",
                "is_source_leakage": last_trans.get("is_leakage", False),
                "beam_size": last_trans.get("beam_size", 1),
                "voice_id": last_tts.get("voice_id", "default") if last_tts else None,
                "accent_id": last_tts.get("accent_id", "model_native") if last_tts else None,
                "speaker_id": last_tts.get("speaker_id", "0") if last_tts else None,
                "audio_duration_ms": last_tts.get("duration_ms", 0.0) if last_tts else None,
                "tts_rtf": last_tts.get("rtf", 0.0) if last_tts else None,
                "audio_peak_amplitude": last_tts.get("peak_amplitude", 0.0) if last_tts else None,
                "audio_rms": last_tts.get("rms", 0.0) if last_tts else None,
                "audio_clipping_samples": last_tts.get("clipping_samples", 0) if last_tts else None,
                "cold_initialization_ms": round(cold_init_ms, 2),
                "translation_validation": last_trans.get("validation", {}),
                "leakage_info": last_trans.get("leakage_info", {}),
                "timing_statistics": {
                    "normalization": calculate_statistics(case_norm),
                    "segmentation": calculate_statistics(case_seg),
                    "tokenization": calculate_statistics(case_tok),
                    "translation_inference": calculate_statistics(case_trans),
                    "detokenization": calculate_statistics(case_detok),
                    "validation": calculate_statistics(case_val),
                    "tts_preprocess": calculate_statistics(case_tts_pre) if case_tts_pre else None,
                    "tts_inference": calculate_statistics(case_tts_inf) if case_tts_inf else None,
                    "tts_postprocess": calculate_statistics(case_tts_post) if case_tts_post else None,
                    "formatting": calculate_statistics(case_form),
                    "audio_write_io": calculate_statistics(case_io) if case_io else None,
                    "total_pipeline": calculate_statistics(case_total),
                },
                # Backward compatibility keys
                "translation": calculate_statistics(case_trans),
                "formatter": calculate_statistics(case_form),
                "tts": calculate_statistics(case_tts_inf) if case_tts_inf else None,
                "audio_io": calculate_statistics(case_io) if case_io else None,
                "total": calculate_statistics(case_total),
                "tts_rtf_mean": round(float(np.mean(case_rtf)), 4) if case_rtf else None,
            }
            case_results.append(case_stats)

        trans_success_rate = (successful_translation_runs / total_translation_runs * 100.0) if total_translation_runs > 0 else 0.0
        source_leakage_rate = (source_leakage_runs / total_translation_runs * 100.0) if total_translation_runs > 0 else 0.0
        decoder_failure_rate = (decoder_failure_runs / total_translation_runs * 100.0) if total_translation_runs > 0 else 0.0
        unsupported_pair_rate = (unsupported_pair_runs / total_translation_runs * 100.0) if total_translation_runs > 0 else 0.0
        empty_output_rate = (empty_output_runs / total_translation_runs * 100.0) if total_translation_runs > 0 else 0.0

        overall_summary = {
            "metrics": {
                "total_runs": total_translation_runs,
                "translation_success_count": successful_translation_runs,
                "translation_success_rate": round(trans_success_rate, 2),
                "source_leakage_count": source_leakage_runs,
                "source_leakage_rate": round(source_leakage_rate, 2),
                "decoder_failure_count": decoder_failure_runs,
                "decoder_failure_rate": round(decoder_failure_rate, 2),
                "unsupported_pair_count": unsupported_pair_runs,
                "unsupported_pair_rate": round(unsupported_pair_rate, 2),
                "empty_output_count": empty_output_runs,
                "empty_output_rate": round(empty_output_rate, 2),
            },
            "normalization": calculate_statistics(all_norm),
            "segmentation": calculate_statistics(all_seg),
            "tokenization": calculate_statistics(all_tok),
            "translation_inference": calculate_statistics(all_trans),
            "detokenization": calculate_statistics(all_detok),
            "validation": calculate_statistics(all_val),
            "tts_preprocess": calculate_statistics(all_tts_pre),
            "tts_inference": calculate_statistics(all_tts_inf),
            "tts_postprocess": calculate_statistics(all_tts_post),
            "formatting": calculate_statistics(all_form),
            "audio_write_io": calculate_statistics(all_io),
            "total_pipeline": calculate_statistics(all_total),
            # Backward-compatible keys
            "translation": calculate_statistics(all_trans),
            "formatter": calculate_statistics(all_form),
            "tts": calculate_statistics(all_tts_inf),
            "audio_io": calculate_statistics(all_io),
            "mean_tts_rtf": round(float(np.mean(all_tts_rtfs)), 4) if all_tts_rtfs else 0.0,
            "mean_e2e_rtf": round(
                float(np.mean(all_total) / (np.mean(all_tts_inf) * 2 + 100)), 4
            ) if all_total and all_tts_inf else 0.0,
        }

        report_payload = {
            "metadata": system_meta,
            "config": {
                "warmup_runs": warmup_runs,
                "measured_runs": measured_runs,
                "cold_start": cold_start,
                "total_cases": len(test_cases),
            },
            "overall_summary": overall_summary,
            "cases": case_results,
        }

        self._print_console_summary(overall_summary)
        return report_payload

    def _print_console_summary(self, summary: dict[str, Any]):
        """Prints tabular summary separating core model latency from I/O."""
        print("\n" + "=" * 75)
        print("               iTantra Receiver Pipeline Benchmark Summary")
        print("=" * 75)
        metrics = summary.get("metrics", {})
        if metrics:
            print(f"Total Runs: {metrics.get('total_runs', 0)}")
            print(f"Translation Success Rate: {metrics.get('translation_success_rate', 0.0)}% ({metrics.get('translation_success_count', 0)}/{metrics.get('total_runs', 0)})")
            print(f"Source Leakage Rate:      {metrics.get('source_leakage_rate', 0.0)}% ({metrics.get('source_leakage_count', 0)})")
            print(f"Decoder Failure Rate:     {metrics.get('decoder_failure_rate', 0.0)}% ({metrics.get('decoder_failure_count', 0)})")
            print(f"Unsupported Pair Rate:    {metrics.get('unsupported_pair_rate', 0.0)}% ({metrics.get('unsupported_pair_count', 0)})")
            print(f"Empty Output Rate:        {metrics.get('empty_output_rate', 0.0)}% ({metrics.get('empty_output_count', 0)})")
            print("-" * 75)
        print(f"{'Stage':<25} {'Mean (ms)':<12} {'Median':<10} {'P95':<10} {'StdDev':<10}")
        print("-" * 75)
        stages = [
            ("normalization", "1. Normalization"),
            ("segmentation", "2. Segmentation"),
            ("tokenization", "3. Tokenization"),
            ("translation_inference", "4. Translation Inf"),
            ("detokenization", "5. Detokenization"),
            ("validation", "6. Validation"),
            ("tts_preprocess", "7. TTS Preprocess"),
            ("tts_inference", "8. TTS Inference"),
            ("tts_postprocess", "9. TTS Postprocess"),
            ("formatting", "10. Formatting"),
            ("audio_write_io", "11. Audio File I/O"),
            ("total_pipeline", "TOTAL PIPELINE"),
        ]
        for key, label in stages:
            st = summary[key]
            print(f"{label:<25} {st['mean']:<12.2f} {st['median']:<10.2f} {st['p95']:<10.2f} {st['std_dev']:<10.2f}")
        print("-" * 75)
        print(f"Mean TTS RTF: {summary['mean_tts_rtf']:.4f}")
        print("=" * 75 + "\n")

    def export_reports(
        self,
        report_payload: dict[str, Any],
        output_dir: Path | str = config.REPORTS_DIR,
    ) -> tuple[Path, Path]:
        """Exports machine-readable JSON and Markdown reports."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_path = out_dir / f"benchmark_{timestamp_str}.json"
        md_path = out_dir / f"benchmark_{timestamp_str}.md"

        # 1. Write JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2, ensure_ascii=False)

        # 2. Write Markdown
        md_content = self._generate_markdown(report_payload)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"[Benchmark] Reports saved:\n  - JSON: {json_path}\n  - Markdown: {md_path}")
        return json_path, md_path

    def _generate_markdown(self, payload: dict[str, Any]) -> str:
        meta = payload["metadata"]
        cfg = payload["config"]
        sum_ = payload["overall_summary"]
        cases = payload["cases"]
        threads = meta["thread_configuration"]
        metrics = sum_.get("metrics", {})

        lines = [
            f"# iTantra Receiver Pipeline - Benchmark Report",
            f"",
            f"**Timestamp (UTC):** `{meta['timestamp_utc']}`  ",
            f"**Operating System:** {meta['os']}  ",
            f"**CPU Architecture:** {meta['cpu']}  ",
            f"**Python Version:** {meta['python_version']} | **ONNX Runtime:** {meta['onnxruntime_version']}  ",
            f"**Active Execution Providers:** `{', '.join(meta['providers'])}`  ",
            f"**Model SHA-256 (Translation):** `{meta['translation_model_sha256']}`  ",
            f"**Model SHA-256 (TTS):** `{meta['tts_model_sha256']}`  ",
            f"**Threads:** Intra={threads['onnx_intra_op_threads']}, Inter={threads['onnx_inter_op_threads']}, Trans={threads['translation_num_threads']}, TTS={threads['tts_num_threads']}  ",
            f"**Benchmark Mode:** {'Cold Start' if cfg['cold_start'] else 'Warm Inference'} (Warmup={cfg['warmup_runs']}, Measured={cfg['measured_runs']})  ",
            f"",
            f"## Reliability & Translation Quality Gates",
            f"",
            f"- **Translation Success Rate:** `{metrics.get('translation_success_rate', 0.0)}%` ({metrics.get('translation_success_count', 0)}/{metrics.get('total_runs', 0)})",
            f"- **Source Leakage Rate:** `{metrics.get('source_leakage_rate', 0.0)}%` ({metrics.get('source_leakage_count', 0)})",
            f"- **Decoder Failure Rate:** `{metrics.get('decoder_failure_rate', 0.0)}%` ({metrics.get('decoder_failure_count', 0)})",
            f"- **Unsupported Language Pair Rate:** `{metrics.get('unsupported_pair_rate', 0.0)}%` ({metrics.get('unsupported_pair_count', 0)})",
            f"- **Empty Output Rate:** `{metrics.get('empty_output_rate', 0.0)}%` ({metrics.get('empty_output_count', 0)})",
            f"",
            f"## Micro-Stage Latency Breakdown (ms)",
            f"",
            f"| Pipeline Stage | Mean | Median | P95 | Min | Max | StdDev |",
            f"|---|---|---|---|---|---|---|",
        ]

        stages = [
            ("normalization", "Text Normalization"),
            ("segmentation", "Word-Safe Segmentation"),
            ("tokenization", "Tokenization"),
            ("translation_inference", "**Translation Inference**"),
            ("detokenization", "Detokenization"),
            ("validation", "Translation Validation"),
            ("tts_preprocess", "Pronunciation / TTS Preprocess"),
            ("tts_inference", "**TTS Model Inference**"),
            ("tts_postprocess", "TTS Postprocess"),
            ("formatting", "Receiver Formatting"),
            ("audio_write_io", "Audio Disk Write (WAV)"),
            ("total_pipeline", "**TOTAL PIPELINE**"),
        ]

        for s_key, s_label in stages:
            st = sum_[s_key]
            lines.append(
                f"| {s_label} | {st['mean']} | {st['median']} | {st['p95']} | {st['min']} | {st['max']} | {st['std_dev']} |"
            )

        lines.extend([
            f"",
            f"- **Mean TTS Real-Time Factor (RTF):** `{sum_['mean_tts_rtf']}`",
            f"",
            f"## Individual Test Case & Quality Validation Details",
            f"",
            f"| ID | Lang Pair | Source Text | Translation Output | Valid | Trans (ms) | TTS (ms) | Total (ms) | RTF |",
            f"|---|---|---|---|---|---|---|---|---|",
        ])

        for c in cases:
            src_snippet = (c['source_text'][:25] + '...') if len(c['source_text']) > 28 else c['source_text']
            tgt_snippet = (c['translated_text'][:25] + '...') if len(c['translated_text']) > 28 else c['translated_text']
            val_status = "PASS" if c['translation_validation'].get('valid', True) else "WARN"
            lines.append(
                f"| `{c['id']}` | {c['source_language']}→{c['target_language']} | {src_snippet} | {tgt_snippet} | `{val_status}` | {c['timing_statistics']['translation_inference']['mean']} | {c['timing_statistics']['tts_inference']['mean']} | {c['timing_statistics']['total_pipeline']['mean']} | {c['tts_rtf']} |"
            )

        # Reference evaluation section if available
        has_refs = any(c.get("reference_translation") and c.get("reference_translation") != "N/A" for c in cases)
        if has_refs:
            lines.extend([
                f"",
                f"## Reference Translation Comparison",
                f"",
                f"| ID | Source | Reference | System Output |",
                f"|---|---|---|---|",
            ])
            for c in cases:
                if c.get("reference_translation") and c.get("reference_translation") != "N/A":
                    lines.append(
                        f"| `{c['id']}` | {c['source_text']} | {c['reference_translation']} | {c['translated_text']} |"
                    )

        lines.append("")
        return "\n".join(lines)


# ==============================================================================
# CLI EXECUTION ENTRYPOINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="iTantra Receiver Pipeline Benchmark Harness")
    parser.add_argument("--cases", type=str, default="benchmark_cases.json", help="Path to test cases JSON")
    parser.add_argument("--eval", action="store_true", help="Run reference evaluation using evaluation_cases.json")
    parser.add_argument("--warmup", type=int, default=config.BENCHMARK_WARMUP_RUNS, help="Warmup iterations")
    parser.add_argument("--measured", type=int, default=config.BENCHMARK_MEASURED_RUNS, help="Measured iterations")
    parser.add_argument("--cold-start", action="store_true", help="Include cold-start model initialization")
    args = parser.parse_args()

    cases_file = "evaluation_cases.json" if args.eval else args.cases
    cases_path = Path(cases_file)
    if not cases_path.exists():
        print(f"Error: Cases file not found at {cases_path}")
        sys.exit(1)

    with open(cases_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    benchmark = PipelineBenchmark()
    report = benchmark.run(
        test_cases=cases,
        warmup_runs=args.warmup,
        measured_runs=args.measured,
        cold_start=args.cold_start,
    )
    benchmark.export_reports(report)


if __name__ == "__main__":
    main()
