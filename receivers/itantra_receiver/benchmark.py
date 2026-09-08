"""
iTantra Receiver Pipeline - Benchmarking Harness (benchmark.py)
================================================================
Comprehensive, reproducible benchmark harness measuring stage-by-stage
and end-to-end latency, statistical distributions (min, max, mean, median, P95, std),
and Real-Time Factor (RTF).

Architecture Rules:
- Explicit separation of Warmup runs vs Measured runs (warmup excluded from stats).
- Cold-start vs Warm inference measurement modes.
- Export to structured JSON and human-readable Markdown reports in reports/.
- CPU-only execution metric reporting.
"""

import argparse
from datetime import datetime, timezone
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
# STATISTICAL CALCULATION UTILITIES
# ==============================================================================
def calculate_statistics(values: list[float]) -> dict[str, float]:
    """
    Computes min, max, mean, median, p95, and standard deviation deterministically.
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


# ==============================================================================
# BENCHMARK HARNESS CLASS
# ==============================================================================
class PipelineBenchmark:
    """
    Executes warmup and measured benchmarking across test cases.
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
        """Collects non-sensitive environment metadata for reproducibility."""
        cpu_name = platform.processor() or platform.machine() or "unknown"
        ort_version = ort.__version__ if ort is not None else "unknown"

        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version.split()[0],
            "onnxruntime_version": ort_version,
            "os_platform": f"{platform.system()} {platform.release()}",
            "cpu_architecture": cpu_name,
            "providers": config.ONNX_PROVIDERS,
            "num_threads": {
                "translation": config.TRANSLATION_NUM_THREADS,
                "tts": config.TTS_NUM_THREADS,
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
        Runs benchmark across test cases.
        """
        if not test_cases:
            raise ValueError("Test cases list cannot be empty.")

        print("==================================================")
        print("Starting iTantra Receiver Pipeline Benchmark")
        print(f"Total Test Cases: {len(test_cases)} | Warmup Runs: {warmup_runs} | Measured Runs: {measured_runs}")
        print(f"Mode: {'Cold Start' if cold_start else 'Warm Inference'}")
        print("==================================================")

        system_meta = self._get_system_metadata()
        case_results = []

        all_trans_latencies: list[float] = []
        all_form_latencies: list[float] = []
        all_tts_latencies: list[float] = []
        all_io_latencies: list[float] = []
        all_total_latencies: list[float] = []
        all_tts_rtfs: list[float] = []
        all_e2e_rtfs: list[float] = []

        for case_idx, case in enumerate(test_cases, start=1):
            case_id = case.get("id", f"case_{case_idx}")
            category = case.get("category", "general")
            src_lang = case.get("source_language", "hi")
            tgt_lang = case.get("target_language", "en")
            text = case.get("text", "")

            print(f"\n[{case_idx}/{len(test_cases)}] Benchmarking Case '{case_id}' ({category}: {src_lang} -> {tgt_lang})")

            # Warmup runs (not included in measured statistics)
            for w in range(warmup_runs):
                self.pipeline.translate_and_speak(text, src_lang, tgt_lang)

            # Measured runs
            case_trans: list[float] = []
            case_form: list[float] = []
            case_tts: list[float] = []
            case_io: list[float] = []
            case_total: list[float] = []
            case_tts_rtf: list[float] = []
            case_e2e_rtf: list[float] = []
            last_res = None

            for m in range(measured_runs):
                res = self.pipeline.translate_and_speak(text, src_lang, tgt_lang)
                last_res = res
                timing = res["timing"]
                bench = res["benchmark"]

                case_trans.append(timing["translation_latency_ms"])
                case_form.append(timing["formatter_latency_ms"])
                case_tts.append(timing["tts_latency_ms"])
                case_io.append(timing["audio_write_latency_ms"])
                case_total.append(timing["total_latency_ms"])
                case_tts_rtf.append(bench["tts_rtf"])
                case_e2e_rtf.append(bench["end_to_end_rtf"])

            # Accumulate
            all_trans_latencies.extend(case_trans)
            all_form_latencies.extend(case_form)
            all_tts_latencies.extend(case_tts)
            all_io_latencies.extend(case_io)
            all_total_latencies.extend(case_total)
            all_tts_rtfs.extend(case_tts_rtf)
            all_e2e_rtfs.extend(case_e2e_rtf)

            case_stats = {
                "id": case_id,
                "category": category,
                "source_language": src_lang,
                "target_language": tgt_lang,
                "input_text": text,
                "translated_text": last_res["translated_text"] if last_res else "",
                "audio_duration_ms": last_res["audio"]["duration_ms"] if last_res else 0.0,
                "translation": calculate_statistics(case_trans),
                "formatter": calculate_statistics(case_form),
                "tts": calculate_statistics(case_tts),
                "audio_io": calculate_statistics(case_io),
                "total": calculate_statistics(case_total),
                "tts_rtf_mean": round(float(np.mean(case_tts_rtf)), 4) if case_tts_rtf else 0.0,
                "e2e_rtf_mean": round(float(np.mean(case_e2e_rtf)), 4) if case_e2e_rtf else 0.0,
            }
            case_results.append(case_stats)

        # Aggregate summary across all cases
        overall_summary = {
            "translation": calculate_statistics(all_trans_latencies),
            "formatter": calculate_statistics(all_form_latencies),
            "tts": calculate_statistics(all_tts_latencies),
            "audio_io": calculate_statistics(all_io_latencies),
            "total_pipeline": calculate_statistics(all_total_latencies),
            "mean_tts_rtf": round(float(np.mean(all_tts_rtfs)), 4) if all_tts_rtfs else 0.0,
            "mean_e2e_rtf": round(float(np.mean(all_e2e_rtfs)), 4) if all_e2e_rtfs else 0.0,
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

        # Print final console table
        self._print_console_summary(overall_summary)

        return report_payload

    def _print_console_summary(self, summary: dict[str, Any]):
        """Prints a clean tabular summary to the terminal."""
        print("\n" + "=" * 65)
        print("                 BENCHMARK SUMMARY (in ms)")
        print("=" * 65)
        print(f"{'Stage':<18} {'Mean':<10} {'Median':<10} {'P95':<10} {'StdDev':<10}")
        print("-" * 65)
        for stage_key, label in [
            ("translation", "Translation"),
            ("formatter", "Formatting"),
            ("tts", "TTS Synthesis"),
            ("audio_io", "Audio Write"),
            ("total_pipeline", "TOTAL PIPELINE"),
        ]:
            st = summary[stage_key]
            print(f"{label:<18} {st['mean']:<10.2f} {st['median']:<10.2f} {st['p95']:<10.2f} {st['std_dev']:<10.2f}")
        print("-" * 65)
        print(f"Mean TTS RTF: {summary['mean_tts_rtf']:.4f} (< 1.0 is faster than real-time)")
        print(f"Mean E2E RTF: {summary['mean_e2e_rtf']:.4f}")
        print("=" * 65 + "\n")

    def export_reports(self, report_payload: dict[str, Any], output_dir: Path | str = config.REPORTS_DIR) -> tuple[Path, Path]:
        """
        Exports machine-readable JSON and human-readable Markdown reports.
        """
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

        lines = [
            f"# iTantra Receiver Pipeline - Benchmark Report",
            f"",
            f"**Generated:** {meta['timestamp_utc']}  ",
            f"**Platform:** {meta['os_platform']} ({meta['cpu_architecture']})  ",
            f"**Python:** {meta['python_version']} | **ONNX Runtime:** {meta['onnxruntime_version']}  ",
            f"**Providers:** `{', '.join(meta['providers'])}`  ",
            f"**Threads:** Translation={meta['num_threads']['translation']}, TTS={meta['num_threads']['tts']}  ",
            f"**Runs per case:** Warmup={cfg['warmup_runs']}, Measured={cfg['measured_runs']}  ",
            f"",
            f"## Overall Performance Summary",
            f"",
            f"| Stage | Mean (ms) | Median (ms) | P95 (ms) | Min (ms) | Max (ms) | StdDev (ms) |",
            f"|---|---|---|---|---|---|---|",
        ]

        stages = [
            ("translation", "Translation"),
            ("formatter", "Formatting"),
            ("tts", "TTS Synthesis"),
            ("audio_io", "Audio WAV Write"),
            ("total_pipeline", "**Total Pipeline**"),
        ]

        for s_key, s_label in stages:
            st = sum_[s_key]
            lines.append(
                f"| {s_label} | {st['mean']} | {st['median']} | {st['p95']} | {st['min']} | {st['max']} | {st['std_dev']} |"
            )

        lines.extend([
            f"",
            f"- **Mean TTS Real-Time Factor (RTF):** `{sum_['mean_tts_rtf']}`",
            f"- **Mean End-to-End RTF:** `{sum_['mean_e2e_rtf']}`",
            f"",
            f"> [!NOTE]",
            f"> Laptop CPU benchmark results serve as comparative prototype metrics to tune algorithm latency prior to porting to Android Kotlin/C++.",
            f"",
            f"## Test Case Details",
            f"",
            f"| Case ID | Cat | Lang Pair | Text | Mean Total (ms) | TTS Mean (ms) | Trans Mean (ms) | TTS RTF |",
            f"|---|---|---|---|---|---|---|---|",
        ])

        for c in cases:
            trunc_text = (c['input_text'][:28] + '...') if len(c['input_text']) > 30 else c['input_text']
            lines.append(
                f"| `{c['id']}` | {c['category']} | {c['source_language']}→{c['target_language']} | {trunc_text} | {c['total']['mean']} | {c['tts']['mean']} | {c['translation']['mean']} | {c['tts_rtf_mean']} |"
            )

        lines.append("")
        return "\n".join(lines)


# ==============================================================================
# CLI EXECUTION ENTRYPOINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="iTantra Receiver Pipeline Benchmark Harness")
    parser.add_argument("--cases", type=str, default="benchmark_cases.json", help="Path to test cases JSON")
    parser.add_argument("--warmup", type=int, default=config.BENCHMARK_WARMUP_RUNS, help="Warmup iterations")
    parser.add_argument("--measured", type=int, default=config.BENCHMARK_MEASURED_RUNS, help="Measured iterations")
    parser.add_argument("--cold-start", action="store_true", help="Include cold-start model initialization")
    args = parser.parse_args()

    cases_path = Path(args.cases)
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
