# iTantra Receiver Pipeline — Prototyping & Benchmarking Tool

A fully offline, CPU-only local desktop prototyping, benchmarking, and tuning tool for the **iTantra Receiver-Side Pipeline**:

$$\text{Incoming Source Transcript} \longrightarrow \text{Language Validation} \longrightarrow \text{On-Demand Translation} \longrightarrow \text{Structured Formatting} \longrightarrow \text{TTS Synthesis} \longrightarrow \text{Output}$$

Designed specifically to prototype, tune latency, and establish performance baselines on laptop CPUs before porting the algorithms and ONNX graphs to **Kotlin / C++ for the Android receiver application**.

---

## Key Design Principles

1. **On-Demand Translation Only**: Translation is never executed continuously on partial streams. It is triggered only upon explicit user request to conserve CPU and battery.
2. **Strictly CPU-Only Execution**: All ONNX sessions explicitly enforce `providers=["CPUExecutionProvider"]`. No CUDA/GPU acceleration is permitted.
3. **100% Offline & Network Safe**: Zero external API calls, zero telemetry, zero model auto-downloads.
4. **Clean Portability**: Plain data structures, explicit state, and zero framework-dependent magic to allow direct line-by-line translation to Kotlin/C++.
5. **Precise Micro-Benchmarking**: All stage timings use raw `time.perf_counter()` to compute statistical distributions (Mean, Median, P95, Min, Max, StdDev) and Real-Time Factor (RTF).

---

## File Structure

```text
itantra_receiver/
│
├── models/
│   ├── translation/
│   │   ├── README.md              # Instructions for IndicTrans2 ONNX placement
│   │   └── ...                    # hi_en/model.onnx, etc.
│   └── tts/
│       ├── README.md              # Instructions for Indic-TTS ONNX placement
│       └── ...                    # en/model.onnx, hi/model.onnx, etc.
│
├── config.py                      # Centralized tunable hyperparameters & language configs
├── translation.py                 # OnDemandTranslator with IndicTrans2 ONNX Adapter & chunker
├── tts.py                         # OfflineTTS with Indic-TTS ONNX Adapter & safe normalization
├── formatter.py                   # Model-independent receiver UI payload formatter
├── pipeline.py                    # ReceiverPipeline orchestrator with nanosecond timing
├── benchmark.py                   # Reusable benchmark harness with JSON/Markdown export
├── benchmark_cases.json           # Categorized benchmark test cases (short, medium, long)
├── app.py                         # Offline Gradio testing UI (share=False)
│
├── tests/
│   ├── test_config.py             # Config validation tests
│   ├── test_translation.py        # Translation & chunking unit tests
│   ├── test_tts.py                # TTS synthesis & audio integrity unit tests
│   ├── test_formatter.py          # Formatter data contract tests
│   ├── test_pipeline.py           # End-to-end pipeline tests
│   ├── test_benchmark.py          # Statistical aggregation & export tests
│   ├── test_app.py                # Gradio UI & event handler tests
│   └── test_offline_safety.py     # Socket monkey-patch verifying zero network calls
│
├── test_audio/                    # Reference audio and test outputs
├── reports/                       # Generated benchmark JSON & Markdown reports
└── README.md                      # Complete documentation and Android porting guide
```

---

## Required Models

This tool is designed to integrate locally supplied ONNX models from AI4Bharat:

### 1. Translation: AI4Bharat IndicTrans2
- **Source**: [https://github.com/ai4bharat/IndicTrans2](https://github.com/ai4bharat/IndicTrans2)
- **Supported Languages**: Hindi (`hi`), Gujarati (`gu`), Marathi (`mr`), Kannada (`kn`), Malayalam (`ml`), Tamil (`ta`), Telugu (`te`), Odia (`or`), Bengali (`bn`), English (`en`).
- **Placement**: Place models in `models/translation/<src>_<tgt>/model.onnx` or `models/translation/multilingual/model.onnx`.

### 2. Text-to-Speech: AI4Bharat Indic-TTS
- **Source**: [https://github.com/AI4Bharat/Indic-TTS](https://github.com/AI4Bharat/Indic-TTS)
- **Architecture**: FastPitch / VITS acoustic model + HiFi-GAN vocoder ONNX graphs.
- **Placement**: Place models in `models/tts/<lang>/model.onnx`.

> [!NOTE]
> **Deterministic Mock Fallback:** When model files have not yet been copied into `models/`, the pipeline automatically falls back to deterministic mock adapters (`MockTranslationAdapter` and `MockTTSAdapter`) so the tool and all tests remain 100% testable and operable out-of-the-box.

---

## Installation & Setup

### Requirements
- Python 3.10+ (tested on Python 3.13)
- Required packages: `numpy`, `soundfile`, `librosa`, `onnxruntime`, `gradio`, `matplotlib`

```bash
pip install numpy soundfile librosa onnxruntime gradio matplotlib
```

---

## Usage

### 1. Launch Interactive Prototyping UI
Launch the offline Gradio interface:

```bash
python app.py
```
Open your browser at `http://127.0.0.1:7860`.

**UI Features:**
- **On-Demand Receiver Pipeline**: Input transcript, select source/target languages, choose beam decoding size (Greedy/1, Beam 4, Beam 8), optionally set emergency flags, and click **Translate + Speak**.
- **Real-Time Outputs**: Instant playback of synthesized audio, display of duration, human-readable receiver card, raw Android JSON payload, and stage-by-stage latency table.
- **Append-Only Debug Log**: Live timestamped execution logs.
- **Benchmarking & Report Generator**: Run multi-iteration warmup and measured benchmarks, view statistics, and download timestamped JSON and Markdown reports.

---

### 2. Run Standalone Benchmarks
To run the automated benchmark harness from CLI across all test cases in `benchmark_cases.json`:

```bash
python benchmark.py --cases benchmark_cases.json --warmup 1 --measured 5
```

This outputs a full summary table and generates reports in `reports/`:
- `reports/benchmark_YYYYMMDD_HHMMSS.json`
- `reports/benchmark_YYYYMMDD_HHMMSS.md`

---

### 3. Run Test Suite
To run all unit tests and offline safety verifications:

```bash
python -m unittest discover tests
```

---

## Receiver Output Contract (Android Target)

The pipeline produces a structured output dictionary matching the target Kotlin data class:

```json
{
  "source_text": "मदद कीजिए।",
  "source_language": "hi",
  "translated_text": "Please help.",
  "target_language": "en",
  "audio": {
    "path": "C:/.../audio_1725838000_a1b2c3.wav",
    "sample_rate": 22050,
    "duration_ms": 600.0
  },
  "formatted_output": {
    "source": {
      "language": "hi",
      "language_name": "Hindi",
      "text": "मदद कीजिए।"
    },
    "translation": {
      "language": "en",
      "language_name": "English",
      "text": "Please help."
    },
    "emergency": {
      "is_emergency": false,
      "priority": "NORMAL",
      "matched_keywords": []
    },
    "display": {
      "headline": "Received (Hindi → English)",
      "body": "Please help.",
      "status": "NORMAL",
      "formatted_summary": "SOURCE (Hindi):\n\"मदद कीजिए।\"\n\nTRANSLATION (English):\n\"Please help.\"\n\nSTATUS: NORMAL"
    },
    "timestamp_iso": "2026-09-09T00:30:00.000000+00:00"
  },
  "benchmark": {
    "translation_ms": 18.42,
    "formatting_ms": 0.05,
    "tts_ms": 42.10,
    "audio_io_ms": 5.80,
    "total_ms": 66.37,
    "tts_rtf": 0.0702,
    "end_to_end_rtf": 0.1106
  }
}
```

---

## Latency & Real-Time Factor (RTF) Interpretation

- **Real-Time Factor (RTF)** is defined as:
  $$\text{RTF} = \frac{\text{Processing Time (ms)}}{\text{Synthesized Audio Duration (ms)}}$$
- **$\text{RTF} < 1.0$**: System generates audio faster than real-time (e.g. $\text{RTF} = 0.2$ means 1 second of speech is synthesized in 200 ms).
- **Comparative Prototype Numbers**: Laptop CPU execution benchmarks serve to tune chunking thresholds, beam sizes, and threading configurations. Android CPU execution numbers will depend on target ARM chipsets and ONNX Runtime NNAPI / CPU delegates.

---

## Porting to Android (Kotlin / C++)

1. **`config.py` $\rightarrow$ `ReceiverConfig.kt` / `config.h`**: Keep all hyperparameters as constants.
2. **`translation.py` $\rightarrow$ `IndicTrans2Adapter.kt`**: Bind to Android ONNX Runtime (`com.microsoft.onnxruntime`) with CPU configuration.
3. **`tts.py` $\rightarrow$ `IndicTTSAdapter.kt`**: Synthesize PCM float32 buffers and write to `AudioTrack` or `.wav`.
4. **`formatter.py` $\rightarrow$ `ReceiverFormatter.kt`**: Clean pure-Kotlin data formatting with zero ML dependencies.
5. **`pipeline.py` $\rightarrow$ `ReceiverPipeline.kt`**: Coordinate on-demand translation on user action (`TranslateButton.onClick`).
