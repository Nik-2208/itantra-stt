# Indic-TTS Speech Synthesis Models Directory

Place your local Indic-TTS ONNX acoustic and vocoder models in this directory.

## Expected Directory Layout

```text
models/
└── tts/
    ├── en/
    │   ├── model.onnx (or acoustic.onnx + vocoder.onnx)
    │   ├── phones.json / config.json
    │   └── ...
    ├── hi/
    │   ├── model.onnx
    │   └── ...
    ├── gu/
    │   └── model.onnx
    └── ...
```

## Model Source
- **Indic-TTS Architecture**: [AI4Bharat Indic-TTS](https://github.com/AI4Bharat/Indic-TTS) (FastPitch / VITS acoustic + HiFi-GAN vocoder).
- **Target Execution**: ONNX Runtime with `CPUExecutionProvider`.
- **Audio Output**: 22,050 Hz Mono float32 PCM WAV.

## Fallback & Testing Behavior
When models are absent, `MockTTSAdapter` produces clean, harmonic waveforms for prototyping and UI benchmarking without runtime failures.
