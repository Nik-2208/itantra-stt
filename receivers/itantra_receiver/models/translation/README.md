# IndicTrans2 Translation Models Directory

Place your local IndicTrans2 ONNX translation models and tokenizers in this directory.

## Expected Directory Layout

You may supply either language-pair specific directories or a unified multilingual ONNX model:

### Option 1: Per-Language-Pair Structure (Recommended)
```text
models/
└── translation/
    ├── hi_en/
    │   ├── model.onnx (or encoder_model.onnx + decoder_model.onnx)
    │   ├── vocab.json / spm.model (IndicTrans2 SentencePiece model)
    │   └── config.json
    ├── en_hi/
    │   └── model.onnx
    ├── gu_en/
    │   └── model.onnx
    └── ...
```

### Option 2: Unified Multilingual Model
```text
models/
└── translation/
    └── multilingual/
        ├── model.onnx
        ├── vocab.json
        └── config.json
```

## Model Source
- **IndicTrans2 Architecture**: [AI4Bharat IndicTrans2](https://github.com/ai4bharat/IndicTrans2)
- **Target Execution**: ONNX Runtime with `CPUExecutionProvider`.
- **Supported Languages**: Hindi (`hi`), Gujarati (`gu`), Marathi (`mr`), Kannada (`kn`), Malayalam (`ml`), Tamil (`ta`), Telugu (`te`), Odia (`or`), Bengali (`bn`), English (`en`).

## Fallback & Testing Behavior
If no local ONNX files are placed in this directory, the receiver pipeline uses `MockTranslationAdapter` with deterministic dictionaries and predictable translation mocks so unit tests and UI prototypes remain functional.
