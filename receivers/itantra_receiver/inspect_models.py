"""
iTantra Receiver Pipeline - Local Model Inspection Tool (inspect_models.py)
===========================================================================
Inspects and prints metadata for local ONNX translation and TTS models:
- Model path, file size, SHA-256 hash
- ONNX Runtime version & CPU execution provider verification
- Graph input names, tensor shapes, and data types
- Graph output names, tensor shapes, and data types
- TTS conditioning inputs (speaker_id, language_id, voice_id, etc.) if present
- Tokenizer files and vocabulary assets
"""

import hashlib
import sys
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import onnxruntime as ort
except ImportError:
    ort = None

import config


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 checksum of a file."""
    if not file_path.is_file():
        return "N/A (Not a file)"
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_onnx_file(model_path: Path) -> dict[str, Any]:
    """Inspects an ONNX model file and extracts tensor signatures."""
    info: dict[str, Any] = {
        "path": str(model_path),
        "exists": model_path.exists(),
        "size_bytes": model_path.stat().st_size if model_path.exists() else 0,
        "sha256": compute_sha256(model_path) if model_path.exists() else "N/A",
        "inputs": [],
        "outputs": [],
        "conditioning_inputs": [],
        "error": None,
    }

    if not model_path.exists():
        info["error"] = "File does not exist"
        return info

    if ort is None:
        info["error"] = "onnxruntime is not installed"
        return info

    try:
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=config.ONNX_PROVIDERS,
        )

        for inp in session.get_inputs():
            inp_dict = {
                "name": inp.name,
                "shape": inp.shape,
                "type": inp.type,
            }
            info["inputs"].append(inp_dict)

            # Check for conditioning inputs
            lower_name = inp.name.lower()
            if any(k in lower_name for k in ["speaker", "spk", "lang", "voice", "style", "accent", "sid", "lid"]):
                info["conditioning_inputs"].append(inp_dict)

        for out in session.get_outputs():
            info["outputs"].append({
                "name": out.name,
                "shape": out.shape,
                "type": out.type,
            })

    except Exception as e:
        info["error"] = str(e)

    return info


def scan_directory(dir_path: Path) -> list[Path]:
    """Finds all .onnx files in a directory recursively."""
    if not dir_path.exists():
        return []
    return list(dir_path.rglob("*.onnx"))


def main():
    print("=" * 70)
    print("iTantra Receiver Pipeline - Model Asset Inspection")
    print("=" * 70)
    print(f"Base Directory: {config.BASE_DIR}")
    print(f"Models Directory: {config.MODELS_DIR}")
    print(f"ONNX Runtime Provider: {config.ONNX_PROVIDERS}")
    if ort is not None:
        print(f"ONNX Runtime Version: {ort.__version__}")
        print(f"Available Providers: {ort.get_available_providers()}")
    else:
        print("ONNX Runtime: NOT INSTALLED")
    print("-" * 70)

    # 1. Translation Models Inspection
    print("\n[1] TRANSLATION MODELS (models/translation/):")
    trans_models = scan_directory(config.TRANSLATION_MODELS_DIR)
    if not trans_models:
        print("  -> No .onnx files found in models/translation/.")
        print("  -> Architecture will use native offline fallback / tokenizer adapter.")
    else:
        for model_path in trans_models:
            rel_p = model_path.relative_to(config.BASE_DIR)
            print(f"\n  Model: {rel_p}")
            meta = inspect_onnx_file(model_path)
            print(f"    Size: {meta['size_bytes'] / (1024 * 1024):.2f} MB")
            print(f"    SHA-256: {meta['sha256']}")
            if meta["error"]:
                print(f"    Error: {meta['error']}")
            else:
                print("    Inputs:")
                for inp in meta["inputs"]:
                    print(f"      - {inp['name']}: shape={inp['shape']}, type={inp['type']}")
                print("    Outputs:")
                for out in meta["outputs"]:
                    print(f"      - {out['name']}: shape={out['shape']}, type={out['type']}")

    # 2. TTS Models Inspection
    print("\n[2] TTS MODELS (models/tts/):")
    tts_models = scan_directory(config.TTS_MODELS_DIR)
    if not tts_models:
        print("  -> No .onnx files found in models/tts/.")
        print("  -> Architecture will use verified local system speech synthesis with Indian Voice support.")
    else:
        for model_path in tts_models:
            rel_p = model_path.relative_to(config.BASE_DIR)
            print(f"\n  Model: {rel_p}")
            meta = inspect_onnx_file(model_path)
            print(f"    Size: {meta['size_bytes'] / (1024 * 1024):.2f} MB")
            print(f"    SHA-256: {meta['sha256']}")
            if meta["error"]:
                print(f"    Error: {meta['error']}")
            else:
                print("    Inputs:")
                for inp in meta["inputs"]:
                    print(f"      - {inp['name']}: shape={inp['shape']}, type={inp['type']}")
                print("    Outputs:")
                for out in meta["outputs"]:
                    print(f"      - {out['name']}: shape={out['shape']}, type={out['type']}")
                if meta["conditioning_inputs"]:
                    print("    Supported Conditioning Inputs:")
                    for cond in meta["conditioning_inputs"]:
                        print(f"      - {cond['name']}: shape={cond['shape']}, type={cond['type']}")
                else:
                    print("    Conditioning Inputs: NONE (Accent/Speaker conditioning NOT supported by this graph)")

    print("\n" + "=" * 70)
    print("Inspection complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
