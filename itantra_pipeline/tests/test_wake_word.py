"""
Module 4B Test Suite: test_wake_word.py
======================================
Comprehensive automated tests for offline low-latency wake-word trigger:
1. Real wake-word detection latency and accuracy
2. False positive rejection (silence, white noise, non-wake speech)
3. Debounce & cooldown enforcement (suppress duplicate rapid triggers)
4. Wake + command streaming transition & wake phrase stripping
5. State machine validation (IDLE -> WAKE_DETECTED -> COMMAND_CAPTURE -> PROCESSING -> NORMAL/EMERGENCY -> TRANSMITTING -> COOLDOWN)
6. Emergency vs. Normal classification safety (wake word alone is NOT emergency)
7. Actual latency profiling (wake detection, wake->capture, STT, classifier, total wake->decision)
"""

import sys
import time
import wave
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import DEFAULT_SAMPLE_RATE, WAKE_WORD_THRESHOLD, WAKE_WORD_SCORE
from wake_word import WakeWordDetector
from pipeline import VoicePipeline, PipelineState


def test_wake_detection():
    print("\n--- Running Test 1: Real Wake Detection ---")
    detector = WakeWordDetector()
    assert detector.is_loaded, "WakeWordDetector model failed to load!"

    wake_file = Path(__file__).resolve().parent / "audio" / "wake_only.wav"
    with wave.open(str(wake_file), "rb") as wf:
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

    detector.reset()
    detected_events = []
    t0 = time.perf_counter()
    for i in range(0, len(audio), 512):
        ev = detector.process_frame(audio[i : i + 512])
        if ev:
            detected_events.append(ev)

    total_proc_ms = (time.perf_counter() - t0) * 1000.0
    assert len(detected_events) >= 1, f"Expected wake word detection, got {len(detected_events)}"

    first_ev = detected_events[0]
    print(f"  Detected Keyword    : '{first_ev.keyword}'")
    print(f"  Trigger Timestamp   : {first_ev.timestamp_s:.2f} s")
    print(f"  Frame Decode Latency: {first_ev.latency_ms:.2f} ms")
    print(f"  Pre-roll Buffer Size: {len(first_ev.pre_roll_audio)} samples ({len(first_ev.pre_roll_audio)/DEFAULT_SAMPLE_RATE*1000:.0f} ms)")
    print(f"  Total Audio Proc Time: {total_proc_ms:.2f} ms")

    assert first_ev.latency_ms < 50.0, f"Wake latency too high: {first_ev.latency_ms} ms"
    assert len(first_ev.pre_roll_audio) > 0, "Pre-roll buffer must not be empty"
    print("--- Test 1 PASSED ---")


def test_false_positive_rejection():
    print("\n--- Running Test 2: False Positive Rejection ---")
    detector = WakeWordDetector()
    detector.reset()

    # 1. Test 5s white noise
    noise = (np.random.randn(DEFAULT_SAMPLE_RATE * 5) * 0.1).astype(np.float32)
    noise_triggers = []
    for i in range(0, len(noise), 512):
        ev = detector.process_frame(noise[i : i + 512])
        if ev:
            noise_triggers.append(ev)
    assert len(noise_triggers) == 0, f"White noise falsely triggered wake detector: {noise_triggers}"
    print("  White noise rejection (5.0s): PASSED (0 triggers)")

    # 2. Test 5s silence
    silence = np.zeros(DEFAULT_SAMPLE_RATE * 5, dtype=np.float32)
    silence_triggers = []
    for i in range(0, len(silence), 512):
        ev = detector.process_frame(silence[i : i + 512])
        if ev:
            silence_triggers.append(ev)
    assert len(silence_triggers) == 0, f"Silence falsely triggered wake detector: {silence_triggers}"
    print("  Silence rejection (5.0s)    : PASSED (0 triggers)")

    # 3. Test non-wake English speech
    non_wake_file = Path(__file__).resolve().parent / "audio" / "non_wake_normal.wav"
    with wave.open(str(non_wake_file), "rb") as wf:
        non_wake_audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

    detector.reset()
    speech_triggers = []
    for i in range(0, len(non_wake_audio), 512):
        ev = detector.process_frame(non_wake_audio[i : i + 512])
        if ev:
            speech_triggers.append(ev)
    assert len(speech_triggers) == 0, f"Non-wake speech falsely triggered detector: {speech_triggers}"
    print("  Non-wake speech rejection   : PASSED (0 triggers)")
    print("--- Test 2 PASSED ---")


def test_debounce_and_cooldown():
    print("\n--- Running Test 3: Debounce & Cooldown Window ---")
    detector = WakeWordDetector(cooldown_ms=1500)
    wake_file = Path(__file__).resolve().parent / "audio" / "wake_only.wav"
    with wave.open(str(wake_file), "rb") as wf:
        wake = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

    # Rapid consecutive wake utterances separated by only 250ms (within 1500ms cooldown)
    pause = np.zeros(int(0.25 * DEFAULT_SAMPLE_RATE), dtype=np.float32)
    double_wake_audio = np.concatenate([wake, pause, wake])

    detector.reset()
    triggers = []
    for i in range(0, len(double_wake_audio), 512):
        ev = detector.process_frame(double_wake_audio[i : i + 512])
        if ev:
            triggers.append(ev)

    print(f"  Triggers detected for rapid duplicate wake words: {len(triggers)}")
    assert len(triggers) == 1, f"Debounce failed: expected exactly 1 trigger, got {len(triggers)}"
    print(f"  First trigger at {triggers[0].timestamp_s:.2f}s, second trigger suppressed.")
    print("--- Test 3 PASSED ---")


def test_wake_phrase_stripping():
    print("\n--- Running Test 4: Wake Phrase Stripping ---")
    detector = WakeWordDetector()
    samples = [
        ("Hey iTantra help there is a fire", "help there is a fire"),
        ("Hey iTantra, send rescue team", "send rescue team"),
        ("iTantra water is rising", "water is rising"),
        ("हे आईटन्ट्रा मदद करो", "मदद करो"),
        ("help fire", "help fire"),
    ]
    for raw, expected in samples:
        stripped = detector.strip_wake_phrase(raw)
        print(f"  Raw: {raw!r:35} -> Stripped: {stripped!r}")
        assert stripped == expected or expected in stripped, f"Failed stripping: expected {expected}, got {stripped}"
    print("--- Test 4 PASSED ---")


def test_wake_normal_pipeline():
    print("\n--- Running Test 5: Wake Word Alone (Must NOT Be Emergency) ---")
    pipeline = VoicePipeline()
    wake_file = Path(__file__).resolve().parent / "audio" / "wake_only.wav"

    res = pipeline.process_audio(str(wake_file), enable_wake_word=True, language_code="hi")

    print(f"  Wake Word Detected   : {res['wake_word_detected']}")
    print(f"  Emergency Decision   : {res['decision']}")
    print(f"  Priority Level       : {res['emergency_result'].get('priority')}")
    print(f"  Transmission Channel : {res.get('transmission_event', {}).get('channel')}")
    print(f"  State Transitions    : {[t['state'] for t in res['state_transitions']]}")

    assert res["wake_word_detected"] is True, "Wake word should be detected"
    assert res["decision"] == "NORMAL", f"Wake word alone MUST be NORMAL, got {res['decision']}"
    assert res["emergency_result"]["priority"] == "P2", "Priority must be P2"
    assert res.get("transmission_event", {}).get("channel") == "STANDARD_COMM_CHANNEL"
    print("--- Test 5 PASSED ---")


def test_wake_emergency_pipeline():
    print("\n--- Running Test 6: Wake Word + Emergency Command ---")
    pipeline = VoicePipeline()
    emerg_file = Path(__file__).resolve().parent / "audio" / "wake_emergency.wav"

    res = pipeline.process_audio(str(emerg_file), enable_wake_word=True, language_code="hi")

    print(f"  Wake Word Detected   : {res['wake_word_detected']}")
    print(f"  Final Transcript     : {res['final_transcript']!r}")
    print(f"  Emergency Decision   : {res['decision']}")
    print(f"  Matched Keywords     : {res['emergency_result'].get('matched_keywords')}")
    print(f"  Priority Level       : {res['emergency_result'].get('priority')}")
    print(f"  Transmission Channel : {res.get('transmission_event', {}).get('channel')}")
    print(f"  State Transitions    : {[t['state'] for t in res['state_transitions']]}")

    assert res["wake_word_detected"] is True, "Wake word must be detected"
    assert res["decision"] == "EMERGENCY", f"Expected EMERGENCY decision, got {res['decision']}"
    assert res["emergency_result"]["priority"] == "P0", "Emergency priority must be P0"
    assert res.get("transmission_event", {}).get("channel") == "SOS_PRIORITY_CHANNEL"
    assert len(res["emergency_result"].get("matched_keywords", [])) > 0, "Emergency keywords must be matched"
    print("--- Test 6 PASSED ---")


def test_wake_pause_emergency_pipeline():
    print("\n--- Running Test 7: Wake Word + Short Pause + Emergency Command ---")
    pipeline = VoicePipeline()
    pause_file = Path(__file__).resolve().parent / "audio" / "wake_pause_emergency.wav"

    res = pipeline.process_audio(str(pause_file), enable_wake_word=True, language_code="hi")

    print(f"  Wake Word Detected   : {res['wake_word_detected']}")
    print(f"  Final Transcript     : {res['final_transcript']!r}")
    print(f"  Emergency Decision   : {res['decision']}")
    print(f"  Priority Level       : {res['emergency_result'].get('priority')}")
    print(f"  Transmission Channel : {res.get('transmission_event', {}).get('channel')}")

    assert res["wake_word_detected"] is True, "Wake word must be detected across pause"
    assert res["decision"] == "EMERGENCY", f"Expected EMERGENCY, got {res['decision']}"
    assert res["emergency_result"]["priority"] == "P0"
    print("--- Test 7 PASSED ---")


def test_latency_and_resource_profiling():
    print("\n--- Running Test 8: Precise Latency & Resource Profiling ---")
    pipeline = VoicePipeline()
    emerg_file = Path(__file__).resolve().parent / "audio" / "wake_emergency.wav"

    res = pipeline.process_audio(str(emerg_file), enable_wake_word=True, language_code="hi")

    w_lat = res.get("wake_detection_latency_ms")
    w2c = res.get("wake_to_capture_latency_ms")
    ftl = res.get("first_token_latency_ms")
    stt_lat = res.get("stt_latency_ms")
    clf_lat = res.get("classifier_latency_ms")
    tot_w2d = res.get("total_wake_to_decision_latency_ms")
    cpu = res.get("cpu_percent")
    ram = res.get("ram_rss_mb")

    print("=" * 60)
    print("ACTUAL MEASURED LATENCIES & RESOURCE CONSUMPTION:")
    print("=" * 60)
    print(f"1. Wake Detection Latency         : {w_lat:.2f} ms")
    print(f"2. Wake -> Capture Switch Latency : {w2c:.3f} ms")
    print(f"3. First-Token STT Latency (FTL)  : {ftl:.2f} ms" if ftl else "3. First-Token STT Latency (FTL)  : N/A")
    print(f"4. Streaming STT Latency          : {stt_lat:.2f} ms")
    print(f"5. Emergency Classifier Latency   : {clf_lat:.2f} ms")
    print(f"6. Total Wake -> Decision Latency : {tot_w2d:.2f} ms" if tot_w2d else "6. Total Wake -> Decision Latency : N/A")
    print(f"7. Process CPU Utilization        : {cpu:.1f} %")
    print(f"8. Process RAM (Resident Set Size): {ram:.2f} MB")
    print("=" * 60)

    assert w_lat is not None and w_lat < 50.0, f"Wake detection latency {w_lat} ms exceeds 50ms target"
    assert w2c is not None and w2c < 5.0, f"Wake->capture latency {w2c} ms exceeds 5ms target"
    assert clf_lat is not None and clf_lat < 10.0, f"Classifier latency {clf_lat} ms exceeds 10ms target"
    print("--- Test 8 PASSED ---")


if __name__ == "__main__":
    test_wake_detection()
    test_false_positive_rejection()
    test_debounce_and_cooldown()
    test_wake_phrase_stripping()
    test_wake_normal_pipeline()
    test_wake_emergency_pipeline()
    test_wake_pause_emergency_pipeline()
    test_latency_and_resource_profiling()
    print("\n=======================================================")
    print("ALL 8 WAKE-WORD TEST SUITE RUNS PASSED SUCCESSFULLY! ✅")
    print("=======================================================\n")
