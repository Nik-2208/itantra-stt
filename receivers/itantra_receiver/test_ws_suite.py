"""
Automated Test & Benchmark Runner for iTantra Receiver Web App
Runs Test Cases 1 through 5 over WebSocket and compares BASELINE vs OPTIMIZED.
"""

import asyncio
import json
import sys
import websockets

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

TEST_CASES = [
    {
        "name": "Case 1: Same Language (HI -> HI) - No MT",
        "payload": {
            "text": "मुझे आपकी आवाज़ सुनाई दे रही है।",
            "source_language": "HI",
            "target_language": "HI",
            "mode": "OPTIMIZED",
            "chunk_size": 32
        }
    },
    {
        "name": "Case 2: Cross Language (GU -> HI) - Help Emergency",
        "payload": {
            "text": "મને મદદ જોઈએ",
            "source_language": "GU",
            "target_language": "HI",
            "mode": "OPTIMIZED",
            "chunk_size": 32
        }
    },
    {
        "name": "Case 3: Cross Language (MR -> EN)",
        "payload": {
            "text": "मला मदत हवी आहे",
            "source_language": "MR",
            "target_language": "EN",
            "mode": "OPTIMIZED",
            "chunk_size": 32
        }
    },
    {
        "name": "Case 4: Emergency Fire Alert (HI -> HI)",
        "payload": {
            "text": "आग लग गई है, तुरंत मदद भेजिए",
            "source_language": "HI",
            "target_language": "HI",
            "style_tag": "ALERT",
            "mode": "OPTIMIZED",
            "chunk_size": 32
        }
    },
    {
        "name": "Case 5: SOS Priority Preemption",
        "payload": {
            "text": "SOS! बाढ़ में 4 लोग फंसे हैं, तुरंत नाव भेजिए",
            "source_language": "HI",
            "target_language": "HI",
            "style_tag": "SOS",
            "mode": "OPTIMIZED",
            "chunk_size": 32
        }
    },
    {
        "name": "Case 6: Baseline Full-Sentence WAV Generation (Comparison)",
        "payload": {
            "text": "मुझे आपकी आवाज़ सुनाई दे रही है।",
            "source_language": "HI",
            "target_language": "HI",
            "mode": "BASELINE",
            "chunk_size": 32
        }
    }
]

async def run_tests():
    uri = "ws://127.0.0.1:8000/ws/receiver"
    print(f"Connecting to {uri}...")
    async with websockets.connect(uri) as ws:
        for tc in TEST_CASES:
            print(f"\n==========================================")
            print(f"RUNNING: {tc['name']}")
            print(f"Payload: {tc['payload']}")
            await ws.send(json.dumps(tc['payload']))
            
            while True:
                resp = await ws.recv()
                evt = json.loads(resp)
                evt_type = evt.get("event_type")
                
                if evt_type == "ORIGINAL_DISPLAYED":
                    print(f" -> [DISPLAY] Text: {evt['data']['text']} (Latency: {evt['data']['latency_ms']} ms)")
                elif evt_type == "EMERGENCY_DETECTED":
                    print(f" -> [EMERGENCY] Detected: {evt['data']['is_emergency']}, Priority: {evt['data']['priority']}, Terms: {evt['data']['matched_terms']}")
                elif evt_type == "TRANSLATION_COMPLETE":
                    print(f" -> [TRANSLATION] Output: {evt['data']['translated_text']} (Time: {evt['data']['total_translation_ms']} ms)")
                elif evt_type == "TTS_FIRST_AUDIO":
                    print(f" -> [TTS FIRST AUDIO] TTFA: {evt['data']['ttfa_ms']} ms")
                elif evt_type == "ERROR":
                    print(f" -> [ERROR] {evt['data']}")
                    break
                elif evt_type == "BENCHMARK_RESULT":
                    d = evt['data']
                    print(f" -> [BENCHMARK] Mode: {d['mode']} | TTFA: {d['ttfa_ms']} ms | E2E: {d['e2e_latency_ms']} ms | RTF: {d['rtf']} | RAM: {d['peak_ram_mb']} MB")
                    break

if __name__ == "__main__":
    asyncio.run(run_tests())
