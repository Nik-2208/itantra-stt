"""
iTantra Receiver Web App - REST & WebSocket API (web_app/api/routes.py)
========================================================================
REST endpoints for health status, language list, test sentences, and benchmark export.
WebSocket endpoint for real-time streaming audio pipeline.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, JSONResponse
from web_app.config.settings import LanguageCode, LANGUAGE_NAMES, BENCHMARK_TOKEN_CHUNK_SIZES
from web_app.pipeline.receiver_pipeline import ReceiverPipeline
from web_app.schemas.message import ReceivedMessage, VoiceProfile
from web_app.schemas.events import WebSocketEvent, EventType
from web_app.benchmark.metrics import benchmark_store
import json

router = APIRouter()
pipeline = ReceiverPipeline()

@router.get("/api/status")
async def get_status():
    return {
        "status": "online",
        "offline_mode": True,
        "models": {
            "tts": "SPRINGLab/Indic-Mio (MioCodec Stream Decoder)",
            "mt": "AI4Bharat IndicTrans2",
            "emergency_detector": "Deterministic 10-Language Two-Pass",
        },
        "supported_languages": [
            {"code": code.value, "name": LANGUAGE_NAMES[code]} for code in LanguageCode
        ],
        "supported_chunk_sizes": BENCHMARK_TOKEN_CHUNK_SIZES,
    }

@router.get("/api/benchmark/export/json")
async def export_json():
    return JSONResponse(content=json.loads(benchmark_store.export_json()))

@router.get("/api/benchmark/export/csv")
async def export_csv():
    return PlainTextResponse(content=benchmark_store.export_csv(), media_type="text/csv")

@router.get("/api/benchmark/comparison")
async def get_comparison():
    return benchmark_store.get_comparison()

@router.websocket("/ws/receiver")
async def websocket_receiver_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw_text = await websocket.receive_text()
            data = json.loads(raw_text)

            # Build ReceivedMessage
            msg = ReceivedMessage(
                text=data.get("text", ""),
                source_language=LanguageCode(data.get("source_language", "HI")),
                target_language=LanguageCode(data.get("target_language", "HI")),
                voice_profile=VoiceProfile(
                    accent_label=data.get("accent_label", "Standard / Neutral"),
                    style_tag=data.get("style_tag", "NORMAL"),
                ),
            )
            mode = data.get("mode", "OPTIMIZED")
            chunk_size = int(data.get("chunk_size", 32))

            # Stream execution events
            async for event in pipeline.execute_stream(msg, mode=mode, chunk_size=chunk_size):
                await websocket.send_text(event.model_dump_json())

    except WebSocketDisconnect:
        pass
    except Exception as e:
        err_event = WebSocketEvent(
            event_type=EventType.ERROR,
            data={"error": str(e)},
        )
        try:
            await websocket.send_text(err_event.model_dump_json())
        except Exception:
            pass
