"""
iTantra Receiver Web App - Main Application Entrypoint (web_app/main.py)
========================================================================
Runs local FastAPI server serving frontend static assets and WebSocket endpoint.
Strictly offline, local binding (127.0.0.1).
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from web_app.api.routes import router
from web_app.config.settings import SERVER_HOST, SERVER_PORT

app = FastAPI(
    title="iTantra Receiver Pipeline & Performance Lab",
    description="Offline Indian Language Receiver Prototype with Streaming Indic-Mio TTS & IndicTrans2",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Mount frontend directory
frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    print(f"Starting iTantra Receiver Web App on http://{SERVER_HOST}:{SERVER_PORT}")
    uvicorn.run("web_app.main:app", host=SERVER_HOST, port=SERVER_PORT, reload=False)
