"""
Runs the ASR server.
"""

import base64
from fastapi import FastAPI, Request
from asr_manager import ASRManager

app = FastAPI()
manager = ASRManager()

@app.post("/asr")
async def asr_route(request: Request) -> dict:
    payload = await request.json()
    instances = payload.get("instances", [])
    predictions = []
    for instance in instances:
        audio_bytes = base64.b64decode(instance.get("b64", ""))
        transcription = manager.asr(audio_bytes)
        predictions.append(transcription)
    return {"predictions": predictions}

@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"message": "ok"}
