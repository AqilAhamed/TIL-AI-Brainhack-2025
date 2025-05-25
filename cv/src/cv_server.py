import base64
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

from cv_manager import CVManager

app = FastAPI()
manager = CVManager()

class Instance(BaseModel):
    key: Any
    b64: str

class RequestPayload(BaseModel):
    instances: List[Instance]

@app.post("/cv")
async def cv(payload: RequestPayload) -> Dict[str, List[List[Dict[str, Any]]]]:
    predictions = []
    for inst in payload.instances:
        try:
            img_bytes = base64.b64decode(inst.b64)
            detections = manager.cv(img_bytes)
        except Exception as e:
            print("Error processing instance", inst.key, e)
            detections = []
        predictions.append(detections)
    return {"predictions": predictions}

@app.get("/health")
def health() -> Dict[str, str]:
    return {"message": "health ok"}