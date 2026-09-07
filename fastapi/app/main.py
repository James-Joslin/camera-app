import os
from pathlib import Path

import cv2
import numpy as np
import psycopg
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.model_runtime import ModelService

app = FastAPI(title="Camera Inference API", version="0.2.0")
models = ModelService()


def connection_string() -> str:
    return (
        f"host={os.getenv('POSTGRES_HOST', 'db')} port={os.getenv('POSTGRES_PORT', '5432')} "
        f"dbname={os.getenv('POSTGRES_DB', 'cameras')} user={os.getenv('POSTGRES_USER', 'camera_app')} "
        f"password={os.getenv('POSTGRES_PASSWORD', 'change-me')}"
    )


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "camera-inference", "status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> JSONResponse:
    checks: dict[str, object] = {"model": models.status()}
    try:
        with psycopg.connect(connection_string(), connect_timeout=3) as connection:
            connection.execute("SELECT 1")
        checks["postgres"] = "ok"
    except Exception as exc:
        checks["postgres"] = str(exc)
    ready_now = checks["postgres"] == "ok" and bool(checks["model"]["ready"])
    return JSONResponse({"status": "ready" if ready_now else "not_ready", **checks}, status_code=200 if ready_now else 503)


@app.get("/api/inference/status")
def inference_status() -> dict[str, object]:
    return models.status()


@app.post("/api/inference/detect")
async def detect(image: UploadFile = File(...), threshold: float = 0.5) -> dict[str, object]:
    if not 0.01 <= threshold <= 0.99:
        raise HTTPException(400, "threshold must be between 0.01 and 0.99")
    payload = await image.read()
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(400, "The uploaded file is not a supported image.")
    try:
        detections, inference_ms = models.predict(decoded, threshold)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {
        "model": models.status()["model"],
        "image": {"width": int(decoded.shape[1]), "height": int(decoded.shape[0])},
        "inferenceMs": round(inference_ms, 2),
        "detections": detections,
    }


@app.get("/api/camera/models")
def model_files() -> dict[str, object]:
    model_dir = Path(os.getenv("MODEL_DIR", "/models"))
    files = sorted(path.name for path in model_dir.glob("*") if path.is_file()) if model_dir.exists() else []
    return {"modelDirectory": str(model_dir), "models": files, "active": models.status()}

