"""
ORAM – FastAPI Backend Server
==============================
REST API + WebSocket server for the robot inspection platform.

Endpoints
---------
GET  /api/status          – System health + model status
POST /api/analyze          – Upload image → get detections + segmentation
WS   /api/stream           – Live camera frame analysis via WebSocket
POST /api/camera/connect   – Connect to robot camera (URL/IP)
GET  /api/camera/snapshot  – Grab single frame from camera
GET  /api/datasets         – List available datasets
POST /api/train            – Start training job
GET  /api/train/{job_id}   – Training progress

Run
---
    uvicorn ai.server:app --host 0.0.0.0 --port 8000 --reload
    # or
    python -m ai.server
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from loguru import logger

# ── ensure parent is on sys.path ─────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

# ── ORAM modules ─────────────────────────────────────────────────────────
from ai.detection import (
    get_inspection_pipeline,
    InspectionPipeline,
    FrameSkipController,
)
from ai.preprocessing import get_lighting_normalizer, LightingNormalizer
from ai.segmentation import get_segmentor, SAM2Segmentor, SAM2_AVAILABLE
from ai.datasets import get_dataset_manager
from ai.training import (
    AgentConfig,
    AgentType,
    get_orchestrator,
    simulate_training,
    TORCH_AVAILABLE,
)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
except ImportError:
    Image = None

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ORAM – Intelligent Inspection Platform",
    description="TGV undercarriage anomaly detection with AI + SAM 2 segmentation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

pipeline: Optional[InspectionPipeline] = None
normalizer: Optional[LightingNormalizer] = None
segmentor: Optional[SAM2Segmentor] = None
frame_skip = FrameSkipController(max_skip=3, confidence_threshold=0.90)

# Camera connection state
camera_state: Dict[str, Any] = {
    "connected": False,
    "url": None,
    "capture": None,  # cv2.VideoCapture
    "protocol": None,  # "mjpeg", "rtsp", "webcam"
}


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    global pipeline, normalizer, segmentor
    logger.info("Starting ORAM server ...")

    pipeline = get_inspection_pipeline()
    await pipeline.initialize()

    normalizer = get_lighting_normalizer()
    segmentor = get_segmentor()

    logger.info("ORAM server ready")


@app.on_event("shutdown")
async def shutdown():
    _release_camera()
    logger.info("ORAM server stopped")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CameraConnectRequest(BaseModel):
    url: str  # e.g. "http://192.168.1.100:8080/video" or "rtsp://..." or "0" for webcam
    protocol: str = "auto"  # "mjpeg", "rtsp", "webcam", "auto"


class TrainRequest(BaseModel):
    agent: str = "general"  # "crack", "corrosion", "leak", "general", "all"
    model_name: str = "efficientnet_b0"
    dataset_key: str = "synthetic"
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3


class AnalyzeSettings(BaseModel):
    threshold: float = 0.5
    use_sam2: bool = True
    preprocess: bool = True


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _image_from_bytes(data: bytes) -> np.ndarray:
    """Decode image bytes to BGR numpy array."""
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    return img


def _image_to_base64(image: np.ndarray) -> str:
    """Encode BGR image to base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _grab_camera_frame() -> Optional[np.ndarray]:
    """Read a frame from the connected camera."""
    cap = camera_state.get("capture")
    if cap is None or not cap.isOpened():
        return None
    ret, frame = cap.read()
    return frame if ret else None


def _release_camera():
    """Release camera resources."""
    cap = camera_state.get("capture")
    if cap is not None:
        cap.release()
    camera_state.update({"connected": False, "url": None, "capture": None, "protocol": None})


async def _analyze_frame(
    image: np.ndarray,
    threshold: float = 0.5,
    use_sam2: bool = True,
    preprocess: bool = True,
) -> Dict[str, Any]:
    """Run full analysis pipeline on a single frame."""
    t0 = time.perf_counter()

    # 1. Preprocess for lighting
    if preprocess and normalizer:
        image_proc = normalizer.normalize(image)
    else:
        image_proc = image

    # 2. Run detection pipeline
    pipeline.set_threshold(threshold)
    result = await pipeline.process_frame(image_proc)

    # 3. Optional SAM 2 segmentation
    seg_masks = []
    if use_sam2 and segmentor and segmentor.available and result.detections:
        # Convert to RGB for SAM 2
        rgb = cv2.cvtColor(image_proc, cv2.COLOR_BGR2RGB)
        masks = segmentor.segment_detections(rgb, result.detections)
        for m in masks:
            seg_masks.append(m.to_dict() if m else None)

    total_ms = (time.perf_counter() - t0) * 1000

    return {
        "detections": [d.to_dict() for d in result.detections],
        "has_anomalies": result.has_anomalies,
        "total_detections": len(result.detections),
        "confidence_scores": result.confidence_scores,
        "segmentation_masks": seg_masks,
        "processing_time_ms": round(total_ms, 2),
        "preprocessing_applied": preprocess,
        "sam2_used": use_sam2 and segmentor is not None and segmentor.available,
    }


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=RedirectResponse)
async def root():
    """Serve the frontend dashboard."""
    return RedirectResponse(url="/static/index.html")


@app.get("/api/status")
async def status():
    """System health + model status."""
    return {
        "status": "ok",
        "timestamp": time.time(),
        "models": {
            "detection_pipeline": pipeline is not None,
            "preprocessing": normalizer is not None,
            "sam2_available": SAM2_AVAILABLE,
            "sam2_loaded": segmentor._loaded if segmentor else False,
            "pytorch_available": TORCH_AVAILABLE,
            "opencv_available": CV2_AVAILABLE,
        },
        "camera": {
            "connected": camera_state["connected"],
            "url": camera_state["url"],
            "protocol": camera_state["protocol"],
        },
    }


@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    threshold: float = 0.5,
    use_sam2: bool = True,
    preprocess: bool = True,
):
    """Upload an image and get detections + segmentation."""
    if not CV2_AVAILABLE:
        raise HTTPException(500, "OpenCV is required for image analysis")

    data = await file.read()
    try:
        image = _image_from_bytes(data)
    except ValueError as e:
        raise HTTPException(400, str(e))

    result = await _analyze_frame(image, threshold, use_sam2, preprocess)

    # Include annotated image as base64
    annotated = _draw_detections(image, result["detections"])
    result["annotated_image"] = _image_to_base64(annotated)

    return JSONResponse(result)


@app.post("/api/camera/connect")
async def connect_camera(req: CameraConnectRequest):
    """Connect to a robot camera via URL."""
    if not CV2_AVAILABLE:
        raise HTTPException(500, "OpenCV is required for camera")

    # Release any existing connection
    _release_camera()

    url = req.url
    protocol = req.protocol

    # Auto-detect protocol
    if protocol == "auto":
        if url.isdigit():
            protocol = "webcam"
            url = int(url)
        elif "rtsp://" in url:
            protocol = "rtsp"
        else:
            protocol = "mjpeg"

    try:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            raise HTTPException(400, f"Cannot open camera at: {req.url}")

        # Test read
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise HTTPException(400, f"Camera opened but cannot read frames from: {req.url}")

        camera_state.update({
            "connected": True,
            "url": req.url,
            "capture": cap,
            "protocol": protocol,
        })

        return {
            "status": "connected",
            "url": req.url,
            "protocol": protocol,
            "frame_size": [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Camera connection failed: {e}")


@app.post("/api/camera/disconnect")
async def disconnect_camera():
    """Disconnect the camera."""
    _release_camera()
    return {"status": "disconnected"}


@app.get("/api/camera/snapshot")
async def camera_snapshot(
    threshold: float = 0.5,
    use_sam2: bool = True,
    preprocess: bool = True,
):
    """Grab a single frame from the camera and analyze it."""
    frame = _grab_camera_frame()
    if frame is None:
        raise HTTPException(400, "No camera connected or cannot read frame")

    result = await _analyze_frame(frame, threshold, use_sam2, preprocess)
    annotated = _draw_detections(frame, result["detections"])
    result["annotated_image"] = _image_to_base64(annotated)
    result["raw_image"] = _image_to_base64(frame)

    return JSONResponse(result)


@app.get("/api/datasets")
async def list_datasets():
    """List available training datasets."""
    dm = get_dataset_manager()
    return {"datasets": dm.list_datasets()}


@app.post("/api/train")
async def start_training(req: TrainRequest):
    """Start a training job."""
    orch = get_orchestrator()

    if req.agent == "all":
        agent_types = [AgentType.CRACK, AgentType.CORROSION, AgentType.LEAK, AgentType.GENERAL]
    else:
        try:
            agent_types = [AgentType(req.agent)]
        except ValueError:
            raise HTTPException(400, f"Unknown agent type: {req.agent}")

    agents = [
        AgentConfig(
            agent_type=at,
            model_name=req.model_name,
            dataset_key=req.dataset_key,
            epochs=req.epochs,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
        )
        for at in agent_types
    ]

    # Use simulated training if PyTorch is not available
    if not TORCH_AVAILABLE:
        results = await simulate_training(agents)
        return {"status": "completed (simulated)", "metrics": results}

    job = orch.create_job(agents)

    # Run in background
    dm = get_dataset_manager()
    data_root = dm.root

    asyncio.create_task(_run_training_bg(job.job_id, data_root))

    return {
        "status": "started",
        "job_id": job.job_id,
        "agents": [a.agent_type.value for a in agents],
    }


async def _run_training_bg(job_id: str, data_root: Path):
    """Background training task."""
    orch = get_orchestrator()
    try:
        await orch.run_job(job_id, data_root)
    except Exception as e:
        logger.error(f"Background training failed: {e}")


@app.get("/api/train/{job_id}")
async def training_status(job_id: str):
    """Get training job status."""
    orch = get_orchestrator()
    job = orch.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")

    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "current_agent": job.current_agent,
        "current_epoch": job.current_epoch,
        "total_epochs": job.total_epochs,
        "metrics": job.metrics,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "error": job.error,
    }


# ---------------------------------------------------------------------------
# WebSocket – Live camera stream
# ---------------------------------------------------------------------------


@app.websocket("/api/stream")
async def stream_websocket(ws: WebSocket):
    """
    Live camera analysis via WebSocket.

    Client sends JSON:
        {"action": "start", "threshold": 0.5, "use_sam2": true, "preprocess": true}
        {"action": "stop"}
        {"action": "frame", "data": "<base64 jpeg>"}

    Server sends JSON per frame:
        {"detections": [...], "annotated_image": "<base64>", "processing_time_ms": ...}
    """
    await ws.accept()
    logger.info("WebSocket client connected")

    streaming = False
    threshold = 0.5
    use_sam2 = True
    preprocess = True

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            action = msg.get("action", "")

            if action == "start":
                streaming = True
                threshold = msg.get("threshold", 0.5)
                use_sam2 = msg.get("use_sam2", True)
                preprocess = msg.get("preprocess", True)
                await ws.send_json({"status": "streaming_started"})

                # Start streaming loop
                while streaming:
                    # Try camera first
                    frame = _grab_camera_frame()
                    if frame is not None:
                        if frame_skip.should_process():
                            result = await _analyze_frame(frame, threshold, use_sam2, preprocess)
                            annotated = _draw_detections(frame, result["detections"])
                            result["annotated_image"] = _image_to_base64(annotated)
                            result["raw_image"] = _image_to_base64(frame)

                            # Report to frame skip controller
                            is_normal = not result["has_anomalies"]
                            conf = max(result.get("confidence_scores", {}).values()) if result.get("confidence_scores") else 0.5
                            frame_skip.report_result(is_normal, conf)

                            await ws.send_json(result)
                        else:
                            # Skip frame, just send raw
                            await ws.send_json({
                                "skipped": True,
                                "raw_image": _image_to_base64(frame),
                            })
                    else:
                        await ws.send_json({"status": "no_camera"})

                    await asyncio.sleep(0.033)  # ~30 FPS target

                    # Check for stop command (non-blocking)
                    try:
                        stop_data = await asyncio.wait_for(ws.receive_text(), timeout=0.001)
                        stop_msg = json.loads(stop_data)
                        if stop_msg.get("action") == "stop":
                            streaming = False
                            frame_skip.reset()
                            await ws.send_json({"status": "streaming_stopped"})
                    except asyncio.TimeoutError:
                        pass

            elif action == "frame":
                # Client sends a single frame for analysis
                frame_b64 = msg.get("data", "")
                if frame_b64:
                    frame_bytes = base64.b64decode(frame_b64)
                    frame = _image_from_bytes(frame_bytes)
                    result = await _analyze_frame(frame, threshold, use_sam2, preprocess)
                    annotated = _draw_detections(frame, result["detections"])
                    result["annotated_image"] = _image_to_base64(annotated)
                    await ws.send_json(result)

            elif action == "stop":
                streaming = False
                frame_skip.reset()
                await ws.send_json({"status": "stopped"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

# Severity → colour (BGR)
SEVERITY_COLORS = {
    "critical": (0, 0, 255),  # Red
    "high": (0, 100, 255),    # Orange
    "medium": (0, 200, 255),  # Yellow
    "low": (0, 255, 0),       # Green
}

ANOMALY_ICONS = {
    "crack": "⚡",
    "leak": "💧",
    "corrosion": "🔶",
    "missing_part": "❌",
    "wear": "🔧",
    "deformation": "📐",
    "normal": "✅",
}


def _draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes and labels on the image."""
    if not CV2_AVAILABLE:
        return image

    out = image.copy()

    for det in detections:
        bbox = det.get("bbox")
        if not bbox:
            continue

        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        severity = det.get("severity", "medium")
        color = SEVERITY_COLORS.get(severity, (255, 255, 255))
        confidence = det.get("confidence", 0)
        anomaly_type = det.get("anomaly_type", "unknown")

        # Draw rectangle
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

        # Label background
        label = f"{anomaly_type} {confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x, y - th - 8), (x + tw + 4, y), color, -1)
        cv2.putText(out, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import uvicorn
    logger.info("Starting ORAM server on http://0.0.0.0:8000")
    uvicorn.run(
        "ai.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
