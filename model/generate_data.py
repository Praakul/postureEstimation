import sys
import os
import json
import base64
import re
from collections import deque
from datetime import datetime
from typing import Optional, Deque, List

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------
# PATH / IMPORTS
# ---------------------------------------------------------

# Assume this file is in `server/main.py`
# Project root has: nn/model.py and utils/normalization.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from nn.model import STGCN  # type: ignore
from utils.normalization import normalize_skeleton  # type: ignore

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

MODEL_PATH = "stgcn_posture_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global objects
model: Optional[STGCN] = None
pose: Optional[mp.solutions.pose.Pose] = None

# Indices used in generate_data.py (must match EXACTLY)
JOINT_INDICES: List[int] = [
    0,   # nose
    2,   # left_eye? / left_side of head
    5,   # right_eye? / right_side of head
    7,   # left_ear / side
    8,   # right_ear / side
    11,  # left_shoulder
    12,  # right_shoulder
    13,  # left_elbow
    14,  # right_elbow
    15,  # left_wrist
    16,  # right_wrist
    23,  # left_hip
    24,  # right_hip
    25,  # left_knee
    26,  # right_knee
    27,  # left_ankle
    28,  # right_ankle
]
NUM_JOINTS = len(JOINT_INDICES)  # should be 17

# ---------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------

app = FastAPI(title="Industrial Posture ST-GCN Inference Server")

# CORS for local Reflex dev (adjust origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# STARTUP / SHUTDOWN: LOAD MODEL + MEDIAPIPE
# ---------------------------------------------------------


@app.on_event("startup")
async def startup_event():
    global model, pose

    # Load ST-GCN model
    try:
        print("🔄 Loading ST-GCN model...")
        model = STGCN(num_classes=3, in_channels=6)  # 3 classes, 6 channels (pos+vel)
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        print(f"✅ ST-GCN model loaded on {DEVICE}")
    except Exception as e:
        print(f"❌ Failed to load ST-GCN model: {e}")
        model = None

    # Initialize MediaPipe Pose (same style as generate_data.py, but for video)
    try:
        print("🔄 Initializing MediaPipe Pose...")
        mp_pose = mp.solutions.pose
        pose_options = dict(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        globals()["pose"] = mp_pose.Pose(**pose_options)
        print("✅ MediaPipe Pose initialized")
    except Exception as e:
        print(f"❌ Failed to initialize MediaPipe Pose: {e}")
        globals()["pose"] = None


@app.on_event("shutdown")
async def shutdown_event():
    global pose
    if pose is not None:
        pose.close()
        pose = None
    print("🛑 Server shutting down.")


# ---------------------------------------------------------
# BASIC HEALTH CHECK
# ---------------------------------------------------------


@app.get("/")
async def root():
    return {
        "status": "Industrial Safety AI Online",
        "device": DEVICE,
        "model_loaded": model is not None,
        "num_joints": NUM_JOINTS,
    }


# ---------------------------------------------------------
# UTILS: DATA URI → OpenCV BGR IMAGE
# ---------------------------------------------------------


def decode_data_uri_to_cv2(data_uri: str) -> Optional[np.ndarray]:
    """
    Convert 'data:image/jpeg;base64,...' into a OpenCV BGR image.
    """
    match = re.match(r"data:image/[^;]+;base64,(.+)", data_uri)
    if match:
        b64_data = match.group(1)
    else:
        # assume it's pure base64 w/o header
        b64_data = data_uri

    try:
        img_bytes = base64.b64decode(b64_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # BGR image
        return img
    except Exception as e:
        print("❌ Failed to decode frame:", e)
        return None


# ---------------------------------------------------------
# UTILS: MEDIAPIPE → 17×6 SKELETON (MATCHING generate_data.py)
# ---------------------------------------------------------


def extract_skeleton_17x6(
    img_bgr: np.ndarray,
    prev_norm_skel: Optional[np.ndarray],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Run MediaPipe Pose on a BGR frame and return a single-frame skeleton:
        shape: (17, 6) = [x, y, z, vx, vy, vz] per joint.

    This replicates the pipeline in generate_data.py:
        - pose.process(image)
        - indices = [0,2,5,7,8,11,12,13,14,15,16,23,24,25,26,27,28]
        - skel_raw from lms[i].x, y, z
        - normalize_skeleton(skel_raw)
        - velocity = norm_skel - prev_norm_skel
        - combined = concat(norm_skel, velocity)
    """
    global pose
    if pose is None:
        return None, prev_norm_skel

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if not results.pose_landmarks:
        # No person detected
        return None, prev_norm_skel

    lms = results.pose_landmarks.landmark

    # Raw skeleton (17,3) in same order & features as generate_data.py
    skel_raw = np.array(
        [[lms[i].x, lms[i].y, lms[i].z] for i in JOINT_INDICES],
        dtype=np.float32,
    )  # (17, 3)

    # Normalize skeleton (imported from utils.normalization)
    norm_skel = normalize_skeleton(skel_raw)  # (17, 3)

    # Velocity
    if prev_norm_skel is None:
        velocity = np.zeros_like(norm_skel)
    else:
        velocity = norm_skel - prev_norm_skel

    # Combined pos+vel: (17,6)
    combined = np.concatenate((norm_skel, velocity), axis=1)

    return combined, norm_skel  # return new prev_norm_skel as norm_skel


# ---------------------------------------------------------
# LEGACY /ws/predict: TAKES (50,17,6) SKELETON SEQUENCE
# ---------------------------------------------------------


@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    """
    Legacy endpoint:
        - Receives JSON-encoded skeleton sequence of shape (50,17,6).
        - Runs ST-GCN and returns prediction.
    """
    await websocket.accept()
    print(f"🔌 Client connected to /ws/predict: {websocket.client}")

    try:
        while True:
            data = await websocket.receive_text()
            skeleton_seq = json.loads(data)  # expect list with shape (50,17,6)

            if model is None:
                await websocket.send_text(json.dumps({"status": "Error", "code": -1}))
                continue

            np_seq = np.array(skeleton_seq, dtype=np.float32)  # (50,17,6)

            # ST-GCN expects (B, C, T, V) = (1, 6, 50, 17)
            tensor = (
                torch.from_numpy(np_seq)
                .permute(2, 0, 1)  # (6,50,17)
                .unsqueeze(0)      # (1,6,50,17)
                .to(DEVICE)
            )

            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)
                pred_idx = int(torch.argmax(probs, dim=1).item())
                confidence = float(torch.max(probs).item())

            response = {
                "status": "OK",
                "prediction": pred_idx,  # 0=Safe, 1=Warning, 2=Critical
                "confidence": confidence,
            }
            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        print(f"🔌 Client disconnected from /ws/predict: {websocket.client}")
    except Exception as e:
        print(f"❌ Error in /ws/predict: {e}")
        await websocket.close()


# ---------------------------------------------------------
# NEW /ws/stream: REAL-TIME JPEG FRAMES FROM BROWSER → ST-GCN
# ---------------------------------------------------------


@app.websocket("/ws/stream")
async def stream_endpoint(websocket: WebSocket):
    """
    Real-time streaming endpoint for Reflex/web UI.

    Client sends text messages:
        { "frame": "data:image/jpeg;base64,..." }

    Pipeline:
        1. Decode image.
        2. Run MediaPipe Pose → skel_raw (17,3).
        3. normalize_skeleton → norm_skel (17,3).
        4. Velocity = norm_skel - prev_norm_skel → (17,3).
        5. Combined frame_skel (17,6).
        6. Maintain a deque of 50 frames: (50,17,6).
        7. When buffer full, run ST-GCN and respond with:

        {
          "status": "safe" | "warning" | "critical",
          "confidence": float,          # 0.0–1.0
          "risk_factors": [str, ...],
          "timestamp": "HH:MM:SS"
        }
    """
    await websocket.accept()
    print(f"🔌 Client connected to /ws/stream: {websocket.client}")

    if model is None:
        await websocket.send_text(
            json.dumps(
                {
                    "status": "safe",
                    "confidence": 0.0,
                    "risk_factors": ["Model not loaded on server"],
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                }
            )
        )
        await websocket.close()
        return

    # Per-connection buffers
    frame_buffer: Deque[np.ndarray] = deque(maxlen=50)  # each (17,6)
    prev_norm_skel: Optional[np.ndarray] = None         # (17,3)

    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)

            frame_data = data.get("frame")
            if not frame_data:
                continue

            img = decode_data_uri_to_cv2(frame_data)
            if img is None:
                continue

            # 1. Frame → combined (17,6), and update prev_norm_skel
            combined, prev_norm_skel = extract_skeleton_17x6(img, prev_norm_skel)
            if combined is None:
                # No person detected; option: clear buffer or just skip
                continue

            frame_buffer.append(combined)

            # 2. Wait until we have 50 frames
            if len(frame_buffer) < 50:
                continue

            # 3. Build sequence: (50,17,6)
            seq = np.stack(frame_buffer, axis=0).astype(np.float32)

            # 4. Convert to tensor: (1,6,50,17)
            tensor = (
                torch.from_numpy(seq)
                .permute(2, 0, 1)  # (6,50,17)
                .unsqueeze(0)      # (1,6,50,17)
                .to(DEVICE)
            )

            # 5. Run ST-GCN
            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)
                pred_idx = int(torch.argmax(probs, dim=1).item())
                confidence = float(torch.max(probs).item())

            # 6. Map class idx → status
            if pred_idx == 0:
                status = "safe"
            elif pred_idx == 1:
                status = "warning"
            else:
                status = "critical"

            # 7. Simple risk messages (you can refine later)
            if status == "safe":
                risk_factors = []
            elif status == "warning":
                risk_factors = ["Form deviation detected by model"]
            else:
                risk_factors = [
                    "Unsafe lift detected by model",
                    "Check lumbar & knee angles",
                ]

            resp = {
                "status": status,
                "confidence": confidence,
                "risk_factors": risk_factors,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }

            await websocket.send_text(json.dumps(resp))

    except WebSocketDisconnect:
        print(f"🔌 Client disconnected from /ws/stream: {websocket.client}")
    except Exception as e:
        print(f"❌ Error in /ws/stream: {e}")
        await websocket.close()
