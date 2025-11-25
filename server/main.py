import sys
import os
import json
import torch
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager

# Fix path to allow importing from 'core' (assuming server/ is in project root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nn.model import STGCN

# --- CONFIG ---
MODEL_PATH = "stgcn_posture_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global Model Variable
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load Model
    global model
    try:
        # Initialize model structure (6 channels for pos+vel)
        model = STGCN(num_classes=3, in_channels=6).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"✅ Server: AI Model Loaded on {DEVICE}")
    except Exception as e:
        print(f"❌ Server: Failed to load model: {e}")
    
    yield
    
    # Shutdown logic (if any)
    print("🛑 Server: Shutting down")

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"status": "Industrial Safety AI Online"}

@app.websocket("/ws/predict")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(f"🔌 Client Connected: {websocket.client}")
    
    try:
        while True:
            # 1. Receive Data (Expects JSON list of 50 frames)
            data = await websocket.receive_text()
            skeleton_seq = json.loads(data) # format: (50, 17, 6)
            
            if model is None:
                await websocket.send_text(json.dumps({"status": "Error", "code": -1}))
                continue

            # 2. Prepare Tensor
            # Input: (Batch, Channels, Time, Vertices) -> (1, 6, 50, 17)
            np_seq = np.array(skeleton_seq, dtype=np.float32)
            tensor = torch.tensor(np_seq).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            
            # 3. Inference
            with torch.no_grad():
                logits = model(tensor)
                pred_idx = torch.argmax(logits, dim=1).item()
            
            # 4. Send Result Back
            response = {
                "status": "OK",
                "prediction": pred_idx, # 0=Safe, 1=Warn, 2=Critical
                "confidence": float(torch.max(torch.softmax(logits, dim=1)))
            }
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        print(f"🔌 Client Disconnected: {websocket.client}")
    except Exception as e:
        print(f"❌ Error: {e}")
        await websocket.close()