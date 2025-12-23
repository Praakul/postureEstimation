#  Real-Time Posture Analysis for Industrial Weightlifting

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green?logo=qt&logoColor=white)
![PyTorch](https://img.shields.io/badge/AI-PyTorch-orange?logo=pytorch&logoColor=white)
![MediaPipe](https://img.shields.io/badge/Vision-MediaPipe-lightgrey)

> **Protecting the workforce in the era of automation.**

##  Overview

In an industrial landscape where manual material handling remains a primary cause of musculoskeletal disorders (MSDs), standard safety protocols often fail to provide immediate, actionable feedback.

This project implements an **Industrial IoT (IIoT) Solution** for real-time ergonomic monitoring. Unlike simple pose detectors, it uses a **Hybrid Intelligence** approach—combining deterministic bio-mechanical physics with probabilistic Deep Learning—distributed across a scalable **Edge-Cloud Architecture**.

---

##  System Architecture & Logic

The system is decoupled into three distinct layers to ensure scalability, low latency, and robustness.

### 1. The Edge Client (The Eye) 
Running on factory floor hardware (Laptop/NUC/Jetson).
* **Vision:** Extracts a 33-point 3D skeleton using **MediaPipe Pose**.
* **Signal Processing:** Raw keypoints are noisy. We apply a **OneEuroFilter** to smooth signals in real-time, removing high-frequency jitter while preserving low-latency responsiveness.
* **Normalization Bridge:** Before data leaves the edge, it is mathematically normalized:
    * *Scale Invariance:* Torso length scaled to 1.0 units.
    * *View Invariance:* Skeleton rotated to align hips with the camera axis (canonical view).

### 2. The Inference Server (The Brain) 
Running on a central GPU server or Cloud instance.
* **Model:** A **Spatial-Temporal Graph Convolutional Network (ST-GCN)**.
* **Input:** It analyzes **6 Channels** of data (X, Y, Z Position + X, Y, Z Velocity) over a **50-frame sequence**.
* **Attention:** Integrated **SE-Blocks** (Squeeze-and-Excitation) allow the model to focus on critical load-bearing joints (Spine/Hips) while ignoring peripheral noise.

### 3. The Safety Logic (The Guard) 
A hybrid decision engine combines AI predictions with deterministic heuristics:
* **Bio-Mechanical Heuristic:** Differentiates between a **Stoop** (Back bent + Legs straight = Danger) and a **Squat** (Back bent + Knees bent = Safe).
* **Context Awareness:** Only triggers alerts if the hands are detected below the knees (lifting context).
* **Persistence (Debouncing):** A generic alarm is useless if it flickers. We use a rolling buffer logic that triggers a "Critical" alert only if the dangerous posture persists for **>1 second**.

---

## Getting Started

### 1. Development (Training the Model)

If you want to retrain the model from scratch using the CarDA dataset:

```bash
# Generate normalized dataset with velocity features
cd development
python generate_data.py

# Train the model (Using optimized hyperparameters)
python train.py --batch_size 64 --lr 0.008 --epochs 100
```

### 2. Server Deployment (The Brain)

Start the inference engine. This listens for incoming skeleton data via WebSockets.

```bash
cd server
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Output:** Server: AI Model Loaded on CUDA

### 3. Client Deployment (The Edge)

Run the GUI on the laptop connected to the camera.

```bash
cd client
python run_app.py
```

**Visual Feedback:**
- 🟢 **Green:** Safe Posture.
- 🟠 **Orange:** Warning (Bad Form).
- 🔴 **Red:** CRITICAL RISK (Unsafe Lift Detected).

---

## 📊 Performance & Standards

The system is calibrated against EAWS (European Assembly Worksheet) and NIOSH lifting standards.

| Risk Zone | Angle (Flexion) | Context | System Response |
|-----------|----------------|---------|-----------------|
| **Neutral** | 0° - 20° | Natural standing. | Safe |
| **Mild** | 20° - 60° | Acceptable for short duration. | Warning |
| **Severe** | > 60° | High lumbar disc compression. | Critical |

### Model Metrics (Production v1.0):
- **Validation Accuracy:** 88.24% (Robust, Regularized)
  - *Note:* While 99% is possible on training data, 88% on validation represents true generalization to unseen workers.
- **Architecture:** ST-GCN + Attention + Physics Augmentation.
- **Latency:** < 50ms end-to-end via WebSocket.

---

## Future Roadmap

- [ ] **Multi-Person Tracking:** Upgrade Client logic to support RTMPose for tracking multiple workers simultaneously.
- [ ] **Database Integration:** Connect Server to PostgreSQL to log incident timestamps and video snippets for safety audits.
- [ ] **Edge Optimization:** Convert PyTorch model to TensorRT for deployment on NVIDIA Jetson Nano.

---

<p align="center">
<i>Engineered for Safety. Powered by AI.</i>
</p>
