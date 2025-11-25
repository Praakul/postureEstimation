# Real-Time Posture Analysis for Industrial Weightlifting

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green?logo=qt&logoColor=white)
![PyTorch](https://img.shields.io/badge/AI-PyTorch-orange?logo=pytorch&logoColor=white)
![MediaPipe](https://img.shields.io/badge/Vision-MediaPipe-lightgrey)

> **Protecting the workforce in the era of automation.**

## Overview

In an era increasingly defined by the rise of AGI and automation, a significant portion of the global industrial sector still relies on human manpower. Manual material handling—specifically lifting heavy objects—remains a primary cause of musculoskeletal disorders (MSDs) and lifelong injuries.

This project bridges the gap between **Occupational Safety** and **Computer Vision**. It is an end-to-end application that monitors workers in real-time, analyzes their lifting mechanics using bio-mechanical rules and Deep Learning, and provides instant feedback to prevent injuries before they happen.

---

## Conceptual Architecture

This system is not just a simple pose detector. It uses a **Hybrid Intelligence** approach, combining deterministic physics with probabilistic AI.

### 1. The Eye: Vision & Extraction 
We use **MediaPipe Pose** (or YOLOv11-Pose) to extract a 33-point 3D skeleton from the raw video feed.
* **Jitter Reduction:** Raw signals are noisy. We implement a **OneEuroFilter** to smooth movements, removing high-frequency jitter while maintaining low latency for fast motions.

### 2. The Bridge: Mathematical Normalization 
A raw skeleton depends on camera distance and angle. A worker standing far away looks "smaller" than one close by.
* **Scale Invariance:** We normalize the skeleton so the torso length is always `1.0` units.
* **View Invariance:** We mathematically rotate the skeleton so the hips align with the camera axis.
* **Result:** The AI model sees a standardized "canonical" body, regardless of camera placement or worker height.

### 3. The Brain: ST-GCN (Spatial-Temporal Graph ConvNet) 
Traditional AI sees a flat list of numbers. Our model uses **ST-GCN**, which understands:
* **Spatial Structure:** It knows the wrist is connected to the elbow, not the foot.
* **Temporal Dynamics:** It analyzes a sequence of **50 frames** (approx. 2 seconds). It differentiates between *static bending* and *dynamic lifting*.
* **Attention:** Integrated **SE-Blocks** (Squeeze-and-Excitation) allow the model to focus on critical joints (Spine/Hips) while ignoring noise (Wrists).

### 4. The Logic: Bio-Mechanical Heuristics 
We overlay AI predictions with strict geometric rules based on **EAWS (European Assembly Worksheet)** and **NIOSH** standards:
* **The Stoop (Danger):** Back bent (>60°) + Legs straight. High lumbar load.
* **The Squat (Safe):** Back bent + Knees bent. Load transferred to legs.
* **Context Awareness:** The system only alerts if the hands are below the knees (indicating a lift is happening), reducing false alarms during resting periods.

---

## Project Structure

A modular, industry-standard architecture designed for scalability.

```text
IndustrialSafetyApp/
├── run_app.py              # ENTRY POINT: The GUI Application
├── train.py                # ENTRY POINT: The Model Trainer
├── generate_data.py        # ENTRY POINT: Data Processor
│
├── nn/                     # Neural Network Core
│   ├── model.py            # ST-GCN Architecture Definition
│   ├── dataset.py          # Custom PyTorch Dataset with Augmentation
│   └── trainer.py          # Training Engine & Checkpointing
│
├── ui/                     # User Interface
│   ├── videoThread.py      # Multi-threaded Video Processing
│   ├── style.py            # QSS Styling & Themes
│   └── camUtils.py         # Camera Management
│
├── utils/                  # Shared Logic
│   ├── poselogic.py        # The "Brain" (Inference Engine)
│   ├── normalization.py    # The "Bridge" (Math transforms)
│   ├── filter.py           # Signal Smoothing (OneEuro)
│   └── argparser.py        # Configuration Management
│
└── Data/                   # Training Datasets (BVH/Video)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Webcam (or video file)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/praakul/industrial-safety-pose.git
cd industrial-safety-pose
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```
*(Key libs: torch, opencv-python, mediapipe, pyqt6, numpy, optuna)*

---

## 🛠️ Workflow: From Data to Deployment

### Phase 1: Data Generation 🧬
We convert raw videos into a normalized dataset suitable for AI training.

```bash
python generate_data.py
```

- Scans `Data/` for videos.
- Extracts skeletons using MediaPipe.
- Calculates Ground Truth labels using Bio-Mechanical Geometry.
- Saves `X_train.npy` (Features) and `y_train.npy` (Labels).

### Phase 2: Training the Brain 
We train the ST-GCN model on the generated data.

```bash
python train.py --epochs 50 --batch_size 32 --lr 0.1
```

- Uses Weighted Loss to handle class imbalance.
- Applies Physics-based Augmentation (Random rotation/jitter).
- Saves the best model to `stgcn_posture_model.pth`.

### Phase 3: Deployment 
Launch the real-time safety guard.

```bash
python run_app.py
```

- Select your camera source.
- Visual feedback:
  - 🟢 **Green:** Safe Posture.
  - 🟠 **Orange:** Warning (Bad Form).
  - 🔴 **Red:** CRITICAL RISK (Unsafe Lift Detected).

---

## 📊 Performance & Standards

The system is calibrated against international ergonomic standards:

| Zone | Angle (Flexion) | Bio-Mechanical Context | Alert Level |
|------|----------------|------------------------|-------------|
| **Neutral** | 0° - 20° | Natural standing/walking. | ✅ Safe |
| **Mild Flexion** | 20° - 60° | Acceptable for short durations. | 🟡 Monitor |
| **Severe Flexion** | > 60° | EAWS Class 4. High disc compression. | 🔴 Critical |

### Current Model Metrics:
- **Validation Accuracy:** ~92.2%
- **Inference Speed:** ~30 FPS (CPU), ~90 FPS (GPU)
- **Latency:** < 50ms (Real-time)

---

## 🔮 Future Roadmap

- [ ] **Multi-Person Tracking:** Upgrade from MediaPipe to RTMPose for crowd monitoring.
- [ ] **Cloud Dashboard:** Send critical alert logs to a Flask/FastAPI server for safety managers.
- [ ] **Edge Deployment:** Optimize model (ONNX/TensorRT) for NVIDIA Jetson devices.

---

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

<p align="center">
<i>Built with ❤️ for Worker Safety</i>
</p>