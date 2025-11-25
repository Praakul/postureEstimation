import cv2
import mediapipe as mp
import numpy as np
import time
import logging
from collections import deque

from utils.normalization import normalize_skeleton
from utils.filter import OneEuroFilter

# --- SAFETY LOGIC CLASS (The "Debouncer") ---
class SafetyLogic:
    def __init__(self, fps=30):
        self.fps = fps
        # Rolling buffer of recent predictions (e.g., last 2 seconds)
        self.history = deque(maxlen=self.fps * 2) 
        self.cooldown = 0 # Frames to wait before next alert

    def update(self, prediction_class):
        """
        Updates history and returns a stable alert status.
        prediction_class: 0 (Safe), 1 (Warn), 2 (Critical)
        """
        self.history.append(prediction_class)
        
        # Decrement cooldown
        if self.cooldown > 0:
            self.cooldown -= 1
            return None # Suppress alerts during cooldown

        # Logic: Require sustained bad posture (e.g., > 25 frames in last 30)
        # This prevents a single glitch frame from triggering an alarm.
        if len(self.history) >= 30:
            recent_window = list(self.history)[-30:]
            
            critical_count = recent_window.count(2)
            if critical_count > 20: # If >20/30 frames were critical
                self.cooldown = self.fps * 3 # Wait 3 seconds before next scream
                return "CRITICAL"
            
            warning_count = recent_window.count(1)
            if warning_count > 20:
                self.cooldown = self.fps * 5 # Warn less frequently
                return "WARNING"
        
        return "SAFE"

# --- MAIN POSE LOGIC ---
class PoseLogic:
    def __init__(self, model_path="stgcn_posture_model.pth"):
        self.logger = logging.getLogger("PoseLogic")
        
        # 1. MediaPipe Setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 2. AI Model Setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Note: in_channels=6 (3 Pos + 3 Vel)
        self.model = STGCN(num_classes=3, in_channels=6).to(self.device)
        self.use_ai = False
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.use_ai = True
            self.logger.info("AI Model Loaded Successfully.")
        except Exception as e:
            self.logger.warning(f"AI Model not found ({e}). Running in Heuristic Mode.")

        # 3. Inference Buffer
        self.SEQ_LEN = 50
        self.buffer = deque(maxlen=self.SEQ_LEN)
        
        # 4. Smoothing Filters
        self.filters = {} 
        
        # 5. Safety Logic Engine (NEW)
        self.safety_monitor = SafetyLogic(fps=30)
        
        # State
        self.status = "Initializing"
        self.color = (0, 255, 0) # Green

    def get_smoothed_landmarks(self, raw_landmarks):
        """Applies OneEuroFilter to all 33 landmarks."""
        timestamp = time.time()
        smoothed = []
        
        for i, lm in enumerate(raw_landmarks):
            if i not in self.filters:
                self.filters[i] = {
                    'x': OneEuroFilter(timestamp, lm.x),
                    'y': OneEuroFilter(timestamp, lm.y),
                    'z': OneEuroFilter(timestamp, lm.z)
                }
            
            f = self.filters[i]
            s_x = f['x'](timestamp, lm.x)
            s_y = f['y'](timestamp, lm.y)
            s_z = f['z'](timestamp, lm.z)
            
            # Create a simple object to mimic MediaPipe landmark
            class SmoothPoint:
                def __init__(self, x, y, z, v):
                    self.x, self.y, self.z, self.visibility = x, y, z, v
            
            smoothed.append(SmoothPoint(s_x, s_y, s_z, lm.visibility))
            
        return smoothed

    def get_coco17_skeleton(self, landmarks):
        indices = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        points = []
        for i in indices:
            lm = landmarks[i]
            points.append([lm.x, lm.y, lm.z])
        return np.array(points)

    def calculate_angle(self, a, b, c):
        a = np.array([a.x, a.y, a.z])
        b = np.array([b.x, b.y, b.z])
        c = np.array([c.x, c.y, c.z])
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle

    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        
        pred_class = 0 # Default Safe
        
        if results.pose_landmarks:
            # 1. Smooth Data
            raw_lms = results.pose_landmarks.landmark
            lms = self.get_smoothed_landmarks(raw_lms)
            
            # 2. Context: Lifting? (Wrist below Knee)
            left_wrist_y = lms[15].y
            left_knee_y = lms[25].y
            is_lifting = left_wrist_y > left_knee_y

            # 3. AI Inference Prep
            skel = self.get_coco17_skeleton(lms)
            norm_skel = normalize_skeleton(skel)
            self.buffer.append(norm_skel)
            
            # 4. Run AI
            if self.use_ai and len(self.buffer) == self.SEQ_LEN:
                pos_seq = np.array(self.buffer)
                vel_seq = np.diff(pos_seq, axis=0, prepend=pos_seq[0:1])
                full_seq = np.concatenate((pos_seq, vel_seq), axis=2)
                
                inp = torch.tensor(full_seq, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    probs = self.model(inp)
                    pred_class = torch.argmax(probs, dim=1).item()

            # 5. Geometric Verification (Fallback/Override)
            # Stoop check: Torso bent (<135) but Knees straight (>150)
            torso_angle = self.calculate_angle(lms[11], lms[23], lms[25])
            knee_angle = self.calculate_angle(lms[23], lms[25], lms[27])
            
            is_stooping = (torso_angle < 135) and (knee_angle > 150)
            
            # Override AI if Physics detects a stoop
            if is_stooping:
                pred_class = 2
            elif pred_class == 2 and not is_lifting:
                # AI thinks it's bad, but hands are high? Maybe just stretching.
                pred_class = 1 

            # 6. Safety Logic (Debouncing)
            stable_status = self.safety_monitor.update(pred_class)
            
            if stable_status == "CRITICAL":
                self.status = "CRITICAL: UNSAFE LIFT"
                self.color = (0, 0, 255) # Red
                # Here you would trigger: send_email() or play_sound()
            elif stable_status == "WARNING":
                self.status = "Warning: Bad Form"
                self.color = (0, 165, 255) # Orange
            elif stable_status == "SAFE":
                self.status = "Safe"
                self.color = (0, 255, 0) # Green

            # 7. Draw
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Dashboard
            cv2.rectangle(frame, (0,0), (450, 80), self.color, -1)
            cv2.putText(frame, self.status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            
            # Debug Info
            debug_text = f"Torso: {int(torso_angle)} | Lift: {is_lifting}"
            cv2.putText(frame, debug_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            
        return frame