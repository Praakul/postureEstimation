import cv2
import mediapipe as mp
import numpy as np
import torch
import time
import logging
from collections import deque

# Imports based on your folder structure
from nn.model import STGCN
from utils.normalization import normalize_skeleton
from utils.filtering import OneEuroFilter

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
        # NOTE: Assuming 6 channels (Pos + Vel) based on our Pro Training plan
        self.model = STGCN(num_classes=3, in_channels=6).to(self.device)
        self.use_ai = False
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.use_ai = True
            self.logger.info("AI Model Loaded Successfully.")
        except Exception as e:
            self.logger.warning(f"AI Model not found ({e}). Running in Heuristic Mode.")

        # 3. Inference Buffer (Position + Velocity)
        self.SEQ_LEN = 50
        self.buffer = deque(maxlen=self.SEQ_LEN) # Stores Normalized Skeletons (17, 3)
        
        # 4. ONE EURO FILTERS (Smoothing)
        # We map Index -> {x: Filter, y: Filter, z: Filter}
        self.filters = {} 
        
        # 5. State
        self.status = "Initializing"
        self.color = (0, 255, 0)
        self.alarm_counter = 0

    def get_smoothed_landmarks(self, raw_landmarks):
        """
        Applies OneEuroFilter to all 33 MediaPipe landmarks.
        Returns a list of objects similar to MediaPipe's output, but smoothed.
        """
        timestamp = time.time()
        smoothed = []
        
        for i, lm in enumerate(raw_landmarks):
            # Initialize filters for this landmark if not exists
            if i not in self.filters:
                self.filters[i] = {
                    'x': OneEuroFilter(timestamp, lm.x),
                    'y': OneEuroFilter(timestamp, lm.y),
                    'z': OneEuroFilter(timestamp, lm.z)
                }
            
            # Apply Filter
            f = self.filters[i]
            s_x = f['x'](timestamp, lm.x)
            s_y = f['y'](timestamp, lm.y)
            s_z = f['z'](timestamp, lm.z)
            
            # Store in a simple object structure to mimic MediaPipe
            class SmoothPoint:
                def __init__(self, x, y, z, v):
                    self.x, self.y, self.z, self.visibility = x, y, z, v
            
            smoothed.append(SmoothPoint(s_x, s_y, s_z, lm.visibility))
            
        return smoothed

    def get_coco17_skeleton(self, landmarks):
        """Extracts 17 COCO points from smoothed landmarks list."""
        indices = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        points = []
        for i in indices:
            lm = landmarks[i]
            points.append([lm.x, lm.y, lm.z])
        return np.array(points)

    def calculate_angle(self, a, b, c):
        """Helper for Heuristic checks."""
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
        
        frame_status = "OK"
        frame_color = (0, 255, 0)
        
        if results.pose_landmarks:
            # --- 1. SMOOTHING ---
            # This removes the jitter before we do ANY math
            raw_lms = results.pose_landmarks.landmark
            lms = self.get_smoothed_landmarks(raw_lms)
            
            # --- 2. HEURISTIC CONTEXT (LIFTING) ---
            # Using smoothed Y values
            left_wrist_y = lms[15].y
            left_knee_y = lms[25].y
            right_wrist_y = lms[16].y
            right_knee_y = lms[26].y
            
            is_lifting = (left_wrist_y > left_knee_y) or (right_wrist_y > right_knee_y)

            # --- 3. AI INFERENCE PREP ---
            skel = self.get_coco17_skeleton(lms)
            norm_skel = normalize_skeleton(skel) # (17, 3)
            self.buffer.append(norm_skel)
            
            pred_class = 0
            
            # --- 4. RUN AI (If buffer full) ---
            if self.use_ai and len(self.buffer) == self.SEQ_LEN:
                # Convert buffer to array (50, 17, 3)
                pos_seq = np.array(self.buffer)
                
                # Calculate Velocity (Diff) -> (50, 17, 3)
                vel_seq = np.diff(pos_seq, axis=0, prepend=pos_seq[0:1])
                
                # Stack (Pos + Vel) -> (50, 17, 6)
                full_seq = np.concatenate((pos_seq, vel_seq), axis=2)
                
                # Tensor: (1, 6, 50, 17)
                inp = torch.tensor(full_seq, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    probs = self.model(inp)
                    pred_class = torch.argmax(probs, dim=1).item()

            # --- 5. GEOMETRIC FALLBACK (Verification) ---
            # Even if AI says OK, check Bio-Mechanical Rules (Stoop vs Squat)
            # Torso Angle (Shoulder-Hip-Knee)
            torso_angle = self.calculate_angle(lms[11], lms[23], lms[25])
            # Knee Angle (Hip-Knee-Ankle)
            knee_angle = self.calculate_angle(lms[23], lms[25], lms[27])
            
            is_stooping = (torso_angle < 135) and (knee_angle > 150)
            
            # --- 6. FINAL DECISION LOGIC ---
            # Priority: AI Critical > Geometric Stoop > AI Warning
            
            if pred_class == 2:
                frame_status = "CRITICAL: AI DETECTED RISK"
                frame_color = (0, 0, 255) # Red
            elif is_stooping and is_lifting:
                frame_status = "CRITICAL: UNSAFE LIFT (STOOP)"
                frame_color = (0, 0, 255) # Red
            elif pred_class == 1:
                frame_status = "Warning: Bad Posture"
                frame_color = (0, 165, 255) # Orange
            elif is_lifting:
                frame_status = "Lifting (Form OK)"
                frame_color = (255, 255, 0) # Yellow

            # --- 7. DRAWING ---
            # Note: We draw the RAW landmarks because MediaPipe's util expects a specific protobuf object.
            # Drawing custom smoothed points is possible but complex. 
            # The visual smoothness comes from the STABILITY of the Status Text/Color.
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Dashboard
            cv2.rectangle(frame, (0,0), (450, 80), frame_color, -1)
            cv2.putText(frame, frame_status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            
            # Debug Stats
            debug_text = f"Torso: {int(torso_angle)} | Knee: {int(knee_angle)}"
            cv2.putText(frame, debug_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            
        return frame