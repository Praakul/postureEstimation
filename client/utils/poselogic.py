import cv2
import mediapipe as mp
import numpy as np
import time
import logging
from collections import deque

from utils.normalization import normalize_skeleton
from utils.filtering import OneEuroFilter

# --- SAFETY LOGIC CLASS ---a
class SafetyLogic:
    def __init__(self, fps=30):
        self.fps = fps
        self.history = deque(maxlen=self.fps * 2) 
        self.cooldown = 0 

    def update(self, prediction_class):
        self.history.append(prediction_class)
        if self.cooldown > 0:
            self.cooldown -= 1
            return None 

        if len(self.history) >= 30:
            recent_window = list(self.history)[-30:]
            critical_count = recent_window.count(2)
            if critical_count > 20:
                self.cooldown = self.fps * 3
                return "CRITICAL"
            warning_count = recent_window.count(1)
            if warning_count > 20:
                self.cooldown = self.fps * 5 
                return "WARNING"
        return "SAFE"

# --- MAIN POSE LOGIC ---
class PoseLogic:
    def __init__(self):
        self.logger = logging.getLogger("PoseLogic")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.use_ai = False 
        self.SEQ_LEN = 50
        self.buffer = deque(maxlen=self.SEQ_LEN)
        self.filters = {} 
        self.safety_monitor = SafetyLogic(fps=30)
        self.status = "Initializing"
        self.color = (0, 255, 0) 

    def get_smoothed_landmarks(self, raw_landmarks):
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
        # 1. CRITICAL FIX: Create a separate copy for drawing immediately
        # We will read from 'frame' but draw on 'annotated_img'
        annotated_img = frame.copy()
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            raw_lms = results.pose_landmarks.landmark
            lms = self.get_smoothed_landmarks(raw_lms)
            
            # Context Logic
            left_wrist_y = lms[15].y
            left_knee_y = lms[25].y
            is_lifting = left_wrist_y > left_knee_y

            # Buffer Logic
            skel = self.get_coco17_skeleton(lms)
            norm_skel = normalize_skeleton(skel)
            self.buffer.append(norm_skel)
            
            # Geometric Logic
            torso_angle = self.calculate_angle(lms[11], lms[23], lms[25])
            knee_angle = self.calculate_angle(lms[23], lms[25], lms[27])
            is_stooping = (torso_angle < 135) and (knee_angle > 150)
            
            # Draw on the COPY, never the original
            self.mp_drawing.draw_landmarks(
                annotated_img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            cv2.rectangle(annotated_img, (0,0), (450, 80), self.color, -1)
            cv2.putText(annotated_img, self.status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            
            debug_text = f"Torso: {int(torso_angle)} | Lift: {is_lifting}"
            cv2.putText(annotated_img, debug_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            
        return annotated_img