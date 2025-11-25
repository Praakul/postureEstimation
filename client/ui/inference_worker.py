import cv2
import numpy as np
import json
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage

# Import Logic & Networking
from utils.poselogic import PoseLogic
from utils.network_client import NetworkClient

SERVER_URL = "ws://localhost:8000/ws/predict"

class InferenceWorker(QObject):
    """
    Responsible for:
    1. Running MediaPipe (Locally)
    2. Sending data to Server
    3. Preparing Images for GUI
    """
    # Signals to update the GUI
    update_raw_feed = pyqtSignal(QImage)
    update_model_feed = pyqtSignal(QImage)
    server_status = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.logic = PoseLogic()
        self.logic.use_ai = False # We use server for AI
        
        # Network Client
        self.client = NetworkClient(SERVER_URL)
        self.client.connect()

    @pyqtSlot(np.ndarray)
    def process_frame(self, frame):
        """
        Slot connected to CameraWorker.frame_captured
        """
        # 1. Check Server for latest prediction
        server_pred = self.client.get_latest_prediction()
        
        # Update Server Status Signal
        if self.client.is_connected():
            self.server_status.emit("Online")
        else:
            self.server_status.emit("Offline")

        # 2. Update Logic State (Color/Text) based on Server
        if server_pred == 2:
            self.logic.status = "CRITICAL (Server)"
            self.logic.color = (0, 0, 255)
        elif server_pred == 1:
            self.logic.status = "Warning (Server)"
            self.logic.color = (0, 165, 255)
        else:
            self.logic.status = "Safe"
            self.logic.color = (0, 255, 0)

        # 3. Run MediaPipe & Draw
        # Note: process_frame modifies the frame in-place or returns copy
        annotated_frame = self.logic.process_frame(frame.copy())

        # 4. Send Data to Server (if buffer ready)
        if len(self.logic.buffer) == self.logic.SEQ_LEN:
            pos_seq = np.array(self.logic.buffer)
            vel_seq = np.diff(pos_seq, axis=0, prepend=pos_seq[0:1])
            full_seq = np.concatenate((pos_seq, vel_seq), axis=2)
            self.client.send_skeleton(full_seq)

        # 5. Convert to QImage for GUI
        self.emit_image(frame, self.update_raw_feed)
        self.emit_image(annotated_frame, self.update_model_feed)

    def emit_image(self, cv_img, signal):
        """Helper to convert CV2 -> QImage"""
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        signal.emit(qt_img)

    def stop(self):
        self.client.close()