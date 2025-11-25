import cv2
import time
import logging
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage
from poseLogic import PoseLogic

class VideoThread(QThread):
    change_pixmap_raw = pyqtSignal(QImage)
    change_pixmap_model = pyqtSignal(QImage)
    error_signal = pyqtSignal(str)

    def __init__(self, camera_index):
        super().__init__()
        self.camera_index = camera_index
        self.logic = PoseLogic()
        self._is_running = True
        self.logger = logging.getLogger("VideoThread")

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        
        # LATENCY FIX: Small buffer
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            self.error_signal.emit(f"Camera {self.camera_index} failed.")
            return

        while self._is_running:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 1. Raw Feed
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_raw = QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)
            self.change_pixmap_raw.emit(qt_raw)
            
            # 2. Inference
            annotated = self.logic.process_frame(frame.copy())
            
            # 3. Model Feed
            rgb_mod = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            qt_mod = QImage(rgb_mod.data, w, h, ch*w, QImage.Format.Format_RGB888)
            self.change_pixmap_model.emit(qt_mod)
            
        cap.release()

    def stop(self):
        self._is_running = False