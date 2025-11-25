import cv2
import logging
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

class CameraWorker(QThread):
    """
    Responsible ONLY for capturing frames from the hardware.
    """
    # Emits the raw numpy array (OpenCV format)
    frame_captured = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self._is_running = True
        self.logger = logging.getLogger("CameraWorker")

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        # Low latency setting
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            self.error_occurred.emit(f"Cannot open camera {self.camera_index}")
            return

        while self._is_running:
            ret, frame = cap.read()
            if not ret:
                self.error_occurred.emit("Frame drop or camera disconnected")
                break
            
            # Emit the raw frame immediately. No processing.
            self.frame_captured.emit(frame)
            
            # Self-throttling is handled by camera hardware FPS (usually 30)
            # but we can add a tiny sleep if CPU usage spikes
            self.msleep(1)

        cap.release()
        self.logger.info("Camera stopped")

    def stop(self):
        self._is_running = False