import sys
import logging
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox
from PyQt6.QtCore import Qt, QThread
from PyQt6.QtGui import QPixmap

# Imports
from ui.style import STYLESHEET
from ui.camUtils import find_available_cameras
from ui.videoThread import CameraWorker     # The Dumb Capture
from ui.inference_worker import InferenceWorker # The Smart Processor

# Setup Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename='logs/app.log', 
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Industrial Safety: Posture Analysis")
        self.setGeometry(100, 100, 1280, 720)
        self.setStyleSheet(STYLESHEET)
        
        self.camera_thread = None
        self.inference_thread = None
        self.inference_worker = None
        
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)
        
        # --- Video Feeds ---
        self.raw_label = QLabel("Camera Feed")
        self.model_label = QLabel("Analysis Feed")
        self.raw_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.model_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.raw_label.setStyleSheet("background-color: black; border: 1px solid gray;")
        self.model_label.setStyleSheet("background-color: black; border: 1px solid gray;")
        
        main_layout.addWidget(self.raw_label)
        main_layout.addWidget(self.model_label)
        
        # --- Controls ---
        self.combo = QComboBox()
        self.combo.addItems(find_available_cameras())
        main_layout.addWidget(self.combo)
        
        self.btn_start = QPushButton("Start Analysis")
        self.btn_start.clicked.connect(self.start_app)
        main_layout.addWidget(self.btn_start)
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_app)
        main_layout.addWidget(self.btn_stop)
        
        # --- Status Bar ---
        self.statusBar().showMessage("Ready. Connect to Server before starting.")

    def start_app(self):
        # Parse Camera Index
        cam_text = self.combo.currentText()
        if "No Cameras" in cam_text:
            self.statusBar().showMessage("Error: No camera found.")
            return
        idx = int(cam_text.split()[-1])
        
        # 1. Create Workers
        self.camera_thread = CameraWorker(idx)
        self.inference_thread = QThread()
        self.inference_worker = InferenceWorker()
        self.inference_worker.moveToThread(self.inference_thread)
        
        # 2. Connect Signals
        # Camera -> Inference
        self.camera_thread.frame_captured.connect(self.inference_worker.process_frame)
        
        # Inference -> GUI
        self.inference_worker.update_raw_feed.connect(self.set_raw)
        self.inference_worker.update_model_feed.connect(self.set_model)
        self.inference_worker.server_status.connect(self.update_status)
        
        # Cleanup
        self.camera_thread.finished.connect(self.camera_thread.deleteLater)
        self.inference_thread.finished.connect(self.inference_thread.deleteLater)
        
        # 3. Start
        self.inference_thread.start()
        self.camera_thread.start()
        
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.combo.setEnabled(False)
        self.statusBar().showMessage("System Running...")

    def stop_app(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.wait()
        
        if self.inference_worker:
            self.inference_worker.stop()
            
        if self.inference_thread:
            self.inference_thread.quit()
            self.inference_thread.wait()
            
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.combo.setEnabled(True)
        self.statusBar().showMessage("Stopped.")

    def set_raw(self, img): 
        self.raw_label.setPixmap(QPixmap.fromImage(img).scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio))
    
    def set_model(self, img): 
        self.model_label.setPixmap(QPixmap.fromImage(img).scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio))
        
    def update_status(self, msg):
        self.statusBar().showMessage(f"Server Status: {msg}")

    def closeEvent(self, event):
        self.stop_app()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())