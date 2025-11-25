import sys
import logging
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QMessageBox
from PyQt6.QtCore import Qt
from videoThread import VideoThread
from camUtils import find_available_cameras
from style import STYLESHEET

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
        
        self.thread = None
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)
        
        # Labels
        self.raw_label = QLabel("Camera Feed")
        self.model_label = QLabel("Analysis Feed")
        self.raw_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.model_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.raw_label.setStyleSheet("background-color: black;")
        self.model_label.setStyleSheet("background-color: black;")
        
        main_layout.addWidget(self.raw_label)
        main_layout.addWidget(self.model_label)
        
        # Controls
        self.combo = QComboBox()
        self.combo.addItems(find_available_cameras())
        main_layout.addWidget(self.combo)
        
        self.btn_start = QPushButton("Start Analysis")
        self.btn_start.clicked.connect(self.start_app)
        main_layout.addWidget(self.btn_start)
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_app)
        main_layout.addWidget(self.btn_stop)

    def start_app(self):
        idx = int(self.combo.currentText().split()[-1])
        self.thread = VideoThread(idx)
        self.thread.change_pixmap_raw.connect(self.set_raw)
        self.thread.change_pixmap_model.connect(self.set_model)
        self.thread.start()
        
    def stop_app(self):
        if self.thread:
            self.thread.stop()
            self.thread.wait()

    def set_raw(self, img): self.raw_label.setPixmap(img.fromImage(img).scaled(640, 480))
    def set_model(self, img): self.model_label.setPixmap(img.fromImage(img).scaled(640, 480))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())