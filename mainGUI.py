import sys
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QComboBox, QFrame, QSizePolicy,
    QStatusBar, QMessageBox, QApplication
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QSize
from ultralytics import YOLO

# Import our modularized code
from style import STYLESHEET
from videoThread import VideoThread
from camUtils import find_available_cameras

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Industrial Pose Analysis Dashboard")
        self.setGeometry(50, 50, 1600, 900) # Standard 16:9 ratio
        self.setStyleSheet(STYLESHEET)
        
        self.model = None
        self.video_thread = None

        self.load_model()
        self.init_ui()
        self.find_cameras_and_populate()

    def load_model(self):
        try:
            self.model = YOLO("yolo11n-pose.pt")
            print("YOLO model loaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Model Error", f"Could not load YOLO model: {e}")
            sys.exit(1)

    def find_cameras_and_populate(self):
        camera_list = find_available_cameras()
        self.camera_combo.addItems(camera_list)
        if "No Cameras Found" in camera_list[0]:
            self.start_analysis_button.setEnabled(False)

    def init_ui(self):
        """Sets up the entire user interface."""
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)

        # --- 1. Title ---
        self.title_label = QLabel("Industrial Pose Analysis Dashboard")
        self.title_label.setObjectName("TitleLabel")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.title_label)

        # --- 2. Video Feeds Layout ---
        video_layout = QHBoxLayout()
        main_layout.addLayout(video_layout, 1)

        # Raw Feed
        raw_feed_layout = QVBoxLayout()
        raw_title = QLabel("Raw Camera Feed")
        raw_title.setObjectName("FeedTitleLabel")
        raw_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.raw_feed_label = QLabel("Camera Off")
        self.raw_feed_label.setObjectName("VideoFeedLabel")
        self.raw_feed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.raw_feed_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        raw_feed_layout.addWidget(raw_title)
        raw_feed_layout.addWidget(self.raw_feed_label, 1)
        video_layout.addLayout(raw_feed_layout, 1)

        # Model Feed
        model_feed_layout = QVBoxLayout()
        model_title = QLabel("Analyzed Feed")
        model_title.setObjectName("FeedTitleLabel")
        model_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Container to allow overlaying labels
        model_feed_container = QWidget()
        model_feed_container.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        model_overlay_layout = QVBoxLayout(model_feed_container)
        model_overlay_layout.setContentsMargins(0, 0, 0, 0)

        self.model_feed_label = QLabel("Analysis Off")
        self.model_feed_label.setObjectName("VideoFeedLabel")
        self.model_feed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.model_feed_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        model_overlay_layout.addWidget(self.model_feed_label)
        
        # Indicators (Styling is now in QSS)
        self.pause_label = QLabel("RECORDING PAUSED", self.model_feed_label)
        self.pause_label.setObjectName("PauseLabel")
        self.pause_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pause_label.hide()

        self.recording_indicator = QLabel("● RECORDING", self.model_feed_label)
        self.recording_indicator.setObjectName("RecordingIndicator")
        self.recording_indicator.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.recording_indicator.setContentsMargins(15, 15, 0, 0)
        self.recording_indicator.hide()

        model_feed_layout.addWidget(model_title)
        model_feed_layout.addWidget(model_feed_container, 1)
        video_layout.addLayout(model_feed_layout, 1)

        # --- 3. Control Bar ---
        control_bar = QWidget()
        control_bar.setObjectName("ControlBar")
        control_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        control_layout = QHBoxLayout(control_bar)
        main_layout.addWidget(control_bar)

        control_layout.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        control_layout.addWidget(self.camera_combo)
        
        control_layout.addStretch(1)

        # Analysis Buttons
        self.start_analysis_button = QPushButton("Start Analysis")
        self.start_analysis_button.setObjectName("StartAnalysisButton") # ID for styling
        self.start_analysis_button.clicked.connect(self.start_analysis)
        control_layout.addWidget(self.start_analysis_button)

        self.stop_analysis_button = QPushButton("Stop Analysis")
        self.stop_analysis_button.setObjectName("StopButton") # ID for styling
        self.stop_analysis_button.clicked.connect(self.stop_analysis)
        control_layout.addWidget(self.stop_analysis_button)
        
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        control_layout.addWidget(separator)

        # Recording Buttons
        self.start_record_button = QPushButton("Start Recording")
        self.start_record_button.setObjectName("StartRecordButton")
        self.start_record_button.clicked.connect(self.start_recording)
        control_layout.addWidget(self.start_record_button)
        
        self.pause_record_button = QPushButton("Pause Recording")
        self.pause_record_button.setObjectName("PauseRecordButton")
        self.pause_record_button.clicked.connect(self.toggle_pause_recording)
        control_layout.addWidget(self.pause_record_button)
        
        self.stop_record_button = QPushButton("Stop Recording")
        self.stop_record_button.setObjectName("StopRecordButton")
        self.stop_record_button.clicked.connect(self.stop_recording)
        control_layout.addWidget(self.stop_record_button)
        
        control_layout.addStretch(1)

        # Quit Button
        self.quit_button = QPushButton("Close Application")
        self.quit_button.setObjectName("QuitButton")
        self.quit_button.clicked.connect(self.close)
        control_layout.addWidget(self.quit_button)
        
        # --- 4. Status Bar ---
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready. Select a camera and click 'Start Analysis'.")
        
        # Set initial button states
        self.update_button_states(is_running=False, is_recording=False, is_paused=False)

    # --- UI Update & Control Functions ---

    def update_button_states(self, is_running, is_recording, is_paused):
        """
        Manages the enabled/disabled state of all buttons.
        All styling is now handled by the stylesheet.
        """
        self.start_analysis_button.setEnabled(not is_running)
        self.stop_analysis_button.setEnabled(is_running)
        self.camera_combo.setEnabled(not is_running)

        self.start_record_button.setEnabled(is_running and not is_recording)
        self.stop_record_button.setEnabled(is_running and is_recording)
        self.pause_record_button.setEnabled(is_running and is_recording)
        
        if is_running and is_recording:
            if is_paused:
                self.pause_record_button.setText("Resume")
                self.pause_label.show()
                self.recording_indicator.hide()
            else:
                self.pause_record_button.setText("Pause")
                self.pause_label.hide()
                self.recording_indicator.show()
        else:
            self.pause_record_button.setText("Pause")
            self.pause_label.hide()
            self.recording_indicator.hide()

    def start_analysis(self):
        cam_text = self.camera_combo.currentText()
        if "No Cameras Found" in cam_text:
            QMessageBox.warning(self, "Camera Error", "No cameras found. Please connect a camera.")
            return
            
        cam_index = int(cam_text.split()[-1])
        
        self.video_thread = VideoThread(cam_index, self.model)
        # Connect signals
        self.video_thread.change_pixmap_raw.connect(self.set_raw_image)
        self.video_thread.change_pixmap_model.connect(self.set_model_image)
        self.video_thread.error_signal.connect(self.on_video_error)
        self.video_thread.video_saved_signal.connect(self.on_video_saved)
        
        self.video_thread.start()
        
        self.raw_feed_label.setText("Starting camera...")
        self.model_feed_label.setText("Starting model...")
        self.update_button_states(is_running=True, is_recording=False, is_paused=False)
        self.statusBar().showMessage("Analysis running...")

    def stop_analysis(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait() # Wait for thread to finish
            self.video_thread = None
        
        self.raw_feed_label.setText("Camera Off")
        self.model_feed_label.setText("Analysis Off")
        self.update_button_states(is_running=False, is_recording=False, is_paused=False)
        self.statusBar().showMessage("Ready. Select a camera and click 'Start Analysis'.")

    def start_recording(self):
        if self.video_thread:
            self.video_thread.start_recording()
            self.update_button_states(is_running=True, is_recording=True, is_paused=False)
            self.statusBar().showMessage("Recording started...")

    def toggle_pause_recording(self):
        if self.video_thread:
            if self.pause_record_button.text() == "Pause":
                self.video_thread.pause_recording()
                self.update_button_states(is_running=True, is_recording=True, is_paused=True)
                self.statusBar().showMessage("Recording paused.")
            else:
                self.video_thread.resume_recording()
                self.update_button_states(is_running=True, is_recording=True, is_paused=False)
                self.statusBar().showMessage("Recording resumed...")

    def stop_recording(self):
        if self.video_thread:
            self.video_thread.stop_recording()
            self.update_button_states(is_running=True, is_recording=False, is_paused=False)

    # --- Slots for Signals from VideoThread ---

    def set_raw_image(self, image):
        self.raw_feed_label.setPixmap(QPixmap.fromImage(image).scaled(
            self.raw_feed_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        ))

    def set_model_image(self, image):
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(
            self.model_feed_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.model_feed_label.setPixmap(scaled_pixmap)
        
        # Resize overlays to match the video feed
        self.pause_label.resize(scaled_pixmap.size())
        self.recording_indicator.resize(scaled_pixmap.size())


    def on_video_error(self, error_msg):
        QMessageBox.critical(self, "Video Thread Error", error_msg)
        self.stop_analysis()

    def on_video_saved(self, save_path):
        self.statusBar().showMessage(f"Video saved to {save_path}", 5000) # Show for 5 sec

    def closeEvent(self, event):
        print("Closing application...")
        self.stop_analysis() 
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
