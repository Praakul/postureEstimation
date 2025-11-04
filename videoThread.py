import cv2
import os
import datetime
import time
from PyQt6.QtCore import QThread, pyqtSignal, QSize
from PyQt6.QtGui import QImage

class VideoThread(QThread):
    """
    This thread handles all video capture and processing.
    It emits signals to send frames to the main GUI thread.
    """
    
    # --- Signals ---
    # Emits the raw camera frame (as QImage)
    change_pixmap_raw = pyqtSignal(QImage)
    # Emits the YOLO-annotated frame (as QImage)
    change_pixmap_model = pyqtSignal(QImage)
    # Emits an error message (as str)
    error_signal = pyqtSignal(str)
    # Emits the final save path (as str)
    video_saved_signal = pyqtSignal(str)

    def __init__(self, camera_index, model):
        super().__init__()
        self.camera_index = camera_index
        self.model = model
        self.cap = None
        self.video_writer = None
        self.save_path = ""
        
        # Thread control flags
        self._is_running = True
        self._is_recording = False
        self._is_paused = False

    def run(self):
        """Main loop of the thread."""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            self.error_signal.emit(f"Could not open camera {self.camera_index}.")
            return

        while self._is_running:
            success, frame = self.cap.read()
            if not success:
                self.error_signal.emit("Failed to read frame from camera. Is it disconnected?")
                self._is_running = False
                break

            # --- 1. Raw Frame ---
            rgb_raw_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_raw_image.shape
            bytes_per_line = ch * w
            qt_raw_image = QImage(rgb_raw_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.change_pixmap_raw.emit(qt_raw_image)

            # --- 2. Model Frame (YOLO Analysis) ---
            results = self.model(frame, stream=True, verbose=False)
            
            annotated_frame = None
            for result in results:
                annotated_frame = result.plot()
            
            # Fallback if no pose is detected
            if annotated_frame is None:
                annotated_frame = frame 

            # --- 3. Recording ---
            if self._is_recording and not self._is_paused:
                if self.video_writer is None:
                    self.create_video_writer()
                
                self.video_writer.write(annotated_frame)

            # --- 4. Emit Model Frame ---
            rgb_model_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h_model, w_model, ch_model = rgb_model_image.shape
            bytes_per_line_model = ch_model * w_model
            qt_model_image = QImage(rgb_model_image.data, w_model, h_model, bytes_per_line_model, QImage.Format.Format_RGB888)
            self.change_pixmap_model.emit(qt_model_image)
            
            # Small sleep to prevent 100% CPU usage
            time.sleep(0.01)

        # --- Thread Cleanup ---
        self.cap.release()
        if self.video_writer:
            self.video_writer.release()
            self.video_saved_signal.emit(self.save_path)
        print("VideoThread stopped.")

    def create_video_writer(self):
        """Initializes the CV2 VideoWriter."""
        save_dir = "poseVideos"
        os.makedirs(save_dir, exist_ok=True)
        
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if fps == 0: fps = 20  # Fallback

        now = datetime.datetime.now()
        timestamp = now.strftime("%d%m%Y_%H%M%S")
        filename = f"pose_{timestamp}.avi"
        self.save_path = os.path.join(save_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(self.save_path, fourcc, fps, (frame_width, frame_height))
        print(f"Recording started. Saving to '{self.save_path}'")

    # --- Public Control Methods (called from main thread) ---
    def stop(self):
        self._is_running = False

    def start_recording(self):
        self._is_recording = True
        self._is_paused = False

    def pause_recording(self):
        self._is_paused = True

    def resume_recording(self):
        self._is_paused = False

    def stop_recording(self):
        self._is_recording = False
        self._is_paused = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.video_saved_signal.emit(self.save_path)
