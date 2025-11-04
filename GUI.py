import cv2
from ultralytics import YOLO
import os 
import datetime
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
import time

# --- Main Application Class ---
class PoseApp(ctk.CTk):
    
    def __init__(self):
        super().__init__()

        # --- NEW: Font and Color Constants ---
        self.FONT_TITLE = ("Arial", 50, "bold")
        self.FONT_BUTTON = ("Arial", 36, "bold")
        self.FONT_LABEL = ("Arial", 36, "bold")
        self.FONT_OVERLAY = ("Arial", 42, "bold")

        # --- Color Palette (Dark, High-Contrast) ---
        self.COLOR_PRIMARY = ("#00529B", "#00396D") # Dark Blue (fg, hover)
        self.COLOR_SECONDARY = ("#505050", "#303030") # Dark Gray (fg, hover)
        self.COLOR_SUCCESS = ("#006400", "#004D00") # Dark Green (fg, hover)
        self.COLOR_WARNING = ("#CC5500", "#FF8C00") # Dark Orange (fg, hover)
        self.COLOR_DANGER = ("#8B0000", "#B22222") # Dark Red (fg, hover)
        self.TEXT_COLOR = "white"

        # --- Basic App Setup ---
        self.title("Industrial Pose Analysis Dashboard")
        self.geometry("1600x950") # Increased window size
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # --- App State Variables ---
        self.is_analysis_running = False
        self.is_recording = False
        self.is_paused = False
        
        self.cap = None
        self.model = None
        self.video_writer = None
        self.video_thread = None
        
        self.raw_frame = None
        self.bgr_annotated_frame = None 
        self.rgb_annotated_frame = None 

        # --- Load YOLO Model ---
        try:
            self.model = YOLO("yolo11n-pose.pt") 
            print("YOLO model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.quit()

        # --- Find Available Cameras ---
        self.camera_sources = self.find_cameras()
        if not self.camera_sources:
            print("Error: No cameras found. Exiting.")
            self.quit()
        self.selected_camera = ctk.StringVar(value=self.camera_sources[0])

        # --- Configure the GUI Layout ---
        self.grid_rowconfigure(0, weight=0) # Title
        self.grid_rowconfigure(1, weight=1) # Video Feeds
        self.grid_rowconfigure(2, weight=0) # Control Bar
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # --- 1. Title Header ---
        self.title_label = ctk.CTkLabel(self, text="Industrial Pose Analysis Dashboard", 
                                        font=self.FONT_TITLE, text_color="#00AEEF")
        self.title_label.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10))

        # --- 2. Video Feeds ---
        # Frame for Raw Feed
        self.raw_frame_container = ctk.CTkFrame(self)
        self.raw_frame_container.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.raw_frame_container.grid_rowconfigure(0, weight=0)
        self.raw_frame_container.grid_rowconfigure(1, weight=1)
        self.raw_frame_container.grid_columnconfigure(0, weight=1)
        
        self.raw_title = ctk.CTkLabel(self.raw_frame_container, text="Raw Camera Feed", font=self.FONT_LABEL)
        self.raw_title.grid(row=0, column=0, padx=10, pady=10)
        self.raw_feed_label = ctk.CTkLabel(self.raw_frame_container, text="", font=self.FONT_LABEL)
        self.raw_feed_label.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Frame for Model Feed
        self.model_frame_container = ctk.CTkFrame(self)
        self.model_frame_container.grid(row=1, column=1, padx=20, pady=10, sticky="nsew")
        self.model_frame_container.grid_rowconfigure(0, weight=0)
        self.model_frame_container.grid_rowconfigure(1, weight=1)
        self.model_frame_container.grid_columnconfigure(0, weight=1)

        self.model_title = ctk.CTkLabel(self.model_frame_container, text="Analyzed Feed", font=self.FONT_LABEL)
        self.model_title.grid(row=0, column=0, padx=10, pady=10)
        self.model_feed_label = ctk.CTkLabel(self.model_frame_container, text="", font=self.FONT_LABEL)
        self.model_feed_label.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        # Pause Overlay Label
        self.pause_label = ctk.CTkLabel(self.model_frame_container, text="RECORDING PAUSED", 
                                        font=self.FONT_OVERLAY, text_color="white",
                                        fg_color=self.COLOR_DANGER[0], corner_radius=10)
        
        # --- 3. Control Bar ---
        self.control_frame = ctk.CTkFrame(self, height=100)
        self.control_frame.grid(row=2, column=0, columnspan=2, padx=20, pady=20, sticky="nsew")

        # Configure columns to space out controls
        self.control_frame.grid_columnconfigure(0, weight=1) 
        self.control_frame.grid_columnconfigure(1, weight=0) 
        self.control_frame.grid_columnconfigure(2, weight=0) 
        self.control_frame.grid_columnconfigure(3, weight=0) 
        self.control_frame.grid_columnconfigure(4, weight=0) 
        self.control_frame.grid_columnconfigure(5, weight=1) 
        self.control_frame.grid_columnconfigure(6, weight=0) 

        # Camera Selection
        self.cam_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        self.cam_frame.grid(row=0, column=1, padx=10)
        self.camera_label = ctk.CTkLabel(self.cam_frame, text="Select Camera:", font=self.FONT_LABEL)
        self.camera_label.pack(side="left", padx=5)
        self.camera_menu = ctk.CTkOptionMenu(self.cam_frame, values=self.camera_sources, 
                                             variable=self.selected_camera, font=self.FONT_BUTTON,
                                             button_color=self.COLOR_PRIMARY[0], 
                                             button_hover_color=self.COLOR_PRIMARY[1],
                                             text_color=self.TEXT_COLOR)
        self.camera_menu.pack(side="left", padx=5)

        # Analysis Controls
        self.analysis_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        self.analysis_frame.grid(row=0, column=2, padx=10)
        self.start_analysis_button = ctk.CTkButton(self.analysis_frame, text="Start Analysis", 
                                                   command=self.start_analysis, font=self.FONT_BUTTON, height=45,
                                                   fg_color=self.COLOR_PRIMARY[0], hover_color=self.COLOR_PRIMARY[1],
                                                   text_color=self.TEXT_COLOR)
        self.start_analysis_button.pack(side="left", padx=5)
        self.stop_analysis_button = ctk.CTkButton(self.analysis_frame, text="Stop Analysis", 
                                                  command=self.stop_analysis, font=self.FONT_BUTTON, height=45, 
                                                  state="disabled", fg_color=self.COLOR_SECONDARY[0], 
                                                  hover_color=self.COLOR_SECONDARY[1], text_color=self.TEXT_COLOR)
        self.stop_analysis_button.pack(side="left", padx=5)
        
        # Separator
        self.separator = ctk.CTkFrame(self.control_frame, width=2, fg_color="gray")
        self.separator.grid(row=0, column=3, padx=15, sticky="ns")

        # Recording Controls
        self.record_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        self.record_frame.grid(row=0, column=4, padx=10)
        
        self.start_record_button = ctk.CTkButton(self.record_frame, text="Start Recording", 
                                                 command=self.start_recording, font=self.FONT_BUTTON, height=45, 
                                                 state="disabled", fg_color=self.COLOR_SUCCESS[0], 
                                                 hover_color=self.COLOR_SUCCESS[1], text_color=self.TEXT_COLOR)
        self.start_record_button.pack(side="left", padx=5)
        
        self.pause_record_button = ctk.CTkButton(self.record_frame, text="Pause Recording", 
                                                 command=self.toggle_pause_recording, font=self.FONT_BUTTON, height=45, 
                                                 state="disabled", fg_color=self.COLOR_WARNING[0], 
                                                 hover_color=self.COLOR_WARNING[1], text_color=self.TEXT_COLOR)
        self.pause_record_button.pack(side="left", padx=5)
        
        self.stop_record_button = ctk.CTkButton(self.record_frame, text="Stop Recording", 
                                                command=self.stop_recording, font=self.FONT_BUTTON, height=45, 
                                                state="disabled", fg_color=self.COLOR_DANGER[0], 
                                                hover_color=self.COLOR_DANGER[1], text_color=self.TEXT_COLOR)
        self.stop_record_button.pack(side="left", padx=5)

        # Quit Button
        self.quit_button = ctk.CTkButton(self.control_frame, text="Close Application", 
                                         command=self.on_closing, font=self.FONT_BUTTON, height=45,
                                         fg_color=self.COLOR_DANGER[0], hover_color=self.COLOR_DANGER[1],
                                         text_color=self.TEXT_COLOR)
        self.quit_button.grid(row=0, column=6, padx=20)

        # Set protocol for window close button
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    # --- Helper Functions ---

    def find_cameras(self):
        """Checks for available camera indices."""
        index = 0
        arr = []
        while index < 5: 
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                arr.append(f"Camera {index}")
                cap.release()
            index += 1
        if not arr:
            arr.append("No Cameras Found")
        return arr

    def create_video_writer(self):
        """Creates a new VideoWriter object with a unique timestamp."""
        save_dir = "poseVideos"
        os.makedirs(save_dir, exist_ok=True) 
        
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if fps == 0: fps = 20 # Default

        now = datetime.datetime.now()
        timestamp = now.strftime("%d%m%Y_%H%M%S") 
        filename = f"pose_{timestamp}.avi"
        self.save_path = os.path.join(save_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(self.save_path, fourcc, fps, (frame_width, frame_height))
        print(f"Recording started. Saving to '{self.save_path}'")

    # --- Core Logic Functions ---

    def start_analysis(self):
        """Starts the video processing thread to show the feeds."""
        if self.is_analysis_running:
            return

        self.is_analysis_running = True
        
        try:
            cam_index = int(self.selected_camera.get().split()[-1])
        except ValueError:
            print("No valid camera selected.")
            self.is_analysis_running = False
            return
            
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera index {cam_index}.")
            self.is_analysis_running = False
            return

        # Update GUI state
        self.start_analysis_button.configure(state="disabled")
        self.stop_analysis_button.configure(state="normal", fg_color=self.COLOR_DANGER[0], hover_color=self.COLOR_DANGER[1])
        self.camera_menu.configure(state="disabled")
        self.start_record_button.configure(state="normal")
        
        self.video_thread = threading.Thread(target=self.video_processing_loop, daemon=True)
        self.video_thread.start()
        
        self.update_gui_frames()

    def stop_analysis(self):
        """Stops the video analysis thread and releases all resources."""
        if not self.is_analysis_running:
            return
        
        if self.is_recording:
            self.stop_recording()

        self.is_analysis_running = False 
        
        if self.video_thread:
            self.video_thread.join(timeout=1.0) 
        
        if self.cap:
            self.cap.release()
            print("Camera released.")
        
        # Reset frames and GUI
        self.raw_frame = None
        self.bgr_annotated_frame = None
        self.rgb_annotated_frame = None
        self.raw_feed_label.configure(image=None)
        self.model_feed_label.configure(image=None)

        # Update button states
        self.start_analysis_button.configure(state="normal")
        self.stop_analysis_button.configure(state="disabled", fg_color=self.COLOR_SECONDARY[0], hover_color=self.COLOR_SECONDARY[1])
        self.camera_menu.configure(state="normal")
        self.start_record_button.configure(state="disabled", text="Start Recording")
        self.pause_record_button.configure(state="disabled", text="Pause Recording")
        self.stop_record_button.configure(state="disabled")
        self.pause_label.place_forget()
        
        print("Analysis stopped.")


    def video_processing_loop(self):
        """The main loop for reading and processing frames. Runs in a separate thread."""
        while self.is_analysis_running:
            if not self.cap.isOpened():
                print("Error: Camera disconnected.")
                self.is_analysis_running = False # Stop the loop
                break
            
            success, frame = self.cap.read()
            if not success:
                print("Error: Failed to read frame.")
                time.sleep(0.1)
                continue

            # Frame 1: Raw Feed
            self.raw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Frame 2: Model Feed
            results = self.model(frame, stream=True, verbose=False)
            
            self.bgr_annotated_frame = None 
            for result in results:
                self.bgr_annotated_frame = result.plot() 
            
            if self.bgr_annotated_frame is None:
                self.bgr_annotated_frame = frame 

            self.rgb_annotated_frame = cv2.cvtColor(self.bgr_annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Handle Recording
            if self.is_recording and not self.is_paused:
                if self.video_writer is None:
                    # Create writer if it doesn't exist
                    self.create_video_writer()
                
                self.video_writer.write(self.bgr_annotated_frame)

            time.sleep(0.01) # Yield to other threads
        
        print("Video processing thread stopped.")

    def update_gui_frames(self):
        """Updates the video frames in the GUI. Runs in the main thread."""
        
        # Update Raw Feed
        if self.raw_frame is not None:
            display_w, display_h = self.get_display_size(self.raw_frame.shape[1], self.raw_frame.shape[0], self.raw_feed_label)
            
            img = Image.fromarray(self.raw_frame)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(display_w, display_h))
            self.raw_feed_label.configure(image=ctk_img, text="") 
            self.raw_feed_label.image = ctk_img 
        
        # Update Model Feed
        if self.rgb_annotated_frame is not None:
            display_w, display_h = self.get_display_size(self.rgb_annotated_frame.shape[1], self.rgb_annotated_frame.shape[0], self.model_feed_label)
            
            img = Image.fromarray(self.rgb_annotated_frame)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(display_w, display_h))
            self.model_feed_label.configure(image=ctk_img, text="")
            self.model_feed_label.image = ctk_img

        if self.is_analysis_running:
            self.after(33, self.update_gui_frames) # ~30 FPS

    def get_display_size(self, frame_w, frame_h, widget):
        """Calculates the new W/H to fit the widget while maintaining aspect ratio."""
        # Get the widget's current size (minus padding)
        widget_w = widget.winfo_width() - 20  # 10px padding on each side
        widget_h = widget.winfo_height() - 20 # 10px padding on each side
        
        if widget_w < 2 or widget_h < 2: 
            return 640, 480 # Default size if widget not fully drawn
        
        frame_aspect = frame_w / frame_h
        widget_aspect = widget_w / widget_h

        if frame_aspect > widget_aspect:
            # Frame is wider than widget, scale by width
            new_w = widget_w
            new_h = int(new_w / frame_aspect)
        else:
            # Frame is taller than widget, scale by height
            new_h = widget_h
            new_w = int(new_h * frame_aspect)
            
        return new_w, new_h

    # --- Recording Control Functions ---

    def start_recording(self):
        self.is_recording = True
        self.is_paused = False
        self.create_video_writer()
        
        self.start_record_button.configure(state="disabled")
        self.pause_record_button.configure(state="normal")
        self.stop_record_button.configure(state="normal")
        self.pause_label.place_forget()

    def toggle_pause_recording(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_record_button.configure(text="Resume Recording")
            print("Recording paused.")
            self.pause_label.place(relx=0.5, rely=0.5, anchor="center")
        else:
            self.pause_record_button.configure(text="Pause Recording")
            print("Recording resumed.")
            self.pause_label.place_forget()

    def stop_recording(self):
        self.is_recording = False
        self.is_paused = False
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            print(f"Video saved successfully to {self.save_path}")
        
        self.start_record_button.configure(state="normal")
        self.pause_record_button.configure(state="disabled", text="Pause Recording")
        self.stop_record_button.configure(state="disabled")
        self.pause_label.place_forget()

    def on_closing(self):
        """Called when the window is closed."""
        print("Closing application...")
        self.stop_analysis() 
        self.destroy() 


# --- Run the Application ---
if __name__ == "__main__":
    app = PoseApp()
    app.mainloop()

