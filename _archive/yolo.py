import cv2
from ultralytics import YOLO
import os 
import datetime

# --- Main Application ---

# Load the YOLOv11-pose model
model = YOLO("yolo11n-pose.pt") 

# Open the webcam (using your new camera index 2)
cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# --- Get video properties for saving ---
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0: # Handle cases where FPS is not reported
    fps = 20 

# --- Setup for saving the video ---
save_dir = "poseVideos"
# This creates the "poseVideos" directory if it doesn't already exist
os.makedirs(save_dir, exist_ok=True) 

# Get the current time and format it (using your format ddmmYYYY_HHMMSS)
now = datetime.datetime.now()
timestamp = now.strftime("%d%m%Y_%H%M%S") 

# Create a unique filename
filename = f"pose_{timestamp}.avi"
save_path = os.path.join(save_dir, filename)

# Define the codec and create the VideoWriter object with the new path
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))

# Print one clear message
print(f"Camera opened. Saving to '{save_path}'. Press 'q' to quit.")


while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Run inference
        results = model(frame, stream=True, verbose=False)

        annotated_frame = None 

        for result in results:
            # result.plot() is the default Ultralytics function 
            # to draw the pose, boxes, etc.
            annotated_frame = result.plot() 
            
            # --- All angle logic has been removed ---

        # Check if an annotated frame was created
        if annotated_frame is not None:
            # Write the frame to the video file
            out.write(annotated_frame)
            
            # Display the annotated frame in the pop-up window
            cv2.imshow("YOLOv11 Real-Time Pose Estimation", annotated_frame)
        else:
            # If no results, just show/save the original frame
            out.write(frame)
            cv2.imshow("YOLOv11 Real-Time Pose Estimation", frame)


        # Quit the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the video stream ends
        break

# Release everything when done
cap.release()
out.release() # Release the video writer
cv2.destroyAllWindows()

print(f"Video saved successfully as '{save_path}'")
