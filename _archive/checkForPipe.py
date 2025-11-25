import cv2
import mediapipe as mp
import argparse

# --- 1. Setup Command-Line Argument Parsing ---
# This allows you to control the script from your terminal
parser = argparse.ArgumentParser(
    description="MediaPipe Pose & Holistic (Body, Face, Hands) Tester"
)
parser.add_argument(
    "--no-hands", action="store_true", help="Disable hand tracking to improve performance."
)
parser.add_argument(
    "--no-face", action="store_true", help="Disable face mesh tracking to improve performance."
)
args = parser.parse_args()


# --- 2. Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

# --- 3. Determine which model to run ---
# If BOTH hands and face are disabled, we run the much faster
# dedicated POSE model. Otherwise, we run the full HOLISTIC model.
use_pose_only = args.no_hands and args.no_face

if use_pose_only:
    print("Running in POSE-ONLY (Fast) mode.")
    # Initialize the Pose model
    model = mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
else:
    print("Running in HOLISTIC (Pose, Hands, Face) mode.")
    # Initialize the Holistic model
    model = mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

# --- 4. Start Webcam Capture ---
cap = cv2.VideoCapture(0)

# The main `with` block handles model loading
with model as processor:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a selfie-view display
        image = cv2.flip(image, 1)

        # To improve performance, mark the image as non-writeable to
        # pass by reference.
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and get results
        results = processor.process(image_rgb)

        # Convert back to BGR for OpenCV drawing
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # --- 5. Draw Annotations Based on Mode ---

        if use_pose_only:
            # POSE-ONLY drawing
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )
        else:
            # HOLISTIC drawing (with checks for disabled parts)
            
            # Draw Face (if not disabled)
            if not args.no_face:
                mp_drawing.draw_landmarks(
                    image,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                )

            # Draw Pose (always included in holistic)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )

            # Draw Hands (if not disabled)
            if not args.no_hands:
                # Left Hand
                mp_drawing.draw_landmarks(
                    image,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                )
                # Right Hand
                mp_drawing.draw_landmarks(
                    image,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                )

        # --- 6. Display the Image ---
        cv2.imshow("MediaPipe Posture Analysis", image)

        # Exit on 'ESC' key
        if cv2.waitKey(5) & 0xFF == 27:
            break

# --- 7. Cleanup ---
cap.release()
cv2.destroyAllWindows()