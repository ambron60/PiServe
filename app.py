import cv2
import torch
import mediapipe as mp
import numpy as np

# Load the custom YOLOv5 model (trained for ball and paddle detection)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp7/weights/best.pt', force_reload=True)

# Initialize MediaPipe Pose model for wrist and hip tracking
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Path to the test video
video_path = 'video/demo_serve3.mp4'
cap = cv2.VideoCapture(video_path)

# Adjusted semicircle parameters
semicircle_radius = 500  # Reduced radius for more precision around waist
waist_margin = 50  # Increased margin above waist level

paused = False

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # Convert the frame to RGB for pose estimation
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform pose detection
        results = pose.process(frame_rgb)

        # Use YOLO to detect the paddle and ball in the frame
        yolo_results = model(frame)

        # Extract pose landmarks if available
        if results.pose_landmarks:
            # Draw pose landmarks on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extract key landmarks for waist center
            landmarks = results.pose_landmarks.landmark
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

            # Calculate the waist center position
            waist_center_x = int((right_hip.x + left_hip.x) / 2 * frame.shape[1])
            waist_center_y = int((right_hip.y + left_hip.y) / 2 * frame.shape[0]) - waist_margin

            # Draw the semicircle centered around the waist
            cv2.ellipse(
                frame,
                (waist_center_x, waist_center_y),
                (semicircle_radius, semicircle_radius),
                0, 0, 180,
                (255, 255, 0), 2
            )

            # Extract wrist landmarks
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            wrist_y_norm = right_wrist.y * frame.shape[0]  # Wrist y-position normalized to frame height

            # Use YOLO detections to find the paddle
            for *box, conf, cls in yolo_results.xyxy[0]:
                if int(cls) == 1:  # Assuming class 1 corresponds to 'paddle'
                    x1, y1, x2, y2 = map(int, box)  # Paddle bounding box coordinates
                    paddle_y = y2  # Bottom of the paddle box (y-coordinate)

                    # Check legality based on wrist and paddle positions relative to the semicircle and waist
                    if wrist_y_norm < waist_center_y + semicircle_radius:  # Updated legal zone condition
                        cv2.putText(frame, "LEGAL SERVE", (frame.shape[1] - 200, frame.shape[0] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "ILLEGAL SERVE", (frame.shape[1] - 200, frame.shape[0] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Draw bounding boxes and labels
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Paddle", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the frame with detections
        cv2.imshow("Pickleball Detection with YOLOv5", frame)

        # Pause or quit the video
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused

    # If paused, wait for the next key press
    while paused:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            paused = False
            cap.release()
            cv2.destroyAllWindows()
            break
        elif key == ord(' '):
            paused = False

# Release the video capture
cap.release()
cv2.destroyAllWindows()