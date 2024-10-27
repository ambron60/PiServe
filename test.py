import cv2
import torch
import mediapipe as mp
import numpy as np
from collections import deque

# Load the custom YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp7/weights/best.pt', force_reload=True)

# Initialize MediaPipe Pose model for wrist and hip tracking
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Path to the test video
video_path = 'video/demo_serve5.mp4'
cap = cv2.VideoCapture(video_path)

# Smoothing window for pose coordinates
smoothing_window = 5
wrist_y_smooth = deque(maxlen=smoothing_window)

# Margin above the hip level (tunable parameter)
hip_margin = 0.02

# Helper function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

paused = False

# Process the video frame by frame
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

            # Extract key landmarks (hips, right wrist, right elbow, and right shoulder)
            landmarks = results.pose_landmarks.landmark
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

            # Calculate the hip level (average of left and right hip y-coordinates) with a margin
            hip_level_y = (right_hip.y + left_hip.y) / 2 * frame.shape[0]
            hip_level_y_with_margin = hip_level_y + (hip_margin * frame.shape[0])

            # Smooth wrist y-coordinate over a few frames
            wrist_y_smooth.append(right_wrist.y * frame.shape[0])
            avg_wrist_y = np.mean(wrist_y_smooth)

            # Calculate the shoulder-elbow-wrist angle
            arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            # Calculate the triangle angle between hips and wrist
            hip_wrist_angle = calculate_angle(left_hip, right_hip, right_wrist)

            # Adjust thresholds based on triangulation
            legal_threshold_angle = 140  # Adjust angle if needed
            triangulation_threshold = 120  # Expected triangle angle for legal serve

            # Compare the paddle grip base position with the hip level
            for *box, conf, cls in yolo_results.xyxy[0]:
                if int(cls) == 1:  # Assuming class 1 corresponds to 'paddle'
                    x1, y1, x2, y2 = map(int, box)  # Paddle bounding box coordinates
                    grip_base_y = y2  # Bottom of the paddle box (y-coordinate)

                    # Check if the grip base is below or level with the hip line
                    if grip_base_y >= hip_level_y_with_margin and arm_angle > legal_threshold_angle and hip_wrist_angle < triangulation_threshold:
                        cv2.putText(frame, "LEGAL SERVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "ILLEGAL SERVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    # Draw a bounding box around the detected paddle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Paddle", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Display calculated angles for debugging
            cv2.putText(frame, f"Arm Angle: {int(arm_angle)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Hip-Wrist Angle: {int(hip_wrist_angle)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display the frame with detections
    cv2.imshow("Pickleball Detection with YOLOv5 and Stance Evaluation", frame)

    # Pause or quit the video
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused

# Release the video capture
cap.release()
cv2.destroyAllWindows()