import cv2
import torch
import mediapipe as mp
import numpy as np

# Load the custom YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp7/weights/best.pt', force_reload=True)

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Video path
video_path = 'video/demo_serve3.mp4'
cap = cv2.VideoCapture(video_path)

# Adjusted parameters
semicircle_radius = 450
waist_margin = 70
contact_distance_threshold = 50

paused = False

# Helper function to calculate Euclidean distance
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Helper function to calculate angle
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

# Calculate waist center
def calculate_waist_center(landmarks_ref, frame_shape, margin=30):
    right_hip = landmarks_ref[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_hip = landmarks_ref[mp_pose.PoseLandmark.LEFT_HIP.value]
    waist_x = int((right_hip.x + left_hip.x) / 2 * frame_shape[1])
    waist_y = int((right_hip.y + left_hip.y) / 2 * frame_shape[0]) - margin
    return waist_x, waist_y

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        yolo_results = model(frame)

        if hasattr(results, 'pose_landmarks') and results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
            waist_center_x, waist_center_y = calculate_waist_center(landmarks, frame.shape, waist_margin)

            # Draw semicircle around waist center
            cv2.ellipse(frame, (waist_center_x, waist_center_y), (semicircle_radius, semicircle_radius), 0, 0, 180, (255, 255, 0), 2)

            # Extract wrist, elbow, and shoulder positions
            wrist_pos = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * frame.shape[1],
                                  landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * frame.shape[0]])
            elbow_pos = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * frame.shape[1],
                                  landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * frame.shape[0]])
            shoulder_pos = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]])
            waist_center_pos = np.array([waist_center_x, waist_center_y])

            # Calculate wrist-to-waist distance
            wrist_waist_distance = calculate_distance(wrist_pos, waist_center_pos)
            wrist_y_position = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * frame.shape[0]

            # Calculate angle at the elbow
            arm_angle = calculate_angle(shoulder_pos, elbow_pos, wrist_pos)

            # Determine legality based on conditions
            serve_text = "LEGAL SERVE" if wrist_y_position < waist_center_y + semicircle_radius and arm_angle > 150 else "ILLEGAL SERVE"
            color = (0, 255, 0) if serve_text == "LEGAL SERVE" else (0, 0, 255)

            # Display legality text just above the waist info
            cv2.putText(frame, serve_text, (50, frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Debugging text with coordinates and measurements
            debug_text = f"Waist Center: ({waist_center_x}, {waist_center_y})\nWrist Y: {int(wrist_y_position)}\n" \
                         f"Wrist-Waist Dist: {int(wrist_waist_distance)}\nArm Angle: {int(arm_angle)}"
            for idx, line in enumerate(debug_text.split('\n')):
                y_pos = frame.shape[0] - (20 * (len(debug_text.split('\n')) - idx))
                cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imshow("Pickleball Detection with YOLOv5", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused

    while paused:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            paused = False
            cap.release()
            cv2.destroyAllWindows()
            break
        elif key == ord(' '):
            paused = False

cap.release()
cv2.destroyAllWindows()