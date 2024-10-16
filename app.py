import cv2
import torch
import mediapipe as mp

# Load the custom YOLOv5 model (trained for ball and paddle detection)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp7/weights/best.pt', force_reload=True)

# Initialize MediaPipe Pose model for wrist and hip tracking
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Path to the test video
video_path = 'video/demo_serve3.mp4'
cap = cv2.VideoCapture(video_path)

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

            # Extract wrist landmarks
            landmarks = results.pose_landmarks.landmark
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

            # Compare wrist position with paddle and ball position (from YOLO detections)
            for *box, conf, cls in yolo_results.xyxy[0]:
                if int(cls) == 0:  # Assuming class 0 corresponds to 'ball'
                    x1, y1, x2, y2 = map(int, box)  # Ball bounding box coordinates
                    ball_y = y2  # Bottom of the ball box (y-coordinate)

                    # Draw a bounding box around the detected ball
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                elif int(cls) == 1:  # Assuming class 1 corresponds to 'paddle'
                    x1, y1, x2, y2 = map(int, box)  # Paddle bounding box coordinates
                    paddle_y = y2  # Bottom of the paddle box (y-coordinate)

                    # Check if the paddle is below the wrist
                    if paddle_y > right_wrist.y * frame.shape[0]:
                        print("Paddle is below the wrist")
                    else:
                        print("Paddle is above the wrist")

                    # Draw a bounding box around the detected paddle
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

# Release the video capture
cap.release()
cv2.destroyAllWindows()