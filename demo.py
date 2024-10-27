import cv2
import mediapipe as mp

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

video_path = 'video/demo_serve1.mp4'
cap = cv2.VideoCapture(video_path)

paused = False

# Process the video frame by frame
while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform pose detection
        results = pose.process(frame_rgb)

        # Draw pose landmarks if they are detected
        if results.pose_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extract key landmarks (waist and wrist)
            landmarks = results.pose_landmarks.landmark
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

            # Print wrist and waist Y-coordinates for debugging
            print(f"Left Wrist Y: {left_wrist.y}, Left Hip Y: {left_hip.y}")
            print(f"Right Wrist Y: {right_wrist.y}, Right Hip Y: {right_hip.y}")

    # Display the frame
    cv2.imshow("Pickleball Serve Pose Estimation", frame)

    # Wait for user input and control pause/resume
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):  # Press spacebar to pause/resume
        paused = not paused

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()