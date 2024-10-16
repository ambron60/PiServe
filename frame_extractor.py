import cv2
import os
from tqdm import tqdm

# Path to the video file
# video_path = 'raw/pb_match.mp4' # add your own match video here

# Directory to save the extracted frames
output_dir = 'raw/extracted_frames'
os.makedirs(output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get the total number of frames in the video for progress tracking
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Set the interval to extract frames (every 30th frame, for example)
frame_interval = 30

# Frame count
frame_count = 0

# Progress bar with total number of frames divided by interval
with tqdm(total=total_frames // frame_interval, desc="Extracting frames") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Save every nth frame
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f'frame_{frame_count}.jpg')
            cv2.imwrite(frame_filename, frame)
            pbar.update(1)  # Update the progress bar

        frame_count += 1

# Release the video capture object
cap.release()

print(f"Extracted {frame_count // frame_interval} frames to {output_dir}")