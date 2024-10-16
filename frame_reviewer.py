import cv2
import os
import shutil

# Directory with extracted frames
input_dir = 'raw/extracted_frames'

# Directory to move the selected frames
output_dir = 'raw/selected_frames'
os.makedirs(output_dir, exist_ok=True)

# Iterate through each frame in the extracted folder
for frame_file in sorted(os.listdir(input_dir)):
    frame_path = os.path.join(input_dir, frame_file)
    frame = cv2.imread(frame_path)

    # Display the frame
    cv2.imshow('Frame Review', frame)

    # Wait for user input
    print(f"Press 'k' to keep, 'd' to discard: {frame_file}")
    key = cv2.waitKey(0)

    # Press 'k' to keep the frame, 'd' to discard
    if key == ord('k'):
        shutil.move(frame_path, os.path.join(output_dir, frame_file))
        print(f"Kept: {frame_file}")
    elif key == ord('d'):
        os.remove(frame_path)
        print(f"Discarded: {frame_file}")
    else:
        print("Invalid input. Please press 'k' to keep or 'd' to discard.")

    # Close the frame window
    cv2.destroyAllWindows()

print(f"Frame review complete. Selected frames saved in {output_dir}")