import cv2
import torch
import supervision as sv

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

video_path = 'video/demo_serve3.mp4'
cap = cv2.VideoCapture(video_path)

# Create a BoxAnnotator from Supervision
box_annotator = sv.BoxAnnotator()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Use YOLOv5 to detect objects in the current frame
    results = model(frame)

    # Convert YOLOv5 results to Supervision detections format
    detections = sv.Detections.from_yolov5(results)

    # Filter only 'sports ball' (class 32 in COCO dataset)
    pickleball_detections = detections[detections.class_id == 32]

    # Annotate the frame with detection bounding boxes
    annotated_frame = box_annotator.annotate(scene=frame, detections=pickleball_detections)

    # Add the label text manually
    for i, detection in enumerate(pickleball_detections.xyxy):
        x1, y1, x2, y2 = map(int, detection)  # Extract and convert bounding box coordinates to integers
        cv2.putText(annotated_frame, "BALL", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the annotated frame
    cv2.imshow("Pickleball Detection with Supervision", annotated_frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()