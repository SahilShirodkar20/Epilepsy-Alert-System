import cv2
import torch
from ultralytics import YOLO
import time
import winsound

# Load the YOLOv8 model
model = YOLO("best.pt")

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set the width and height of the video frames
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Set the minimum confidence level for detecting the 'Fall Detected' class
MIN_CONFIDENCE = 0.5

# Initialize the alert variables
alert_triggered = False
alert_start_time = 0

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Perform object detection on the frame
    results = model(frame)

    # Reset the alert trigger if no detections are made in the current frame
    if len(results[0].boxes) == 0:
        alert_triggered = False
        alert_start_time = 0

    # Loop through the detected objects
    for box in results[0].boxes:
        # Get the confidence score, coordinates, and class of the object
        conf = box.conf[0].item()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = results[0].names[box.cls[0].item()]

        # Check if the confidence score is above the minimum confidence level and the class is 'Fall Detected'
        if conf > MIN_CONFIDENCE and label == 'Fall Detected':
            # Check if the alert has been triggered already
            if not alert_triggered:
                # If the alert has not been triggered, set the alert start time
                alert_start_time = time.time()
                alert_triggered = True
            # Check if the alert has been triggered for 5 consecutive seconds
            elif time.time() - alert_start_time >= 5:
                # If the alert has been triggered for 5 consecutive seconds, raise an alert
                winsound.Beep(1000, 1000)  # Play a sound

        # If the class is not 'Fall Detected', continue to the next iteration of the loop
        else:
            continue

        # Draw a bounding box around the object
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Print the object's class and coordinates on the frame
        cv2.putText(frame, f"{label}: {x1}, {y1}, {x2}, {y2}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Object Detection", frame)

    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()