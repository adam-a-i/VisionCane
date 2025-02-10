# Import required libraries
from ultralytics import YOLO  # YOLOv8 for object detection
import cv2  # OpenCV for video processing
import json  # JSON library to structure the detected data

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # Using Nano version for speed

# Open the default webcam (0 = built-in camera, change to 1 or 2 for external cameras)
cap = cv2.VideoCapture(0)

# Check if the webcam opens successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Start real-time processing
while cap.isOpened():
    # Capture frame from webcam
    ret, frame = cap.read()

    # If the frame is not read correctly, exit the loop
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Run YOLOv8 object detection on the current frame
    results = model(frame)

    # Create an empty list to store detected object details
    structured_data = []

    # Loop through detected objects
    for result in results:
        for box in result.boxes:
            # Extract bounding box coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Extract confidence score
            conf = box.conf[0].item()

            # Extract class ID (each object type has an ID)
            cls = int(box.cls[0])

            # Extract object name using YOLO's built-in class names
            object_name = model.names[cls]

            # Calculate the object's position (left/middle/right)
            width = frame.shape[1]  # Get frame width
            position = "Left" if x2 < width // 3 else "Right" if x1 > 2 * (width // 3) else "Center"

            # Structure detected object details into a dictionary
            detected_object = {
                "object": object_name,
                "confidence": round(conf, 2),  # Round confidence score
                "position": position
            }

            # Append structured object data to list
            structured_data.append(detected_object)

            # Draw bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

            # Display object name, confidence, and position on the frame
            cv2.putText(frame, f"{object_name} ({position})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert structured data into JSON format for GPT-4 processing
    structured_json = json.dumps(structured_data, indent=4)

    # Print structured data (for debugging before sending to GPT-4)
    print(structured_json)

    # Show the processed frame with detected objects
    cv2.imshow("YOLOv8 Detection", frame)

    # Press 'q' to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close all windows
cap.release()
cv2.destroyAllWindows()
