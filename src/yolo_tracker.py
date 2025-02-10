from ultralytics import YOLO
from collections import defaultdict
import cv2
import numpy as np


class YOLOTracker:
    def __init__(self, model_path="yolov8n.pt", min_confidence=0.5):
        # Load the YOLO model
        self.model = YOLO(model_path)
        self.min_confidence = min_confidence  # Minimum confidence threshold
        self.current_objects = {}  # Dictionary to store current objects
        self.tracked_objects_history = {}  # Dictionary to store object history
        self.on_new_object_callback = None  # Callback for new objects
        self.on_object_shift_callback = None  # Callback for significant object shifts
        self.movement_threshold = 20  # Minimum pixel movement to consider an object "shifted"

    def get_min_confidence(self):
        return self.min_confidence

    def set_min_confidence(self, min_confidence):
        self.min_confidence = min_confidence

    def get_detected_objects(self):
        """Returns the currently detected objects."""
        return self.current_objects

    def set_on_new_object(self, callback):
        """Sets a callback function to be called when a new object is detected."""
        self.on_new_object_callback = callback

    def set_on_object_shift(self, callback):
        """Sets a callback function to be called when an object shifts significantly."""
        self.on_object_shift_callback = callback

    def detect_and_track(self, frame):
        """Processes a frame to detect and track objects."""
        results = self.model.track(frame, persist=True, verbose=False)  # Disable verbose logging

        # Clear current objects
        self.current_objects.clear()

        # Extract detected objects
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
            ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None  # Object IDs
            classes = result.boxes.cls.cpu().numpy()  # Class labels
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores

            for i, box in enumerate(boxes):
                confidence = float(confidences[i])
                if confidence < self.min_confidence:
                    continue  # Skip objects with low confidence

                object_id = int(ids[i]) if ids is not None else i
                class_label = self.model.names[int(classes[i])]

                # Store object details
                self.current_objects[object_id] = {
                    "box": box,
                    "class": class_label,
                    "confidence": confidence,
                }

                # Check if the object is new or has moved significantly
                if object_id not in self.tracked_objects_history:
                    # New object
                    self.tracked_objects_history[object_id] = {
                        "box": box,
                        "class": class_label,
                        "confidence": confidence,
                    }
                    if self.on_new_object_callback:
                        self.on_new_object_callback({
                            "id": object_id,
                            "class": class_label,
                            "confidence": confidence,
                            "box": [float(x) for x in box],
                        })
                else:
                    # Existing object: check if it has moved significantly
                    prev_box = self.tracked_objects_history[object_id]["box"]
                    curr_box = box
                    # Calculate the center points of the previous and current boxes
                    prev_center = np.array([(prev_box[0] + prev_box[2]) / 2, (prev_box[1] + prev_box[3]) / 2])
                    curr_center = np.array([(curr_box[0] + curr_box[2]) / 2, (curr_box[1] + curr_box[3]) / 2])
                    # Calculate Euclidean distance between centers
                    movement_distance = np.linalg.norm(curr_center - prev_center)

                    if movement_distance > self.movement_threshold:
                        # Object has moved significantly
                        self.tracked_objects_history[object_id]["box"] = box
                        if self.on_object_shift_callback:
                            self.on_object_shift_callback({
                                "id": object_id,
                                "class": class_label,
                                "confidence": confidence,
                                "box": [float(x) for x in box],
                                "movement_distance": float(movement_distance),
                            })

        # Draw bounding boxes and labels for all tracked objects
        for object_id, obj in self.tracked_objects_history.items():
            if object_id in self.current_objects:
                x1, y1, x2, y2 = map(int, obj["box"])  # Convert box coordinates to integers
                color = (0, 255, 0)  # Green for current objects
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Draw bounding box
                label = f"{obj['class']} {obj['confidence']:.2f} (ID: {object_id})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)  # Draw label

    def reset_tracking(self):
        """Resets the tracked objects dictionary."""
        self.tracked_objects_history.clear()


if __name__ == '__main__':
    # Initialize the tracker
    tracker = YOLOTracker(model_path="yolov8n.pt", min_confidence=0.5)


    # Set a callback for new objects
    def on_new_object(new_object):
        print(f"New object detected: {new_object}")


    tracker.set_on_new_object(on_new_object)


    # Set a callback for significant object shifts
    def on_object_shift(shifted_object):
        print(f"Object shifted: {shifted_object}")


    tracker.set_on_object_shift(on_object_shift)

    # Open a video stream (e.g., webcam or video file)
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or a file path for a video

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect and track objects in the frame
        tracker.detect_and_track(frame)

        # Display the annotated frame
        cv2.imshow("YOLO Tracking - Less Sensitive", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
