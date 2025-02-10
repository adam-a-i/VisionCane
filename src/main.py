import json
import logging
import os
import sys
import time
from gpio import GPIO
import cv2
from openai import AzureOpenAI
from dotenv import load_dotenv
from yolo_tracker import YOLOTracker  # Import the YOLOTracker module

from gtts_tts import TTSEngine  # Updated TTSEngine with queue_speech and cancel_all
from src.distance_sensor import DistanceSensor


class SmartCaneSystem:
    def __init__(self):
        """Initialize the smart cane system components."""
        # Setup logging
        # logging.basicConfig(level=logging.INFO)
        # self.logger = logging.getLogger(__name__)
        class MyLogger:
            def info(self, msg):
                print(msg)
            def debug(self,msg):
                print(msg)
            def error(self, msg):
                print(msg)

        self.logger = MyLogger()
        self.system_prompt = """You are a smart assistant for a navigation cane aiding visually impaired users. Using real-time camera input, you detect objects ahead, determine their distance, and provide clear, timely navigation instructions. Your responses must be concise and direct, continuously updating as the environment changes. Differentiate between moving and stationary objects, warn about hazards like stairs and curbs, and maintain a calm, confident tone. Prioritize immediate hazards and confirm when the path is clear."""
        self.distance_sensor = DistanceSensor()
        self.running = False  # System running state

        try:
            # Load environment variables
            load_dotenv()

            # Verify environment variables
            required_vars = ["AZURE_OPENAI_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_VERSION"]
            for var in required_vars:
                if not os.getenv(var):
                    raise ValueError(f"Missing required environment variable: {var}")

            # Initialize YOLO tracker
            self.tracker = YOLOTracker(model_path="../resources/yolov8n.pt", min_confidence=0.5)
            self.tracker.set_on_new_object(self.on_new_object)
            self.tracker.set_on_object_shift(self.on_object_shift)

            # Initialize TTS engine
            self.tts_engine = TTSEngine()

            # Setup camera
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open camera")

            # Setup GPIO
            GPIO.setmode(GPIO.BCM)
            self.TRIG = 23
            self.ECHO = 24
            GPIO.setup(self.TRIG, GPIO.OUT)
            GPIO.setup(self.ECHO, GPIO.IN)

            # Initialize Azure OpenAI client
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_version=os.getenv("AZURE_OPENAI_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )

            self.logger.info("Smart Cane System initialized successfully")

        except Exception as e:
            self.logger.error(f"Initialization error: {str(e)}")
            self.cleanup()
            sys.exit(1)

    def on_new_object(self, new_object):
        """Handler for new objects detected by YOLO."""
        message = self.get_llm_description(
            self.distance_sensor.get_distance(),
            [{
                "object": new_object["class"],
                "confidence": new_object["confidence"],
                "position": new_object["position"],
                "distance": new_object["distance"]
            }]
        )
        self.tts_engine.queue_speech(message, interrupt=True)  # Interrupt current speech

    def on_object_shift(self, shifted_object):
        """Handler for significant object movements detected by YOLO."""
        message = self.get_llm_description(
            self.distance_sensor.get_distance(),
            [{
                "object": shifted_object["class"],
                "confidence": shifted_object["confidence"],
                "position": shifted_object["position"],
                "distance": shifted_object["distance"]
            }]
        )
        self.tts_engine.queue_speech(message, interrupt=True)  # Interrupt current speech

    def process_frame(self):
        """Process camera frame and return structured detection data."""
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Process frame with YOLO tracker
        self.tracker.detect_and_track(frame)

        # Get detected objects
        detected_objects = self.tracker.get_detected_objects()
        structured_data = []

        for object_id, obj in detected_objects.items():
            x1, y1, x2, y2 = obj["box"]
            class_label = obj["class"]
            confidence = obj["confidence"]

            # Calculate position
            width = frame.shape[1]
            position = "left" if x2 < width // 3 else "right" if x1 > 2 * (width // 3) else "center"

            # Get distance from sensor
            distance = self.distance_sensor.get_distance()

            structured_data.append({
                "object": class_label,
                "confidence": round(confidence, 2),
                "position": position,
                "distance": distance
            })

        return structured_data, frame

    def get_llm_description(self, sensor_data, detections):
        """Get natural language description from Azure OpenAI."""
        try:
            message_content = f"""
            Sensor Data:
            - Ultrasonic Distance: {sensor_data} cm

            Detected Objects:
            {json.dumps(detections, indent=2)}

            Provide a clear, concise navigation instruction based on this data.
            """

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": message_content}
            ]

            response = self.client.chat.completions.create(
                model="gpt-4o",  # or your specific deployment name
                messages=messages,
                max_tokens=50,
                temperature=0.3
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Error getting description: {str(e)}"

    def start(self):
        """Start the smart cane system."""
        self.running = True
        self.logger.info("Starting Smart Cane System...")
        self.run()

    def stop(self):
        """Stop the smart cane system."""
        self.running = False
        self.tts_engine.cancel_all()  # Cancel all queued speech
        self.logger.info("Smart Cane System stopped.")

    def run(self):
        """Main loop for the smart cane system."""
        last_process_time = 0
        process_interval = 10.0  # Process every 10 seconds

        while self.running:
            current_time = time.time()

            try:
                detections, frame = self.process_frame()
            except Exception as e:
                self.logger.error(f"Processing error: {str(e)}")
                continue

            if current_time - last_process_time >= process_interval:
                # Get sensor data
                try:

                    distance = self.distance_sensor.get_distance()

                    if detections is not None:
                        # Get LLM description
                        description = self.get_llm_description(distance, detections)

                        # Log the full context for debugging
                        self.logger.debug(f"Distance: {distance}cm")
                        self.logger.debug(f"Detections: {json.dumps(detections, indent=2)}")
                        self.logger.debug(f"Description: {description}")

                        # Speak the description
                        if description and not description.startswith("Error"):
                            self.logger.info(f"Speaking: {description}")
                            self.tts_engine.queue_speech(description)

                    last_process_time = current_time

                except Exception as e:
                    self.logger.error(f"Processing error: {str(e)}")
                    continue  # Continue to next iteration instead of crashing

            # Optional: display the processed frame
            if frame is not None:
                cv2.imshow("Smart Cane View", frame)

            # Check for exit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.logger.info("Exit command received")
                break

    def cleanup(self):
        """Clean up resources."""
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()


def main():
    try:
        smart_cane = SmartCaneSystem()
        smart_cane.start()  # Start the system
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()