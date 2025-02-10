import json
import logging
import os
import sys
import threading
import time
from gpio import GPIO
import cv2
from openai import AzureOpenAI
from dotenv import load_dotenv
from yolo_tracker import YOLOTracker  # Import the YOLOTracker module
from vosk_voice_command import VoiceCommandInput  # Import the VoiceCommandInput module

from gtts_tts import TTSEngine  # Updated TTSEngine with queue_speech and cancel_all
from src.distance_sensor import DistanceSensor


class SmartCaneSystem:
    def __init__(self):
        """Initialize the smart cane system components."""
        # Setup logging
        class MyLogger:
            def info(self, msg):
                print(msg)
            def debug(self, msg):
                print(msg)
            def error(self, msg):
                print(msg)

        self.logger = MyLogger()
        self.system_prompt = """You are a smart assistant for a navigation cane aiding visually impaired users. Using real-time camera input, you detect objects ahead, determine their distance, and provide clear, timely navigation instructions. Your responses must be concise and direct, continuously updating as the environment changes. Differentiate between moving and stationary objects, warn about hazards like stairs and curbs, and maintain a calm, confident tone. Prioritize immediate hazards and confirm when the path is clear."""
        self.distance_sensor = DistanceSensor()
        self.running = False  # System running state
        self.voice_listening = True  # Voice command listening state
        self.gpio_button_pin = 17  # GPIO pin for the push button

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
            GPIO.setup(self.gpio_button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.add_event_detect(self.gpio_button_pin, GPIO.FALLING, callback=self.toggle_system, bouncetime=300)

            # Initialize Azure OpenAI client
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_version=os.getenv("AZURE_OPENAI_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )

            # Initialize voice command input
            self.vci = VoiceCommandInput(model_path="../resources/vosk-model")
            self.vci.set_triggers(["hey computer", "computer"])
            self.vci.set_on_command(self.handle_command)
            self.vci.set_on_trigger(self.handle_trigger)
            self.vci.set_min_silent_period(2)  # 2 seconds of silence ends the command
            self.vci.start()

            self.logger.info("Smart Cane System initialized successfully")

        except Exception as e:
            self.logger.error(f"Initialization error: {str(e)}")
            self.cleanup()
            sys.exit(1)

    def on_new_object(self, new_object):
        """Handler for new objects detected by YOLO."""
        if self.running and new_object and "class" in new_object:  # Only notify if the system is running # TODO: better fix
            print(new_object)
            message = self.get_llm_description(
                self.distance_sensor.get_distance(),
                [new_object]
            )
            self.tts_engine.queue_speech(message, interrupt=True)  # Interrupt current speech

    def on_object_shift(self, shifted_object):
        """Handler for significant object movements detected by YOLO."""
        if self.running:  # Only notify if the system is running
            message = self.get_llm_description(
                self.distance_sensor.get_distance(),
                [shifted_object]
            )
            self.tts_engine.queue_speech(message, interrupt=True)  # Interrupt current speech

    def handle_command(self, command):
        """Handle voice commands using OpenAI API to identify the correct function."""
        self.logger.info(f"Command received: {command}")
        intent = self.get_command_intent(command)
        if intent:
            self.execute_command(intent)
        else:
            self.alert_user(f"Sorry, I cannot help with: {command}")

    def get_command_intent(self, command):
        """Use OpenAI API to identify the intent of the command."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Use GPT-4 model
                messages=[
                    {"role": "system", "content": "Identify the intent of the user's command. Possible intents: 'tell_time', 'halt', 'run'."},
                    {"role": "user", "content": command}
                ],
                max_tokens=10,
                temperature=0.0  # Ensure deterministic output
            )
            intent = response.choices[0].message.content.strip().lower()
            return intent
        except Exception as e:
            self.logger.error(f"Error identifying command intent: {str(e)}")
            return None

    def execute_command(self, intent):
        """Execute the function corresponding to the identified intent."""
        if intent == "tell_time":
            self.tell_time()
        elif intent == "halt":
            self.halt()
        elif intent == "run":
            self.run()
        else:
            self.alert_user(f"Unknown intent: {intent}")

    def handle_trigger(self, trigger):
        """Handle trigger phrases."""
        self.logger.info(f"Trigger activated: {trigger}")
        self.alert_user("How can I assist you?")

    def toggle_system(self, channel=None):
        """Toggle the system on/off using GPIO button press."""
        self.running = not self.running
        if self.running:
            self.run()
        else:
            self.halt()

    def run(self):
        """Start the smart cane system."""
        if not self.running:
            self.running = True
            self.alert_user("Smart Cane is now running.")
            # Start processing frames and notifying about the environment
            self.start_environment_processing()

    def halt(self):
        """Stop the smart cane system."""
        if self.running:
            self.running = False
            self.alert_user("Smart Cane is now halted.")
            # Stop processing frames and notifying about the environment
            self.stop_environment_processing()

    def start_environment_processing(self):
        """Start processing frames and notifying about the environment."""
        self.logger.info("Starting environment processing...")
        self.environment_thread = threading.Thread(target=self.process_environment, daemon=True)
        self.environment_thread.start()

    def stop_environment_processing(self):
        """Stop processing frames and notifying about the environment."""
        self.logger.info("Stopping environment processing...")
        if hasattr(self, "environment_thread"):
            self.environment_thread.join()

    def process_environment(self):
        """Continuously process frames and notify about the environment."""
        last_process_time = 0
        process_interval = 10.0  # Process every 10 seconds

        while self.running:
            current_time = time.time()

            detections, frame = self.process_frame()
            if current_time - last_process_time >= process_interval:
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

    def process_frame(self):
        """Process camera frame and return structured detection data."""
        ret, frame = self.cap.read()
        if not ret:
            return None, None

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

    def tell_time(self):
        """Tell the current time."""
        from datetime import datetime
        current_time = datetime.now().strftime("%I:%M %p")
        self.alert_user(f"The current time is {current_time}.")

    def alert_user(self, message):
        """Alert the user with a message."""
        self.logger.info(message)
        self.tts_engine.queue_speech(message)

    def start(self):
        """Start the smart cane system."""
        self.running = True
        self.voice_listening = True
        self.alert_user("Smart Cane is now running.")
        self.start_environment_processing()

    def stop(self):
        """Stop the smart cane system."""
        self.running = False
        self.voice_listening = False
        self.alert_user("Smart Cane is now stopped.")
        self.stop_environment_processing()

    def cleanup(self):
        """Clean up resources."""
        self.vci.stop()
        self.cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()


if __name__ == "__main__":
    smart_cane = SmartCaneSystem()
    smart_cane.start()  # Start the system

    try:
        while True:
            time.sleep(1)  # Keep the program running
    except KeyboardInterrupt:
        smart_cane.stop()
        smart_cane.cleanup()