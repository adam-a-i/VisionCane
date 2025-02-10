import osmnx as ox
import networkx as nx
import json
import requests
import os
from ultralytics import YOLO
import cv2
import RPi.GPIO as GPIO
import time
from gtts_tts import TTSEngine
from dotenv import load_dotenv
from azure.openai import AzureOpenAI
import logging
import sys


def load_map(file_path):
    """Load the pre-downloaded map from a .graphml file."""
    return ox.load_graphml(file_path)


def find_nearest_node(graph, latitude, longitude):
    """Find the nearest graph node to given latitude and longitude."""
    return ox.distance.nearest_nodes(graph, longitude, latitude)


def find_shortest_route(graph, start_coords, dest_coords):
    """Find the shortest route using Dijkstra's algorithm."""
    start_node = find_nearest_node(graph, *start_coords)
    dest_node = find_nearest_node(graph, *dest_coords)

    # Compute shortest path
    shortest_path = nx.shortest_path(graph, start_node, dest_node, weight="length")
    return shortest_path


def generate_directions(graph, path):
    """Generate step-by-step directions from the path."""
    if not path:
        return ["No route found."]

    directions = []
    for i in range(len(path) - 1):
        start_point = (graph.nodes[path[i]]['y'], graph.nodes[path[i]]['x'])
        end_point = (graph.nodes[path[i + 1]]['y'], graph.nodes[path[i + 1]]['x'])
        directions.append(f"Move from {start_point} to {end_point}")

    return directions


def main(map_file, start_coords, dest_coords):
    graph = load_map(map_file)
    path = find_shortest_route(graph, start_coords, dest_coords)
    directions = generate_directions(graph, path)

    print("\n".join(directions))


# Example usage
if __name__ == "__main__":
    map_file = "offline_map.graphml"  # Pre-downloaded map file
    start_coords = (37.7749, -122.4194)  # Example: San Francisco (latitude, longitude)
    dest_coords = (37.8044, -122.2711)  # Example: Oakland (latitude, longitude)

    main(map_file, start_coords, dest_coords)


class SmartCaneSystem:
    def __init__(self):
        """Initialize the smart cane system components."""
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        try:
            # Load environment variables
            load_dotenv()
            
            # Verify environment variables
            required_vars = ["AZURE_OPENAI_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_VERSION"]
            for var in required_vars:
                if not os.getenv(var):
                    raise ValueError(f"Missing required environment variable: {var}")
            
            # Initialize YOLO model
            self.model = YOLO("yolov8n.pt")
            
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

    def get_distance(self):
        """Get distance measurement from ultrasonic sensor."""
        GPIO.output(self.TRIG, False)
        time.sleep(0.1)
        GPIO.output(self.TRIG, True)
        time.sleep(0.00001)
        GPIO.output(self.TRIG, False)

        while GPIO.input(self.ECHO) == 0:
            pulse_start = time.time()
        while GPIO.input(self.ECHO) == 1:
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150
        return round(distance, 2)

    def process_frame(self):
        """Process camera frame and return structured detection data."""
        ret, frame = self.cap.read()
        if not ret:
            return None

        results = self.model(frame)
        structured_data = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0])
                object_name = self.model.names[cls]
                
                # Calculate position
                width = frame.shape[1]
                position = "left" if x2 < width // 3 else "right" if x1 > 2 * (width // 3) else "center"
                
                # Calculate approximate distance based on box size
                box_height = y2 - y1
                frame_height = frame.shape[0]
                size_ratio = box_height / frame_height
                approx_distance = round(1 / size_ratio * 2, 1)  # Rough estimation
                
                structured_data.append({
                    "object": object_name,
                    "confidence": round(conf, 2),
                    "position": position,
                    "estimated_distance": approx_distance
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
                model="gpt-4",  # or your specific deployment name
                messages=messages,
                max_tokens=50,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
                
        except Exception as e:
            return f"Error getting description: {str(e)}"

    def run(self):
        """Main loop for the smart cane system."""
        self.logger.info("Starting Smart Cane System...")
        try:
            last_process_time = 0
            process_interval = 1.0  # Process every 1 second
            
            while True:
                current_time = time.time()
                
                if current_time - last_process_time >= process_interval:
                    # Get sensor data
                    try:
                        distance = self.get_distance()
                        detections, frame = self.process_frame()
                        
                        if detections is not None:
                            # Get LLM description
                            description = self.get_llm_description(distance, detections)
                            
                            # Log the full context for debugging
                            self.logger.debug(f"Distance: {distance}cm")
                            self.logger.debug(f"Detections: {json.dumps(detections, indent=2)}")
                            self.logger.debug(f"Description: {description}")
                            
                            # Speak the description
                            if description and not self.tts_engine.is_speaking:
                                if not description.startswith("Error"):
                                    self.logger.info(f"Speaking: {description}")
                                    self.tts_engine.speak(description)
                                else:
                                    self.logger.error(description)
                        
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
                
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"Runtime error: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()

def main():
    try:
        smart_cane = SmartCaneSystem()
        smart_cane.run()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
