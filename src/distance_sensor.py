from gpio import GPIO
import time
import threading


class DistanceSensor:
    def __init__(self, trig_pin=23, echo_pin=24):
        # Set GPIO mode (BCM or BOARD)
        GPIO.setmode(GPIO.BCM)

        # Define pins
        self.TRIG = trig_pin
        self.ECHO = echo_pin

        # Setup GPIO
        GPIO.setup(self.TRIG, GPIO.OUT)
        GPIO.setup(self.ECHO, GPIO.IN)

        # Initialize variables
        self._distance = 100 #None
        self._stop_thread = False
        self._distance_thread = threading.Thread(target=self._update_distance)
        self._distance_thread.daemon = True  # Ensures thread exits when the main program exits
        self._distance_thread.start()

    def _update_distance(self):
        while not self._stop_thread:
            self._distance = self._get_distance()
            time.sleep(0.3)  # Update every second

    def _get_distance(self):
        GPIO.output(self.TRIG, False)
        time.sleep(0.5)  # Stabilize sensor
        GPIO.output(self.TRIG, True)
        time.sleep(0.00001)  # 10µs pulse
        GPIO.output(self.TRIG, False)

        # Measure Echo pulse duration
        while GPIO.input(self.ECHO) == 0:
            time.sleep(1) # todo fix
            pulse_start = time.time()
        while GPIO.input(self.ECHO) == 1:
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150  # Convert to cm (speed of sound)
        return round(distance, 2)

    def get_distance(self):
        return self._distance

    def stop(self):
        self._stop_thread = True
        self._distance_thread.join()
        GPIO.cleanup()


# Example usage
if __name__ == "__main__":
    sensor = DistanceSensor()
    try:
        while True:
            distance = sensor.get_distance()
            if distance is not None:
                print(f"Distance: {distance} cm")
            time.sleep(1)
    except KeyboardInterrupt:
        sensor.stop()
