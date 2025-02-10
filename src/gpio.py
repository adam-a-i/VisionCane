try:
    import RPi.GPIO as _GPIO  # Try importing the real module

    GPIO = _GPIO
    real_gpio = True
except ImportError:
    real_gpio = False


    class MockGPIO:
        BCM = "BCM"
        BOARD = "BOARD"
        IN = "IN"
        OUT = "OUT"
        HIGH = "HIGH"
        LOW = "LOW"

        def setmode(self, mode):
            return
            print(f"[MOCK] Setting mode to {mode}")

        def setup(self, pin, mode):
            return
            print(f"[MOCK] Setting up pin {pin} as {mode}")

        def output(self, pin, state):
            return
            print(f"[MOCK] Setting pin {pin} to {state}")

        def input(self, pin):
            return self.LOW
            print(f"[MOCK] Reading pin {pin}")
            return self.LOW  # Simulating low signal

        def cleanup(self):
            return
            print("[MOCK] Cleaning up GPIO")


    GPIO = MockGPIO()
