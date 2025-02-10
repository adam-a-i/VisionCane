import json
import os
import sys
import queue
import threading
import time
import sounddevice as sd
from vosk import Model, KaldiRecognizer

class VoiceCommandInput:
    def __init__(self, model_path=None, lang = None, samplerate=16000):
        """
        Initialize the VoiceCommandInput class.
        :param model_path: Path to the Vosk model directory.
        :param samplerate: Audio sample rate (default is 16000 Hz).
        """
        if not lang and not os.path.exists(model_path):
            print(f"Please download a model from https://alphacephei.com/vosk/models and unpack as {model_path}")
            sys.exit(1)

        self.model = Model(model_path) if not lang else Model(lang=lang)
        self.recognizer = KaldiRecognizer(self.model, samplerate)
        self.samplerate = samplerate
        self.blocksize = 8000
        self.device = None  # Use the default input device
        self.q = queue.Queue()
        self.running = False
        self.trigger_mode = True
        self.triggers = set()
        self.min_silent_period = 2  # Default minimum silence period in seconds
        self.last_audio_time = time.time()
        self.current_command = ""
        self.on_command_handler = None
        self.on_trigger_handler = None
        self.audio_stream = None
        self.current_trigger = None

    def start(self):
        """Start listening for voice commands."""
        if self.running:
            print("Already running.")
            return

        self.running = True
        self.trigger_mode = True
        self.current_command = ""
        self.last_audio_time = time.time()

        def callback(indata, frames, time, status):
            """Callback function for audio input."""
            if status:
                print(status, file=sys.stderr)
            self.q.put(bytes(indata))

        self.audio_stream = sd.RawInputStream(
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            device=self.device,
            dtype='int16',
            channels=1,
            callback=callback
        )
        self.audio_stream.start()

        threading.Thread(target=self._process_audio, daemon=True).start()
        print("Voice command input started. Listening for triggers...")

    def stop(self):
        """Stop listening for voice commands."""
        if not self.running:
            print("Not running.")
            return

        self.running = False
        self.audio_stream.stop()
        self.audio_stream.close()
        print("Voice command input stopped.")

    def set_triggers(self, triggers):
        """
        Set the trigger keywords.
        :param triggers: List of trigger keywords.
        """
        self.triggers = set(triggers)
        print(f"Triggers set to: {self.triggers}")

    def set_on_command(self, handler):
        """
        Set the command handler.
        :param handler: Function to handle the command (takes a string as input).
        """
        self.on_command_handler = handler
        print("Command handler set.")

    def set_on_trigger(self, handler):
        """
        Set the trigger handler.
        :param handler: Function to handle the trigger (takes a string as input).
        """
        self.on_trigger_handler = handler
        print("Trigger handler set.")

    def set_min_silent_period(self, seconds):
        """
        Set the minimum silent period to end a command.
        :param seconds: Minimum silence duration in seconds.
        """
        self.min_silent_period = seconds
        print(f"Minimum silent period set to {seconds} seconds.")

    def _process_audio(self):
        while self.running:
            try:
                data = self.q.get(timeout=0.1)
                if self.recognizer.AcceptWaveform(data):
                    result_json = self.recognizer.Result()
                    result = json.loads(result_json).get("text", "").strip()
                    self._handle_result(result)
                else:
                    partial_json = self.recognizer.PartialResult()
                    partial_result = json.loads(partial_json).get("partial", "").strip()
                    self._handle_partial(partial_result)

                if time.time() - self.last_audio_time > self.min_silent_period and not self.trigger_mode:
                    self._handle_silence()
            except queue.Empty:
                continue

    def _handle_result(self, result):
        """Handle a full transcription result."""
        if self.trigger_mode:
            self._check_for_trigger(result, True)
        else:
            self._append_to_command(result)

    def _handle_partial(self, partial_result):
        """Handle a partial transcription result."""
        if self.trigger_mode:
            self._check_for_trigger(partial_result)
        elif partial_result != '':
            self.last_audio_time = time.time()

    def _check_for_trigger(self, text, full_result = False):
        """Check if the text contains a trigger keyword."""
        for trigger in self.triggers:
            if trigger.lower() in text.lower():
                print(f"Trigger detected: {trigger}")
                self.trigger_mode = False
                self.current_command = text if full_result else ""
                self.current_trigger = trigger
                self.last_audio_time = time.time()
                if self.on_trigger_handler:
                    self.on_trigger_handler(trigger)
                break

    def _append_to_command(self, text):
        """Append text to the current command."""
        self.current_command += " " + text.strip()
        self.last_audio_time = time.time()

    def _handle_silence(self):
        """Handle silence after a command."""
        if self.current_command.strip() and self.current_trigger in self.current_command:
            self.current_command = self.current_command.split(self.current_trigger, 1)[-1].strip()
            print(f"Command detected: {self.current_command}")
            if self.on_command_handler:
                self.on_command_handler(self.current_command)
        self.current_command = ""
        self.trigger_mode = True
        print("Listening for triggers...")

if __name__ == "__main__":
    def on_command(command):
        print(f"Command received: {command}")


    def on_trigger(trigger):
        print(f"Trigger activated: {trigger}")


    # Initialize the voice command input
    vci = VoiceCommandInput(model_path="../resources/vosk-model")

    # Set triggers and handlers
    vci.set_triggers(["hey computer", "computer"])
    vci.set_on_command(on_command)
    vci.set_on_trigger(on_trigger)
    vci.set_min_silent_period(2)  # 2 seconds of silence ends the command

    # Start listening
    vci.start()

    # Keep the program running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        vci.stop()
