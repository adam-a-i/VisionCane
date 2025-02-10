import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import sounddevice as sd
import whisper


class VoiceCommandListener:
    def __init__(self, model="medium", sample_rate=16000, block_size=1024, silence_threshold=0.2,
                 min_silence_duration=1.5, check_interval=0.1, window_duration=2.0):
        self.model = whisper.load_model(model)
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.silence_threshold = silence_threshold
        self.min_silence_duration = min_silence_duration
        self.check_interval = check_interval
        self.window_duration = window_duration
        self.audio_queue = de
        self.trigger_terms = []
        self.is_running_flag = threading.Event()
        self.command_listener = None
        self.last_sound = datetime.now().timestamp()

    def audio_callback(self, indata, frames, time, status):
        """Audio callback for microphone input."""

        if status:
            print(status, file=sys.stderr)
        self.audio_queue.put(indata.copy())

    def is_silent(self, audio_data):
        """Check if the audio data is silent based on energy threshold."""
        return np.sqrt(np.mean(audio_data ** 2)) < self.silence_threshold

    def transcribe_audio(self, audio_data):
        """Transcribe audio data using Whisper."""
        audio_data = audio_data.flatten().astype(np.float32)
        result = self.model.transcribe(audio_data, fp16=False, language='en')
        return result["text"].strip().lower()

    def listen_for_trigger(self):
        """Listen for the trigger term using a moving window."""
        print("Listening for trigger term...")
        buffer = deque(maxlen=int(self.sample_rate * self.window_duration / self.block_size))
        timepoints = deque(maxlen=int(self.sample_rate * self.window_duration / self.block_size))
        last_check_time = time.time()

        while self.is_running_flag.is_set():
            now = time.time()
            if len(buffer):
                print(
                    f"Buffer count: {len(buffer)}, duration of buffer: {now - timepoints[0]}, rate of buffer: {len(buffer) * len(buffer[0]) / (now - timepoints[0])}")

            buffer.append(self.audio_queue.get())
            timepoints.append(now)

            if time.time() - last_check_time >= self.check_interval:
                if len(buffer) > 0:
                    audio_data = np.concatenate(buffer)
                    transcription = self.transcribe_audio(audio_data)
                    print(f"Heard: {transcription}")

                    for term, includes in self.trigger_terms:
                        if (includes and term in transcription) or (not includes and term == transcription):
                            print(f"Trigger term '{term}' detected!")
                            return True
                last_check_time = time.time()

    def listen_for_command(self):
        """Listen for a command after the trigger term is detected."""
        print("Listening for command...")
        audio_frames = []
        last_speech_time = time.time()

        while self.is_running_flag.is_set():
            if not self.audio_queue.empty():
                audio_data = self.audio_queue.get()
                audio_frames.append(audio_data)

                if self.is_silent(audio_data):
                    if time.time() - last_speech_time > self.min_silence_duration:
                        break
                else:
                    last_speech_time = time.time()

        audio_data = np.concatenate(audio_frames)
        command = self.transcribe_audio(audio_data)
        print(f"Command: {command}")
        return command

    def process_command(self, command):
        """Process the transcribed command."""
        if self.command_listener: self.command_listener(command)

    def set_command_listener(self, listener):
        """Set a custom command listener."""
        self.command_listener = listener

    def set_triggers(self, triggers):
        """Set a list of triggers with terms and inclusion criteria."""
        self.trigger_terms = triggers

    def is_running(self):
        """Check if the audio thread is running."""
        return self.is_running_flag.is_set()

    def start(self):
        """Start the audio listener thread."""
        if not self.is_running_flag.is_set():
            self.is_running_flag.set()
            self.listener_thread = threading.Thread(target=self._listen_for_commands)
            self.listener_thread.start()
            print("Audio thread started.")
            return True
        else:
            print("Audio thread is already running.")
            return False

    def stop(self):
        """Stop the audio listener thread."""
        if self.is_running_flag.is_set():
            self.is_running_flag.clear()
            self.listener_thread.join()
            print("Audio thread stopped.")
        else:
            print("Audio thread is not running.")

    def wait(self):
        """Wait for the audio thread to halt."""
        print("Waiting for audio thread to halt...")
        self.listener_thread.join()
        print("Audio thread has halted.")

    def _listen_for_commands(self):
        """Main loop to listen for trigger terms and commands."""
        try:
            with sd.InputStream(samplerate=self.sample_rate, blocksize=self.block_size, channels=1,
                                callback=self.audio_callback) as stream:
                stream.close()
        except KeyboardInterrupt:
            print("\nExiting...")


if __name__ == "__main__":
    listener = VoiceCommandListener()

    # Set up triggers (terms to be detected, and whether they should be included or not)
    listener.set_triggers([("hey computer", True)])


    # Set a custom command listener
    def custom_command_listener(command):
        print(f"Custom command listener received command: {command}")
        if command == "exit":
            print("Custom exit action triggered.")
            listener.stop()


    listener.set_command_listener(custom_command_listener)

    # Start the voice command listener
    listener.start()

    # Wait for the thread to halt
    listener.wait()

    # Stop the listener after it halts (if necessary)
    listener.stop()
