import threading
import time
from collections import deque

import numpy as np
import sounddevice
import sounddevice as sd
import whisper
from sympy.stats.sampling.sample_numpy import numpy


class VoiceCommandListener:
    def __init__(self, model="base", sample_rate=16000, block_size=1024, silence_threshold=0.2,
                 min_silence_duration=1.5, check_interval=0.5, window_duration=2.0):
        self.model = whisper.load_model(model)
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.silence_threshold = silence_threshold
        self.min_silence_duration = min_silence_duration
        self.check_interval = check_interval
        self.window_duration = window_duration
        self.trigger_terms = []
        self.is_running_flag = threading.Event()
        self.should_stop_flag = threading.Event()
        self.command_listener = None

    def is_silent(self, audio_data):
        """Check if the audio data is silent based on energy threshold."""
        return np.sqrt(np.mean(audio_data ** 2)) < self.silence_threshold

    def transcribe_audio(self, audio_data):
        """Transcribe audio data using Whisper."""
        result = self.model.transcribe(audio_data, fp16=False, language='en')
        return result["text"].strip().lower()

    def process_command(self, command):
        """Process the transcribed command."""
        print(f"Command: {command}")
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
            self.should_stop_flag.clear()

            trigger_buffer = deque(maxlen=int(self.sample_rate * 3 / self.block_size))
            command_buffer = deque(maxlen=int(self.sample_rate * 60000))
            last_check_time = -self.check_interval
            last_speech_time = 0
            detected_term = None
            stream: sounddevice.InputStream

            def audio_callback(indata, frames, t, status):
                frame_start_time = t.inputBufferAdcTime
                if self.should_stop_flag.is_set():
                    stream.close()
                    self.is_running_flag.clear()
                    return

                nonlocal detected_term
                nonlocal last_check_time
                nonlocal last_speech_time

                if last_speech_time < 0:
                    print("Started listening ...")
                if detected_term is None:
                    trigger_buffer.append(indata.copy())
                    # mean = indata.mean()
                    # if True:
                    #     print(f"Mean: {mean}")
                    if frame_start_time - last_check_time > self.check_interval:
                        audio_data = np.concatenate(trigger_buffer).flatten().astype(np.float32)
                        transcription = self.transcribe_audio(audio_data)
                        print(f"Heard: {transcription}, mean: {np.mean(audio_data ** 2)}")
                        current_term: str
                        if any((current_term := term, includes and term in transcription)[1] or (
                                not includes and term == transcription) for term, includes in
                               self.trigger_terms):
                            print(f"Detected term: {current_term}")
                            command_buffer.clear()
                            command_buffer.extend(trigger_buffer)
                            detected_term = current_term
                        else:
                            last_check_time = frame_start_time
                else:
                    command_buffer.append(indata.copy())
                    if self.is_silent(indata):
                        if frame_start_time - last_speech_time > self.min_silence_duration:
                            audio_data = np.concatenate(command_buffer).flatten().astype(np.float32)
                            transcription: str = self.transcribe_audio(audio_data)
                            self.process_command(transcription[(transcription.index(detected_term) + 1):])
                            detected_term = None
                            trigger_buffer.clear()
                        else:
                            last_speech_time = frame_start_time

            stream = sd.InputStream(samplerate=self.sample_rate, blocksize=self.block_size, channels=1,
                                    callback=audio_callback)
            stream.start()
            return True
        else:
            print("Audio thread is already running.")
            return False

    def stop(self):
        """Stop the audio listener thread."""
        if self.is_running_flag.is_set():
            self.should_stop_flag.set()
            print("Audio thread stopped.")
        else:
            print("Audio thread is not running.")


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
    time.sleep(30000)
    # Stop the listener after it halts (if necessary)
    listener.stop()
