import io
import threading
import time
from gtts import gTTS
from pydub import AudioSegment
import sounddevice as sd
import numpy as np
from queue import Queue

class TTSEngine:
    def __init__(self):
        """Initialize the TTS engine with threading and queue support."""
        self.is_speaking = False
        self.speech_queue = Queue()  # Queue for managing speech tasks
        self._stop_current = False
        self._speech_thread = None
        self._lock = threading.Lock()  # Lock for thread-safe operations

    def queue_speech(self, text, prepend=False, interrupt=False):
        """
        Queue speech text for playback. Returns immediately.

        Args:
            text (str): The text to be spoken.
            prepend (bool): If True, add the text to the front of the queue.
            interrupt (bool): If True, stop current speech and prepend the text.
        """
        with self._lock:
            if interrupt:
                self._stop_current = True
                # if self._speech_thread and self._speech_thread.is_alive():
                #     self._speech_thread.join()  # Wait for the current thread to stop
                self.speech_queue.queue.clear()  # Clear the queue
                self.speech_queue.put(text)  # Prepend the new text
            elif prepend:
                # Create a temporary queue to prepend the new text
                temp_queue = Queue()
                temp_queue.put(text)
                while not self.speech_queue.empty():
                    temp_queue.put(self.speech_queue.get())
                self.speech_queue = temp_queue
            else:
                self.speech_queue.put(text)  # Add to the end of the queue

            # Start the speech thread if it's not already running
            if not self._speech_thread or not self._speech_thread.is_alive():
                self._speech_thread = threading.Thread(target=self._process_queue)
                self._speech_thread.daemon = True
                self._speech_thread.start()

    def cancel_all(self):
        """Cancel all queued speech and stop current speech."""
        with self._lock:
            self._stop_current = True
            self.speech_queue.queue.clear()  # Clear the queue
            # if self._speech_thread and self._speech_thread.is_alive():
            #     self._speech_thread.join()  # Wait for the current thread to stop

    def _process_queue(self):
        """Internal method to process the speech queue."""
        while not self.speech_queue.empty():
            if self._stop_current:
                self._stop_current = False
                break

            text = self.speech_queue.get()
            self._speak(text)

    def _speak(self, text, lang='en'):
        """
        Internal method to handle speech synthesis and playback.

        Args:
            text (str): The text to be spoken.
            lang (str): Language code (default: 'en').
        """
        try:
            self.is_speaking = True

            # Convert text to speech
            tts = gTTS(text=text, lang=lang, slow=False)

            # Save to in-memory buffer
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)

            # Convert to audio segment
            audio = AudioSegment.from_file(audio_buffer, format="mp3")

            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0

            # Handle stereo
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))

            # Play audio
            if not self._stop_current:
                sd.play(samples, samplerate=audio.frame_rate)
                sd.wait()

        except Exception as e:
            print(f"TTS Error: {str(e)}")
        finally:
            self.is_speaking = False
            self._stop_current = False

    def stop(self):
        """Stop current speech if any."""
        with self._lock:
            self._stop_current = True
            # if self._speech_thread and self._speech_thread.is_alive():
            #     self._speech_thread.join()


def test_tts():
    """Test function for the TTS engine."""
    engine = TTSEngine()

    # Test basic speech
    print("Testing basic speech...")
    engine.queue_speech("Hello, this is a test of the text to speech system.")
    time.sleep(3)

    # Test interruption
    print("Testing speech interruption...")
    engine.queue_speech("This is a long sentence that will be interrupted by another one.")
    time.sleep(1)
    engine.queue_speech("This is an interrupting sentence.", interrupt=True)
    time.sleep(3)

    # Test prepend
    print("Testing prepend...")
    engine.queue_speech("This is the first sentence.")
    engine.queue_speech("This is the second sentence.")
    engine.queue_speech("This is the prepended sentence.", prepend=True)
    time.sleep(5)

    # Test cancel all
    print("Testing cancel all...")
    engine.queue_speech("This sentence will be canceled.")
    engine.queue_speech("This sentence will also be canceled.")
    time.sleep(1)
    engine.cancel_all()
    time.sleep(2)

    print("Test complete!")


if __name__ == "__main__":
    test_tts()