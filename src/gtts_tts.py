import io
import threading
import time
from gtts import gTTS
from pydub import AudioSegment
import sounddevice as sd
import numpy as np

class TTSEngine:
    def __init__(self):
        """Initialize the TTS engine with threading support."""
        self.is_speaking = False
        self.speech_thread = None
        self._stop_current = False

    def speak(self, text, lang='en'):
        """
        Speak the given text asynchronously.
        
        Args:
            text (str): The text to be spoken
            lang (str): Language code (default: 'en')
        """
        # If already speaking, stop current speech
        if self.is_speaking:
            self._stop_current = True
            if self.speech_thread:
                self.speech_thread.join()
        
        self._stop_current = False
        self.speech_thread = threading.Thread(target=self._speak_thread, args=(text, lang))
        self.speech_thread.daemon = True
        self.speech_thread.start()

    def _speak_thread(self, text, lang):
        """
        Internal method to handle speech synthesis and playback in a separate thread.
        
        Args:
            text (str): The text to be spoken
            lang (str): Language code
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
        if self.is_speaking:
            self._stop_current = True
            if self.speech_thread:
                self.speech_thread.join()

def test_tts():
    """Test function for the TTS engine."""
    engine = TTSEngine()
    
    # Test basic speech
    print("Testing basic speech...")
    engine.speak("Hello, this is a test of the text to speech system.")
    time.sleep(3)
    
    # Test interruption
    print("Testing speech interruption...")
    engine.speak("This is a long sentence that will be interrupted by another one.")
    time.sleep(1)
    engine.speak("This is an interrupting sentence.")
    time.sleep(3)
    
    # Test multiple languages
    print("Testing multiple languages...")
    engine.speak("Hello in English")
    time.sleep(2)
    engine.speak("Bonjour en français", lang='fr')
    time.sleep(2)
    
    print("Test complete!")

if __name__ == "__main__":
    test_tts()
