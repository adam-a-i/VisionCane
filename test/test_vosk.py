#!python
import os
import queue
import sys

import numpy  # Make sure NumPy is loaded before it is used in the callback
import sounddevice as sd

assert numpy  # avoid "imported but unused" message (W0611)
from vosk import Model, KaldiRecognizer

# Set up the model and recognizer
# model_path = "../resources/vosk-model"  # Replace with the path to your Vosk model
# if not os.path.exists(model_path):
#     print(f"Please download a model from https://alphacephei.com/vosk/models and unpack as {model_path}")
#     sys.exit(1)
# model = Model(model_path=model_path)

model = Model(lang='en-us')
recognizer = KaldiRecognizer(model, 16000)

# Audio settings
samplerate = 16000
blocksize = 8000
device = None  # Use the default input device

q = queue.Queue()

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

try:
    with sd.RawInputStream(samplerate=samplerate, blocksize=blocksize, device=device,
                           dtype='int16', channels=1, callback=callback):
        print('#' * 80)
        print('Press Ctrl+C to stop the recording')
        print('#' * 80)

        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                print(result)
            else:
                partial_result = recognizer.PartialResult()
                print(partial_result)

except KeyboardInterrupt:
    print('\nDone')
except Exception as e:
    print(type(e).__name__ + ': ' + str(e))