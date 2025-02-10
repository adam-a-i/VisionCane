import whisper
import sounddevice as sd
import numpy as np
import time
import torch

# Load the Whisper model (choose 'tiny', 'base', 'small', 'medium', 'large')
model = whisper.load_model("small")

input()
