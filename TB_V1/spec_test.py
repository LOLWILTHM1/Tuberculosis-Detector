import numpy as np
from utils import make_mel_spectrogram
import sounddevice as sd

sr = 16000

print("Recording...")
audio = sd.rec(int(3 * sr), samplerate=sr, channels=1)
sd.wait()
audio = audio.flatten()
print("Done recording")

print("Making spectrogram...")
spec = make_mel_spectrogram(audio)
print("Spectrogram shape:", spec.shape)
print("Spectrogram min/max:", spec.min(), spec.max())
