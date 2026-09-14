import sounddevice as sd
import numpy as np

sr = 16000
print("Recording...")
audio = sd.rec(int(3 * sr), samplerate=sr, channels=1)
sd.wait()
print("Done.")
print("Shape:", audio.shape)
print("First samples:", audio[:10])
