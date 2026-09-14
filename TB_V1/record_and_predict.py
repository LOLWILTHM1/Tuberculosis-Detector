import torch
from torchvision import models
import sounddevice as sd
from utils import make_mel_spectrogram

SR = 16000
DURATION = 7

def record():
    print("Recording...")
    audio = sd.rec(int(SR * DURATION), samplerate=SR, channels=1)
    sd.wait()
    print("Done.")
    return audio.flatten()

# Load model
print("Loading model...")
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/tb_detector.pth", map_location="cpu"))
model.eval()

# Record audio
audio = record()

print("Making spectrogram...")
spec = make_mel_spectrogram(audio)

print("Loading model...")
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

try:
    state = torch.load("models/tb_detector.pth", map_location="cpu")
    print("State keys:", len(state.keys()))
    model.load_state_dict(state)
    print("Model loaded successfully.")
except Exception as e:
    print("ERROR loading model:", e)


print("Spec dtype:", spec.dtype)
print("Spec shape:", spec.shape)
print("Running forward...")

try:
    output = model(spec)
    print("Raw output:", output)
    prob = torch.softmax(output, dim=1)[0][1].item()
    print("TB probability:", prob)
except Exception as e:
    print("ERROR in prediction:", e)

spec = make_mel_spectrogram(audio)
print("Spec shape:", spec.shape)

# Convert grayscale → RGB 3 channel
spec = spec.repeat(1, 3, 1, 1)
print("Fixed shape:", spec.shape)

with torch.no_grad():
    output = model(spec)
    print("Raw output:", output)
    prob = torch.softmax(output, dim=1)[0][1].item()
    print("Probability:", prob)

print("\n========================")
print("TB Probability:", round(prob, 4))
print("========================\n")

if prob > 0.7:
    print("⚠️ Warning: Pattern similar to TB cough")
else:
    print("✔️ Sound does not match TB cough pattern")

input("\nPress ENTER to exit...")
