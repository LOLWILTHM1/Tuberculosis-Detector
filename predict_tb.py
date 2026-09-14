"""
RECORD COUGH AND PREDICT TB
============================
Record a cough sound from your microphone and predict TB probability.

Requirements:
    pip install sounddevice scipy

Usage:
    python record_and_predict.py
"""

import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import librosa
import os


# ============================================
# MODEL DEFINITION
# ============================================

class TBDetectorResNet18(nn.Module):
    def __init__(self, pretrained=False, dropout=0.5):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        return self.backbone(x)


# ============================================
# SPECTROGRAM GENERATION
# ============================================

class SpectrogramGenerator:
    def __init__(self, sr=16000, n_mels=128, n_fft=2048, hop_length=512):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def audio_to_spectrogram(self, audio):
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        return mel_spec_norm
    
    def process_audio_array(self, audio, duration=7):
        target_length = self.sr * duration
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        
        spec = self.audio_to_spectrogram(audio)
        spec_tensor = torch.FloatTensor(spec).unsqueeze(0).repeat(3, 1, 1)
        return spec_tensor


# ============================================
# AUDIO RECORDING
# ============================================

def record_audio(duration=7, sr=16000):
    """
    Record audio from microphone
    
    Args:
        duration: Recording duration in seconds
        sr: Sample rate
    
    Returns:
        audio: numpy array of audio samples
    """
    print(f"\n{'='*60}")
    print("🎤 RECORDING AUDIO")
    print(f"{'='*60}")
    print(f"Duration: {duration} seconds")
    print(f"Sample rate: {sr} Hz")
    print(f"\n🔴 Recording will start in 3 seconds...")
    print("   Get ready to cough!")
    
    import time
    for i in range(3, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    
    print("\n🔴 RECORDING NOW - Please cough!")
    
    # Record audio
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    
    print("✅ Recording complete!")
    
    # Convert to mono
    audio = audio.flatten()
    
    # Save to file
    output_file = "Recordings/recorded_cough.wav"
    wav.write(output_file, sr, (audio * 32767).astype(np.int16))
    print(f"💾 Saved to: {output_file}\n")
    
    return audio, output_file


# ============================================
# PREDICTION
# ============================================

def predict_tb_from_audio_array(audio, sr=16000, model_path='models/tb_detector.pth', device='cpu'):
    """
    Predict TB from audio array
    """
    
    print(f"{'='*60}")
    print("🔍 ANALYZING COUGH...")
    print(f"{'='*60}\n")
    
    # Check model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print(f"   Please train the model first!")
        return None, None, None
    
    # Load model
    print("⏳ Loading AI model...")
    model = TBDetectorResNet18(pretrained=False, dropout=0.5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("✓ Model loaded\n")
    
    # Generate spectrogram
    print("⏳ Converting to spectrogram...")
    spec_gen = SpectrogramGenerator(sr=sr, n_mels=128)
    
    # Resample if needed
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    spec_tensor = spec_gen.process_audio_array(audio, duration=7)
    
    # Resize to match training
    resize_transform = transforms.Resize((128, 256))
    spec_tensor = resize_transform(spec_tensor)
    print("✓ Spectrogram generated\n")
    
    # Predict
    print("⏳ Running AI analysis...")
    with torch.no_grad():
        spec_batch = spec_tensor.unsqueeze(0).to(device)
        output = model(spec_batch)
        prob = F.softmax(output, dim=1)
    
    tb_prob = prob[0][1].item() * 100
    confidence = torch.max(prob).item() * 100
    prediction = "TB" if tb_prob > 50 else "Normal"
    
    print("✓ Analysis complete\n")
    
    return tb_prob, confidence, prediction


def print_results(tb_prob, confidence, prediction):
    """Print formatted results"""
    
    print(f"{'='*60}")
    print(f"📊 PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"Prediction: {prediction}")
    print(f"TB Probability: {tb_prob:.2f}%")
    print(f"Confidence: {confidence:.2f}%")
    print(f"")
    
    if tb_prob > 70:
        print("⚠️  HIGH RISK")
        print("   • Pattern shows strong similarity to TB cough")
        print("   • STRONGLY RECOMMEND medical consultation")
        print("   • Get proper TB testing (sputum, X-ray)")
    elif tb_prob > 50:
        print("⚡ MODERATE-HIGH RISK")
        print("   • Some TB-like patterns detected")
        print("   • Recommend medical consultation")
    elif tb_prob > 30:
        print("⚡ MODERATE RISK")
        print("   • Mild TB-like patterns")
        print("   • Monitor symptoms")
    else:
        print("✅ LOW RISK")
        print("   • Pattern not consistent with TB")
        print("   • Continue monitoring if symptoms persist")
    
    print(f"")
    print(f"⚕️  DISCLAIMER: This is NOT a medical diagnosis.")
    print(f"   Always consult healthcare professionals.")
    print(f"{'='*60}\n")


# ============================================
# MAIN FUNCTION
# ============================================

def main():
    """Main function"""
    
    print(f"\n{'='*60}")
    print("TB COUGH DETECTION - RECORD & PREDICT")
    print(f"{'='*60}\n")
    
    # Record audio
    audio, audio_file = record_audio(duration=7, sr=16000)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Predict
    tb_prob, confidence, prediction = predict_tb_from_audio_array(
        audio=audio,
        sr=16000,
        model_path='models/tb_detector.pth',
        device=device
    )
    
    # Print results
    if tb_prob is not None:
        print_results(tb_prob, confidence, prediction)
        print(f"📁 Recorded audio saved as: {audio_file}")
        print(f"   You can replay it or test again using:")
        print(f"   python predict_tb.py {audio_file}\n")
    else:
        print("\n❌ Prediction failed.\n")


if __name__ == "__main__":
    main()