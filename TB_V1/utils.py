import librosa
import numpy as np
import torch
import torch.nn.functional as F

def make_mel_spectrogram(audio, sr=16000):
    S = librosa.feature.melspectrogram(y=audio, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    tensor = torch.tensor(S_db).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(224,224))
    return tensor
