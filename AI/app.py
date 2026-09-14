"""
TB COUGH DETECTION - STREAMLIT WEB APP
=======================================
A web interface for detecting tuberculosis from cough sounds.

Installation:
    pip install streamlit sounddevice scipy
    -r requirements.txt

Usage:
    streamlit run app.py
"""

import streamlit as st
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import librosa
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path
import tempfile


# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="TB Cough Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


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
        return mel_spec_norm, mel_spec_db
    
    def process_audio_array(self, audio, duration=7):
        target_length = self.sr * duration
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        
        spec, spec_db = self.audio_to_spectrogram(audio)
        spec_tensor = torch.FloatTensor(spec).unsqueeze(0).repeat(3, 1, 1)
        return spec_tensor, spec_db


# ============================================
# LOAD MODEL (CACHED)
# ============================================

@st.cache_resource
def load_model(model_path='models/tb_detector.pth'):
    """Load model (cached to avoid reloading)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(model_path):
        return None, device
    
    model = TBDetectorResNet18(pretrained=False, dropout=0.5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, device


# ============================================
# AUDIO RECORDING
# ============================================

def record_audio(duration=7, sr=16000):
    """Record audio from microphone"""
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()


# ============================================
# PREDICTION
# ============================================

def predict_tb(audio, sr, model, device):
    """Predict TB from audio"""
    
    spec_gen = SpectrogramGenerator(sr=sr, n_mels=128)
    
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    spec_tensor, spec_db = spec_gen.process_audio_array(audio, duration=7)
    
    resize_transform = transforms.Resize((128, 256))
    spec_tensor = resize_transform(spec_tensor)
    
    with torch.no_grad():
        spec_batch = spec_tensor.unsqueeze(0).to(device)
        output = model(spec_batch)
        prob = F.softmax(output, dim=1)
    
    tb_prob = prob[0][1].item() * 100
    normal_prob = prob[0][0].item() * 100
    confidence = torch.max(prob).item() * 100
    prediction = "TB" if tb_prob > 50 else "Normal"
    
    return tb_prob, normal_prob, confidence, prediction, spec_db


# ============================================
# VISUALIZATIONS
# ============================================

def plot_spectrogram(spec_db):
    """Plot spectrogram"""
    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.imshow(spec_db, aspect='auto', origin='lower', cmap='viridis')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Mel Frequency', fontsize=12)
    ax.set_title('Mel Spectrogram', fontsize=14, fontweight='bold')
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    return fig


def create_gauge_chart(value, title):
    """Create a gauge chart for probability"""
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={'projection': 'polar'})
    
    theta = np.radians(180 * value / 100)
    
    ax.barh(0, np.pi, height=0.3, color='lightgray', alpha=0.3)
    ax.barh(0, theta, height=0.3, color='red' if value > 70 else 'orange' if value > 50 else 'green')
    
    ax.set_ylim(0, 1)
    ax.set_xlim(0, np.pi)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)
    
    ax.text(np.pi/2, 0.5, f"{value:.1f}%", 
            ha='center', va='center', fontsize=24, fontweight='bold')
    ax.text(np.pi/2, -0.2, title, 
            ha='center', va='center', fontsize=14)
    
    return fig


# ============================================
# MAIN APP
# ============================================

def main():
    
    st.title("🔬 TB Cough Detection System")
    st.markdown("### AI-Powered Tuberculosis Screening from Cough Sounds")
    st.markdown("---")
    
    # Load model
    model, device = load_model('models/tb_detector.pth')
    
    if model is None:
        st.error("❌ Model not found! Please train the model first using `train_improved.py`")
        st.stop()
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        recording_duration = st.slider(
            "Recording Duration (seconds)",
            min_value=3,
            max_value=10,
            value=7,
            help="How long to record the cough"
        )
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.info(
            """
            This AI system detects patterns in cough sounds that may indicate tuberculosis.
            
            **How it works:**
            1. Record or upload a cough sound
            2. AI converts it to a spectrogram
            3. Analyzes patterns using deep learning
            4. Provides TB probability
            
            **⚠️ Important:**
            This is a screening tool, NOT a medical diagnosis. 
            Always consult healthcare professionals.
            """
        )
        
        st.markdown("---")
        st.markdown("**Device:** " + ("🖥️ GPU (CUDA)" if device.type == 'cuda' else "💻 CPU"))
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🎤 Record Audio", "📁 Upload Audio", "📊 About the Model"])
    
    # ============================================
    # TAB 1: RECORD AUDIO
    # ============================================
    with tab1:
        st.header("Record a Cough Sound")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.info("🎤 Click the button below to start recording. Make sure your microphone is working!")
            
            if st.button("🔴 Start Recording", type="primary", use_container_width=True):
                
                countdown_placeholder = st.empty()
                for i in range(3, 0, -1):
                    countdown_placeholder.warning(f"⏱️ Recording starts in {i}...")
                    time.sleep(1)
                
                countdown_placeholder.success("🔴 RECORDING NOW - Please cough!")
                
                with st.spinner("Recording..."):
                    audio = record_audio(duration=recording_duration, sr=16000)
                
                countdown_placeholder.success("✅ Recording complete!")
                
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                wav.write(temp_file.name, 16000, (audio * 32767).astype(np.int16))
                
                st.audio(temp_file.name, format='audio/wav')
                
                with st.spinner("🔍 Analyzing cough pattern..."):
                    tb_prob, normal_prob, confidence, prediction, spec_db = predict_tb(
                        audio, 16000, model, device
                    )
                
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == "TB":
                        st.error(f"### ⚠️ {prediction}")
                    else:
                        st.success(f"### ✅ {prediction}")
                
                with col2:
                    st.metric("TB Probability", f"{tb_prob:.1f}%")
                
                with col3:
                    st.metric("Confidence", f"{confidence:.1f}%")
                
                st.markdown("#### Detailed Probabilities")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔴 TB Probability**")
                    st.progress(tb_prob / 100)
                    st.markdown(f"**{tb_prob:.2f}%**")
                
                with col2:
                    st.markdown("**🟢 Normal Probability**")
                    st.progress(normal_prob / 100)
                    st.markdown(f"**{normal_prob:.2f}%**")
                
                st.markdown("---")
                st.subheader("⚕️ Risk Assessment")
                
                if tb_prob > 70:
                    st.error("""
                    **⚠️ HIGH RISK**
                    - Pattern shows strong similarity to TB cough
                    - **STRONGLY RECOMMEND** immediate medical consultation
                    - Get proper TB testing (sputum test, chest X-ray)
                    - Do not delay seeking medical attention
                    """)
                elif tb_prob > 50:
                    st.warning("""
                    **⚡ MODERATE-HIGH RISK**
                    - Some TB-like patterns detected
                    - Recommend medical consultation
                    - Monitor symptoms closely
                    - Seek medical attention if symptoms worsen
                    """)
                elif tb_prob > 30:
                    st.warning("""
                    **⚡ MODERATE RISK**
                    - Mild TB-like patterns present
                    - Monitor symptoms
                    - Consult doctor if symptoms persist or worsen
                    """)
                else:
                    st.success("""
                    **✅ LOW RISK**
                    - Pattern not consistent with TB cough
                    - Continue monitoring if symptoms persist
                    - Seek medical attention if condition worsens
                    """)
                
                st.markdown("---")
                st.subheader("📈 Spectrogram Analysis")
                fig = plot_spectrogram(spec_db)
                st.pyplot(fig)
                
                st.markdown("---")
                st.warning("""
                **⚠️ IMPORTANT MEDICAL DISCLAIMER**
                
                This AI system is a **screening tool only** and is NOT a substitute for professional medical diagnosis. 
                
                - Results should not be used as the sole basis for medical decisions
                - Always consult qualified healthcare professionals for proper diagnosis
                - TB diagnosis requires laboratory tests (sputum culture, PCR) and imaging (chest X-ray)
                - If you have persistent cough, fever, night sweats, or weight loss, seek medical attention immediately
                """)
                
                # Clean up
                os.unlink(temp_file.name)
    
    # ============================================
    # TAB 2: UPLOAD AUDIO
    # ============================================
    with tab2:
        st.header("Upload an Audio File")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file (WAV, MP3, etc.)",
            type=['wav', 'mp3', 'ogg', 'm4a', 'flac'],
            help="Upload a recording of a cough sound"
        )
        
        if uploaded_file is not None:
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
            temp_file.write(uploaded_file.read())
            temp_file.close()
            
            try:
                audio, sr = librosa.load(temp_file.name, sr=16000, duration=10)
                
                st.audio(temp_file.name)
                
                with st.spinner("🔍 Analyzing cough pattern..."):
                    tb_prob, normal_prob, confidence, prediction, spec_db = predict_tb(
                        audio, sr, model, device
                    )
                
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == "TB":
                        st.error(f"### ⚠️ {prediction}")
                    else:
                        st.success(f"### ✅ {prediction}")
                
                with col2:
                    st.metric("TB Probability", f"{tb_prob:.1f}%")
                
                with col3:
                    st.metric("Confidence", f"{confidence:.1f}%")
                
                st.markdown("#### Detailed Probabilities")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔴 TB Probability**")
                    st.progress(tb_prob / 100)
                    st.markdown(f"**{tb_prob:.2f}%**")
                
                with col2:
                    st.markdown("**🟢 Normal Probability**")
                    st.progress(normal_prob / 100)
                    st.markdown(f"**{normal_prob:.2f}%**")
                
                st.markdown("---")
                st.subheader("⚕️ Risk Assessment")
                
                if tb_prob > 70:
                    st.error("**⚠️ HIGH RISK** - Strongly recommend immediate medical consultation")
                elif tb_prob > 50:
                    st.warning("**⚡ MODERATE-HIGH RISK** - Recommend medical consultation")
                elif tb_prob > 30:
                    st.warning("**⚡ MODERATE RISK** - Monitor symptoms closely")
                else:
                    st.success("**✅ LOW RISK** - Pattern not consistent with TB")
                
                st.markdown("---")
                st.subheader("📈 Spectrogram Analysis")
                fig = plot_spectrogram(spec_db)
                st.pyplot(fig)
                
                os.unlink(temp_file.name)
                
            except Exception as e:
                st.error(f"❌ Error processing audio file: {e}")
    
    # ============================================
    # TAB 3: ABOUT MODEL
    # ============================================
    with tab3:
        st.header("About the AI Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🧠 Model Architecture
            
            - **Base Model:** ResNet-18
            - **Input:** Mel Spectrogram (128 x 256)
            - **Training:** Transfer Learning
            - **Classes:** TB vs Normal
            
            ### 📊 How It Works
            
            1. **Audio Input:** Cough sound (7 seconds)
            2. **Preprocessing:** Convert to mel spectrogram
            3. **Feature Extraction:** Deep CNN analysis
            4. **Classification:** TB probability output
            """)
        
        with col2:
            st.markdown("""
            ### 🎯 Model Features
            
            - ✅ Transfer learning from ImageNet
            - ✅ Data augmentation for robustness
            - ✅ Cross-validation training
            - ✅ Label smoothing
            - ✅ Dropout regularization
            
            ### ⚙️ Technical Details
            
            - **Sample Rate:** 16 kHz
            - **FFT Size:** 2048
            - **Mel Bins:** 128
            - **Duration:** 7 seconds
            """)
        
        st.markdown("---")
        st.markdown("""
        ### 📚 Training Data
        
        The model was trained on spectrogram images from:
        - **TB Samples:** `dataset/5. Tuberculosis/CSI/`
        - **Normal Samples:** `dataset/9. Normal/CSI/`
        
        ### 🔬 Research Basis
        
        This approach is based on research showing that TB affects cough acoustics through:
        - Changes in lung tissue
        - Airway inflammation
        - Altered breathing patterns
        
        Spectrograms capture these acoustic differences that may not be audible to the human ear.
        """)


# ============================================
# RUN APP
# ============================================

if __name__ == "__main__":
    main()