"""
TB COUGH DETECTION FROM SPECTROGRAM IMAGES
==========================================
Modified to work with existing spectrogram images instead of audio files.
Compares spectrograms from CSI folders (5. Tuberculosis/CSI and 9. Normal/CSI)

Usage:
1. Spectrograms should be in:
   - dataset/5. Tuberculosis/CSI/*.png (or .jpg)
   - dataset/9. Normal/CSI/*.png (or .jpg)
2. Run training on these images
3. For prediction: provide audio -> generate spectrogram -> compare with dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import torchaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve)
import os
import random
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ============================================
# PART 1: SPECTROGRAM GENERATION FROM AUDIO
# ============================================

class SpectrogramGenerator:
    """Generate mel spectrograms from audio files for prediction"""
    
    def __init__(self, sr=16000, n_mels=128, n_fft=2048, hop_length=512):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def load_audio(self, file_path, duration=7):
        """Load audio file and resample"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sr, duration=duration)
            
            # Pad or trim to fixed length
            target_length = self.sr * duration
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                audio = audio[:target_length]
            
            return audio, sr
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None
    
    def audio_to_spectrogram(self, audio):
        """Convert audio to mel spectrogram"""
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
    
    def audio_to_image(self, audio, save_path=None):
        """Convert audio to spectrogram image (for visualization/comparison)"""
        spec = self.audio_to_spectrogram(audio)
        
        # Create image
        plt.figure(figsize=(10, 4))
        plt.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close()
            return save_path
        else:
            plt.close()
            return spec
    
    def process_audio_file(self, file_path, duration=7):
        """Process audio file to tensor (for prediction)"""
        audio, sr = self.load_audio(file_path, duration)
        if audio is None:
            return None
        
        spec = self.audio_to_spectrogram(audio)
        spec_tensor = torch.FloatTensor(spec).unsqueeze(0)
        spec_tensor = spec_tensor.repeat(3, 1, 1)
        
        return spec_tensor


# ============================================
# PART 2: LOAD SPECTROGRAM IMAGES
# ============================================

def load_spectrogram_image(image_path, target_size=(128, 128)):
    """
    Load a spectrogram image and convert to tensor
    
    Args:
        image_path: Path to spectrogram image
        target_size: Resize to this size (height, width)
    
    Returns:
        Tensor of shape [3, H, W]
    """
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])
        
        # Convert to tensor
        img_tensor = transform(img)
        
        return img_tensor
    
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def load_dataset_from_image_folders(tb_folder, normal_folder, target_size=(128, 256)):
    """
    Load dataset from spectrogram image folders
    
    Structure:
    dataset/
    ├── 5. Tuberculosis/
    │   └── CSI/
    │       ├── spec1.png
    │       ├── spec2.png
    │       └── ...
    └── 9. Normal/
        └── CSI/
            ├── spec1.png
            ├── spec2.png
            └── ...
    
    Args:
        tb_folder: Path to TB spectrogram images (e.g., "dataset/5. Tuberculosis/CSI")
        normal_folder: Path to Normal spectrogram images (e.g., "dataset/9. Normal/CSI")
        target_size: Resize images to this size (height, width)
    
    Returns:
        spectrograms: List of image tensors
        labels: List of labels (0=Normal, 1=TB)
    """
    print("Loading spectrogram images from folders...")
    
    spectrograms = []
    labels = []
    
    # Supported image formats
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    
    # Load TB samples
    if os.path.exists(tb_folder):
        tb_files = []
        for ext in image_extensions:
            tb_files.extend(list(Path(tb_folder).glob(ext)))
        
        print(f"Found {len(tb_files)} TB spectrogram images in {tb_folder}")
        
        for file_path in tb_files:
            img_tensor = load_spectrogram_image(str(file_path), target_size)
            if img_tensor is not None:
                spectrograms.append(img_tensor)
                labels.append(1)  # TB = 1
    else:
        print(f"⚠️  TB folder not found: {tb_folder}")
    
    # Load Normal samples
    if os.path.exists(normal_folder):
        normal_files = []
        for ext in image_extensions:
            normal_files.extend(list(Path(normal_folder).glob(ext)))
        
        print(f"Found {len(normal_files)} Normal spectrogram images in {normal_folder}")
        
        for file_path in normal_files:
            img_tensor = load_spectrogram_image(str(file_path), target_size)
            if img_tensor is not None:
                spectrograms.append(img_tensor)
                labels.append(0)  # Normal = 0
    else:
        print(f"⚠️  Normal folder not found: {normal_folder}")
    
    print(f"\n✅ Dataset loaded:")
    print(f"  Total samples: {len(spectrograms)}")
    print(f"  TB samples: {sum(labels)}")
    print(f"  Normal samples: {len(labels) - sum(labels)}")
    
    return spectrograms, labels


# ============================================
# PART 3: DATA AUGMENTATION
# ============================================

class SpectrogramAugmentation:
    """Image augmentation for spectrograms"""
    
    def __init__(self):
        self.transforms = {
            'horizontal_flip': transforms.RandomHorizontalFlip(p=1.0),
            'color_jitter_1': transforms.ColorJitter(brightness=0.2, contrast=0.2),
            'color_jitter_2': transforms.ColorJitter(brightness=0.3, contrast=0.3),
            'gaussian_blur': transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            'rotation_5': transforms.RandomRotation(degrees=5),
            'rotation_10': transforms.RandomRotation(degrees=10),
        }
    
    def freq_mask(self, spec, mask_param=15):
        """Frequency masking"""
        spec = spec.clone()
        num_freq = spec.shape[-2]
        f = random.randint(0, min(mask_param, num_freq))
        f0 = random.randint(0, max(num_freq - f, 1))
        spec[..., f0:f0+f, :] = 0
        return spec
    
    def time_mask(self, spec, mask_param=20):
        """Time masking"""
        spec = spec.clone()
        num_time = spec.shape[-1]
        t = random.randint(0, min(mask_param, num_time))
        t0 = random.randint(0, max(num_time - t, 1))
        spec[..., :, t0:t0+t] = 0
        return spec
    
    def add_noise(self, spec, noise_level=0.01):
        """Add Gaussian noise"""
        noise = torch.randn_like(spec) * noise_level
        return torch.clamp(spec + noise, 0, 1)
    
    def augment_single(self, spec, num_augmentations=20):
        """Generate multiple augmented versions"""
        augmented = [spec]
        
        transform_list = list(self.transforms.values())
        
        for i in range(num_augmentations - 1):
            aug_spec = spec.clone()
            
            # Random transformations
            num_transforms = random.randint(1, 3)
            selected = random.sample(transform_list, num_transforms)
            
            for transform in selected:
                try:
                    aug_spec = transform(aug_spec)
                except:
                    pass
            
            # Masking
            if random.random() > 0.5:
                aug_spec = self.freq_mask(aug_spec)
            if random.random() > 0.5:
                aug_spec = self.time_mask(aug_spec)
            
            # Noise
            if random.random() > 0.7:
                aug_spec = self.add_noise(aug_spec)
            
            augmented.append(aug_spec)
        
        return augmented


# ============================================
# PART 4: DATASET CLASS
# ============================================

class TBSpectrogramDataset(Dataset):
    """Dataset for TB detection from spectrogram images"""
    
    def __init__(self, spectrograms, labels, augment=False, num_augmentations=20):
        self.spectrograms = spectrograms
        self.labels = labels
        self.augment = augment
        self.num_augmentations = num_augmentations
        self.augmenter = SpectrogramAugmentation() if augment else None
    
    def __len__(self):
        if self.augment:
            return len(self.spectrograms) * self.num_augmentations
        return len(self.spectrograms)
    
    def __getitem__(self, idx):
        if self.augment:
            original_idx = idx // self.num_augmentations
            aug_idx = idx % self.num_augmentations
            
            spec = self.spectrograms[original_idx]
            label = self.labels[original_idx]
            
            if aug_idx > 0:
                augmented_list = self.augmenter.augment_single(spec, self.num_augmentations)
                spec = augmented_list[min(aug_idx, len(augmented_list) - 1)]
        else:
            spec = self.spectrograms[idx]
            label = self.labels[idx]
        
        if spec.dim() == 4 and spec.size(0) == 1:
            spec = spec.squeeze(0)
        
        label = torch.tensor(label, dtype=torch.long)
        
        return spec, label


# ============================================
# PART 5: MODEL
# ============================================

class TBDetectorResNet18(nn.Module):
    """ResNet18-based TB detector"""
    
    def __init__(self, pretrained=True, dropout=0.6):
        super().__init__()
        
        if pretrained:
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        else:
            self.backbone = models.resnet18(weights=None)
        
        for param in list(self.backbone.parameters())[:-30]:
            param.requires_grad = False
        
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
# PART 6: LABEL SMOOTHING LOSS
# ============================================

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=2, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


# ============================================
# PART 7: TRAINER
# ============================================

class TBDetectorTrainer:
    """Training pipeline"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(dataloader), 100. * correct / total
    
    def validate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                probs = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        return (total_loss / len(dataloader), 100. * correct / total,
                np.array(all_preds), np.array(all_labels), np.array(all_probs))
    
    def predict_with_tta(self, spectrogram, num_tta=10):
        """Prediction with Test-Time Augmentation"""
        self.model.eval()
        augmenter = SpectrogramAugmentation()
        predictions = []
        
        with torch.no_grad():
            # Original
            spec_batch = spectrogram.unsqueeze(0).to(self.device)
            output = self.model(spec_batch)
            prob = F.softmax(output, dim=1)
            predictions.append(prob.cpu())
            
            # Augmented
            aug_specs = augmenter.augment_single(spectrogram, num_tta)
            for aug_spec in aug_specs[1:]:
                aug_batch = aug_spec.unsqueeze(0).to(self.device)
                output = self.model(aug_batch)
                prob = F.softmax(output, dim=1)
                predictions.append(prob.cpu())
        
        avg_pred = torch.stack(predictions).mean(0)
        confidence = torch.stack(predictions).std(0)
        
        return avg_pred[0], confidence[0]


# ============================================
# PART 8: TRAINING FUNCTION
# ============================================

def train_tb_detector(spectrograms, labels, save_path='models/tb_detector.pth',
                      n_splits=5, epochs=50, batch_size=8):
    """
    Train TB detector on spectrogram images
    
    Args:
        spectrograms: List of image tensors
        labels: List of labels
        save_path: Where to save model
        n_splits: CV folds
        epochs: Training epochs (reduced to 50 for faster training)
        batch_size: Batch size
    """
    
    if len(spectrograms) == 0:
        print("❌ No data loaded!")
        return None
    
    if len(set(labels)) < 2:
        print("❌ Need both TB and Normal samples!")
        return None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    
    labels_np = np.array(labels)
    
    # Choose CV strategy
    if len(labels) < 30:
        print("Small dataset - using Leave-One-Out CV")
        cv = LeaveOneOut()
        n_splits = len(labels)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    print(f"\n{'='*70}")
    print(f"TB DETECTION TRAINING - SPECTROGRAM IMAGES")
    print(f"{'='*70}")
    print(f"Total: {len(labels)} | TB: {sum(labels)} | Normal: {len(labels)-sum(labels)}")
    print(f"CV: {n_splits}-Fold")
    print(f"{'='*70}\n")
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(spectrograms, labels_np)):
        print(f"\n--- FOLD {fold + 1}/{n_splits} ---")
        print("⏳ Preparing data...")
        
        train_specs = [spectrograms[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_specs = [spectrograms[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        
        print(f"✓ Data split complete")
        print(f"⏳ Creating datasets with augmentation...")
        
        train_dataset = TBSpectrogramDataset(train_specs, train_labels, augment=True, num_augmentations=15)
        val_dataset = TBSpectrogramDataset(val_specs, val_labels, augment=False)
        
        print(f"✓ Datasets created")
        print(f"⏳ Creating data loaders...")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"✓ Data loaders ready")
        print(f"⏳ Loading model (this may take a moment)...")
        
        model = TBDetectorResNet18(pretrained=True, dropout=0.6).to(device)
        trainer = TBDetectorTrainer(model, device)
        
        print(f"✓ Model loaded")
        print(f"⏳ Starting training...")
        
        criterion = LabelSmoothingLoss(classes=2, smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.5)
        
        best_val_acc = 0
        patience = 0
        
        for epoch in range(epochs):
            if epoch == 0:
                print(f"⏳ Epoch 1 starting...")
            train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc, _, _, _ = trainer.validate(val_loader, criterion)
            
            scheduler.step(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
                torch.save(model.state_dict(), f'best_model_fold_{fold}.pth')
            else:
                patience += 1
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d} | Train: {train_acc:5.2f}% | Val: {val_acc:5.2f}% | Best: {best_val_acc:.2f}%")
            
            if patience >= 15:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Evaluate
        model.load_state_dict(torch.load(f'best_model_fold_{fold}.pth'))
        _, val_acc, preds, true_labels, probs = trainer.validate(val_loader, criterion)
        
        print(f"\nFold {fold+1} Results: Accuracy = {val_acc:.2f}%")
        
        try:
            auc = roc_auc_score(true_labels, probs)
            print(f"AUC-ROC: {auc:.4f}")
        except:
            auc = 0
        
        cm = confusion_matrix(true_labels, preds)
        print(f"Confusion Matrix:\n{cm}")
        
        fold_results.append({
            'fold': fold + 1,
            'accuracy': val_acc,
            'auc': auc,
            'confusion_matrix': cm
        })
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    accuracies = [r['accuracy'] for r in fold_results]
    print(f"Mean Accuracy: {np.mean(accuracies):.2f}% (±{np.std(accuracies):.2f}%)")
    
    total_cm = sum([r['confusion_matrix'] for r in fold_results])
    print(f"Total Confusion Matrix:\n{total_cm}")
    
    # Train final model
    print(f"\n{'='*70}")
    print("TRAINING FINAL MODEL")
    print(f"{'='*70}\n")
    
    final_model = TBDetectorResNet18(pretrained=True, dropout=0.5).to(device)
    full_dataset = TBSpectrogramDataset(spectrograms, labels, augment=True, num_augmentations=15)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = LabelSmoothingLoss(classes=2, smoothing=0.1)
    optimizer = optim.AdamW(final_model.parameters(), lr=0.0001, weight_decay=1e-3)
    
    final_model.train()
    for epoch in range(100):
        total_loss = 0
        correct = 0
        total = 0
        
        for images, label in full_loader:
            images, label = images.to(device), label.to(device)
            
            optimizer.zero_grad()
            output = final_model(images)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
        
        if (epoch + 1) % 10 == 0:
            acc = 100. * correct / total
            print(f"Epoch {epoch+1:3d}/100 | Loss: {total_loss/len(full_loader):.4f} | Acc: {acc:.2f}%")
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    torch.save(final_model.state_dict(), save_path)
    print(f"\n✅ Model saved to: {save_path}")
    
    return final_model, fold_results


# ============================================
# PART 9: PREDICTION FROM AUDIO
# ============================================

def predict_from_audio(audio_path, model_path, device='cpu'):
    """
    Predict TB from audio file by generating spectrogram
    
    Args:
        audio_path: Path to audio file (.wav, .mp3)
        model_path: Path to trained model
        device: 'cpu' or 'cuda'
    
    Returns:
        tb_probability: Probability of TB (0-1)
        confidence: Prediction confidence
    """
    
    # Load model
    model = TBDetectorResNet18(pretrained=False, dropout=0.5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Generate spectrogram from audio
    spec_gen = SpectrogramGenerator(sr=16000, n_mels=128)
    spec_tensor = spec_gen.process_audio_file(audio_path, duration=7)
    
    if spec_tensor is None:
        print(f"❌ Failed to process audio: {audio_path}")
        return None, None
    
    # Resize to match training size
    resize_transform = transforms.Resize((128, 256))
    spec_tensor = resize_transform(spec_tensor)
    
    # Predict with TTA
    trainer = TBDetectorTrainer(model, device)
    pred, conf = trainer.predict_with_tta(spec_tensor, num_tta=10)
    
    tb_prob = pred[1].item()
    confidence = 1 - conf[1].item()
    
    return tb_prob, confidence


# ============================================
# PART 10: MAIN USAGE
# ============================================

if __name__ == "__main__":
    
    print("="*70)
    print("TB DETECTION FROM SPECTROGRAM IMAGES")
    print("="*70)
    
    # ===== STEP 1: LOAD DATASET FROM IMAGE FOLDERS =====
    tb_folder = "dataset/5. Tuberculosis/CSI"
    normal_folder = "dataset/9. Normal/CSI"
    
    spectrograms, labels = load_dataset_from_image_folders(
        tb_folder=tb_folder,
        normal_folder=normal_folder,
        target_size=(128, 256)  # Height x Width
    )
    
    # ===== STEP 2: TRAIN MODEL =====
    if len(spectrograms) > 0:
        final_model, cv_results = train_tb_detector(
            spectrograms=spectrograms,
            labels=labels,
            save_path='models/tb_detector.pth',
            n_splits=5,
            epochs=50,  # Reduced for faster training
            batch_size=8
        )
    
    # ===== STEP 3: PREDICT FROM NEW AUDIO FILE =====
    test_audio = "test_cough.wav"
    
    if os.path.exists(test_audio) and os.path.exists('models/tb_detector.pth'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tb_prob, confidence = predict_from_audio(
            audio_path=test_audio,
            model_path='models/tb_detector.pth',
            device=device
        )
        
        if tb_prob is not None:
            print(f"\n{'='*50}")
            print("PREDICTION RESULTS")
            print(f"{'='*50}")
            print(f"File: {test_audio}")
            print(f"TB Probability: {tb_prob*100:.2f}%")
            print(f"Confidence: {confidence*100:.2f}%")
            
            if tb_prob > 0.7:
                print("⚠️  HIGH RISK - Recommend medical consultation")
            elif tb_prob > 0.4:
                print("⚡ MODERATE RISK - Monitor symptoms")
            else:
                print("✅ LOW RISK")
            print(f"{'='*50}\n")
    
    print("\n✅ Complete!")