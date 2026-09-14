import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import random

# ============================================
# PART 1: IMAGE AUGMENTATION FOR SPECTROGRAMS
# ============================================

class SpectrogramAugmentation:
    """Image-based augmentation for spectrograms"""
    
    def __init__(self):
        # Define different augmentation transforms
        self.augmentations = [
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=1.0),
            transforms.RandomAutocontrast(p=1.0),
            transforms.RandomEqualize(p=1.0),
        ]
        
        # Additional transforms
        self.noise_transform = lambda x: x + torch.randn_like(x) * 0.01
        self.rotation_5 = transforms.RandomRotation(degrees=5)
        self.rotation_10 = transforms.RandomRotation(degrees=10)
    
    def augment(self, image, num_augmentations=10):
        """Generate multiple augmented versions of spectrogram"""
        augmented_list = [image]  # Original
        
        # Apply various augmentations
        for i, aug_transform in enumerate(self.augmentations):
            if len(augmented_list) >= num_augmentations:
                break
            try:
                augmented = aug_transform(image)
                augmented_list.append(augmented)
            except:
                pass
        
        # Add noise augmentation
        if len(augmented_list) < num_augmentations:
            augmented_list.append(self.noise_transform(image))
        
        # Add rotation
        if len(augmented_list) < num_augmentations:
            augmented_list.append(self.rotation_5(image))
        
        if len(augmented_list) < num_augmentations:
            augmented_list.append(self.rotation_10(image))
        
        # Combined augmentations
        while len(augmented_list) < num_augmentations:
            try:
                # Combine two random augmentations
                aug1 = random.choice(self.augmentations[:4])
                aug2 = random.choice(self.augmentations[4:])
                combined = aug2(aug1(image))
                augmented_list.append(combined)
            except:
                augmented_list.append(image)
        
        return augmented_list[:num_augmentations]


# ============================================
# PART 2: DATASET CLASS
# ============================================

class SpectrogramDataset(Dataset):
    """Dataset with augmentation support for spectrograms"""
    
    def __init__(self, image_list, labels, augment=False, num_augmentations=10):
        self.image_list = image_list
        self.labels = labels
        self.augment = augment
        self.augmenter = SpectrogramAugmentation() if augment else None
        self.num_augmentations = num_augmentations
        
    def __len__(self):
        if self.augment:
            return len(self.image_list) * self.num_augmentations
        return len(self.image_list)
    
    def __getitem__(self, idx):
        if self.augment:
            original_idx = idx // self.num_augmentations
            aug_idx = idx % self.num_augmentations
            
            image = self.image_list[original_idx]
            label = self.labels[original_idx]
            
            # Apply augmentation
            if aug_idx > 0:
                augmented_versions = self.augmenter.augment(image, self.num_augmentations)
                image = augmented_versions[min(aug_idx, len(augmented_versions) - 1)]
        else:
            image = self.image_list[idx]
            label = self.labels[idx]
        
        # Remove batch dimension if present
        if image.dim() == 4 and image.size(0) == 1:
            image = image.squeeze(0)
        
        return image, label


# ============================================
# PART 3: IMPROVED MODEL WITH TRANSFER LEARNING
# ============================================

class ImprovedTBDetector(nn.Module):
    """Transfer learning with ResNet18"""
    
    def __init__(self, freeze_backbone=True):
        super().__init__()
        
        # Load pre-trained ResNet18
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        return self.backbone(x)


# ============================================
# PART 4: TRAINING WITH CROSS-VALIDATION
# ============================================

class Trainer:
    """Training pipeline with cross-validation"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.best_model_state = None
        self.best_score = 0
    
    def train_epoch(self, dataloader, optimizer, criterion):
        """Train for one epoch"""
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
        """Validate model"""
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
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        accuracy = 100. * correct / total
        return total_loss / len(dataloader), accuracy, all_preds, all_labels, all_probs
    
    def train_with_cross_validation(self, image_list, labels, n_splits=5, 
                                    epochs=50, lr=0.001, batch_size=4):
        """Train with k-fold cross-validation"""
        
        # Convert to numpy for sklearn
        labels_np = np.array(labels)
        
        # K-Fold cross-validation
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_results = []
        
        print(f"\n{'='*60}")
        print(f"Starting {n_splits}-Fold Cross-Validation")
        print(f"{'='*60}\n")
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(image_list, labels_np)):
            print(f"\n--- FOLD {fold + 1}/{n_splits} ---")
            
            # Split data
            train_images = [image_list[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            val_images = [image_list[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]
            
            print(f"Training samples: {len(train_images)} (will be augmented to {len(train_images) * 10})")
            print(f"Validation samples: {len(val_images)}")
            
            # Create datasets with augmentation for training
            train_dataset = SpectrogramDataset(train_images, train_labels, 
                                              augment=True, num_augmentations=10)
            val_dataset = SpectrogramDataset(val_images, val_labels, augment=False)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                     shuffle=True, drop_last=False)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                                   shuffle=False)
            
            # Reset model
            self.model = ImprovedTBDetector(freeze_backbone=True).to(self.device)
            
            # Setup training
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 
                                                             patience=5, factor=0.5)
            
            best_val_acc = 0
            patience_counter = 0
            
            # Training loop
            for epoch in range(epochs):
                train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
                val_loss, val_acc, _, _, _ = self.validate(val_loader, criterion)
                
                scheduler.step(val_acc)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Early stopping
                if patience_counter >= 15:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Final validation
            _, val_acc, preds, true_labels, probs = self.validate(val_loader, criterion)
            
            # Calculate metrics
            print(f"\nFold {fold + 1} Results:")
            print(f"Validation Accuracy: {val_acc:.2f}%")
            
            try:
                auc = roc_auc_score(true_labels, probs)
                print(f"AUC-ROC: {auc:.4f}")
            except:
                auc = 0
                print("AUC-ROC: Could not calculate (need both classes in validation)")
            
            print("\nClassification Report:")
            print(classification_report(true_labels, preds, 
                                       target_names=['Non-TB', 'TB'], 
                                       zero_division=0))
            
            print("\nConfusion Matrix:")
            print(confusion_matrix(true_labels, preds))
            
            fold_results.append({
                'accuracy': val_acc,
                'auc': auc,
                'predictions': preds,
                'true_labels': true_labels
            })
        
        # Summary
        print(f"\n{'='*60}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*60}")
        accuracies = [r['accuracy'] for r in fold_results]
        aucs = [r['auc'] for r in fold_results if r['auc'] > 0]
        
        print(f"Mean Accuracy: {np.mean(accuracies):.2f}% (+/- {np.std(accuracies):.2f}%)")
        if aucs:
            print(f"Mean AUC-ROC: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")
        
        return fold_results


# ============================================
# PART 5: MAIN TRAINING SCRIPT
# ============================================

def train_improved_model(image_list, labels, save_path="models/tb_detector_improved.pth"):
    """
    Main training function
    
    Args:
        image_list: List of spectrogram tensors
        labels: List of labels (0 for non-TB, 1 for TB)
        save_path: Path to save the best model
    """
    
    print(f"\n{'='*60}")
    print("TB COUGH DETECTOR - IMPROVED TRAINING")
    print(f"{'='*60}\n")
    print(f"Total samples: {len(image_list)}")
    print(f"TB samples: {sum(labels)}")
    print(f"Non-TB samples: {len(labels) - sum(labels)}")
    print(f"Class distribution: {sum(labels)/len(labels)*100:.1f}% TB")
    
    # Check if we have both classes
    if len(set(labels)) < 2:
        print("\n⚠️  WARNING: Only one class present in dataset!")
        print("You need both TB and non-TB samples to train a classifier.")
        return None
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Initialize model and trainer
    model = ImprovedTBDetector(freeze_backbone=True)
    trainer = Trainer(model, device=device)
    
    # Train with cross-validation
    results = trainer.train_with_cross_validation(
        image_list, 
        labels,
        n_splits=min(5, len(image_list) // 2),  # Adjust folds for small dataset
        epochs=50,
        lr=0.001,
        batch_size=4  # Small batch size for small dataset
    )
    
    # Train final model on all data
    print(f"\n{'='*60}")
    print("Training final model on all data...")
    print(f"{'='*60}\n")
    
    final_model = ImprovedTBDetector(freeze_backbone=True).to(device)
    dataset = SpectrogramDataset(image_list, labels, augment=True, num_augmentations=10)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(final_model.parameters(), lr=0.001, weight_decay=1e-4)
    
    final_model.train()
    for epoch in range(100):
        total_loss = 0
        for images, label in dataloader:
            images, label = images.to(device), label.to(device)
            optimizer.zero_grad()
            output = final_model(images)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/100 - Loss: {total_loss/len(dataloader):.4f}")
    
    # Save model
    torch.save(final_model.state_dict(), save_path)
    print(f"\n✓ Model saved to {save_path}")
    
    return final_model, results