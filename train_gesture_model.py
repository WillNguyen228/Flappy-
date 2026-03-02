"""
Training Script for Gesture Recognition Model

This script trains the CNN model using the deep learning principles from
complete_deep_learning_tutorial.py:
- Data loading and augmentation
- Training loop with gradient descent
- Validation for model selection
- Loss functions and optimizers
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
import cv2
import numpy as np
from gesture_model import GestureRecognitionCNN, count_parameters
import time
import matplotlib.pyplot as plt


class GestureDataset(Dataset):
    """
    Custom Dataset for loading hand gesture images
    
    Following the dataset pattern from the tutorial (Section 8 & 9)
    """
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Directory containing class subdirectories
            transform: Optional transforms to apply
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.classes = ['fist', 'peace']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all image paths and labels
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} does not exist")
                continue
            
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
        
        print(f"Loaded {len(self.samples)} images from {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image as grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Convert to PIL-like format for transforms
        image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
        image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(augment=True):
    """
    Create data augmentation transforms
    
    Following Section 9: Data Augmentation from the tutorial
    - RandomHorizontalFlip: mirrors hand gestures
    - RandomRotation: rotates hand at different angles
    - RandomAffine: translation and slight scaling
    
    Args:
        augment: Whether to apply augmentation (True for training, False for val/test)
        
    Returns:
        transform: Composed transforms
    """
    if augment:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            # Add slight noise for robustness
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05),
            transforms.Lambda(lambda x: torch.clamp(x, 0, 1))  # Keep in valid range
        ])
    else:
        # No augmentation for validation/test
        transform = None
    
    return transform


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Train for one epoch
    
    Following the training pattern from Section 8.3 in the tutorial
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # Zero gradients (important: clear from previous iteration)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization (automatic differentiation)
        loss.backward()
        optimizer.step()
        
        # Track statistics
        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += images.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    """
    Evaluate model on validation set
    
    Following evaluation pattern from Section 8.3
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # No gradient computation during validation
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass only
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Track statistics
            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += images.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


def train_model(data_dir="gesture_data", epochs=20, batch_size=32, 
                learning_rate=0.001, model_save_path="gesture_model.pth"):
    """
    Main training function
    
    Follows the complete training pipeline from the tutorial:
    - Data loading with train/val split
    - Model initialization
    - Loss function (CrossEntropyLoss for classification)
    - Optimizer (Adam - adaptive learning rate)
    - Training loop with validation
    """
    print("=" * 80)
    print("TRAINING HAND GESTURE RECOGNITION MODEL")
    print("=" * 80)
    
    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # -------------------------------------------------------------------------
    # 1. Load and prepare data
    # -------------------------------------------------------------------------
    print("\n1. Loading dataset...")
    
    # Create datasets with augmentation
    train_transform = get_transforms(augment=True)
    val_transform = get_transforms(augment=False)
    
    # Load full dataset
    full_dataset = GestureDataset(data_dir, transform=None)
    
    if len(full_dataset) == 0:
        raise ValueError(f"No images found in {data_dir}. Please run collect_gesture_data.py first.")
    
    # Split into train and validation (80/20 split)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Batch size: {batch_size}")
    
    # -------------------------------------------------------------------------
    # 2. Create model
    # -------------------------------------------------------------------------
    print("\n2. Creating CNN model...")
    
    model = GestureRecognitionCNN(num_classes=2).to(device)
    total_params = count_parameters(model)
    
    print(f"Model: GestureRecognitionCNN")
    print(f"Total trainable parameters: {total_params:,}")
    
    # -------------------------------------------------------------------------
    # 3. Define loss function and optimizer
    # -------------------------------------------------------------------------
    print("\n3. Setting up training...")
    
    # CrossEntropyLoss for multi-class classification (Section 6)
    criterion = nn.CrossEntropyLoss()
    
    # Adam optimizer - adaptive learning rate (better than SGD for most cases)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler - reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    print(f"Loss function: CrossEntropyLoss")
    print(f"Optimizer: Adam (lr={learning_rate})")
    print(f"Scheduler: ReduceLROnPlateau")
    
    # -------------------------------------------------------------------------
    # 4. Training loop
    # -------------------------------------------------------------------------
    print(f"\n4. Training for {epochs} epochs...")
    print("-" * 80)
    
    # Track history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2%}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, model_save_path)
            print(f"  ✓ Saved best model (val_acc: {val_acc:.2%})")
    
    elapsed = time.time() - start_time
    print("-" * 80)
    print(f"\nTraining completed in {elapsed:.1f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.2%}")
    print(f"Model saved to: {model_save_path}")
    
    # -------------------------------------------------------------------------
    # 5. Plot training history
    # -------------------------------------------------------------------------
    print("\n5. Plotting training history...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history saved to: training_history.png")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    
    return model, history


if __name__ == "__main__":
    # Train the model
    model, history = train_model(
        data_dir="gesture_data",
        epochs=20,
        batch_size=32,
        learning_rate=0.001,
        model_save_path="gesture_model.pth"
    )
