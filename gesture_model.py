"""
Gesture Recognition CNN Model
Based on principles from complete_deep_learning_tutorial.py

This model classifies hand gestures (fist vs peace sign) using a CNN architecture
similar to LeNet-5, with adjustments for hand gesture recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GestureRecognitionCNN(nn.Module):
    """
    CNN for Hand Gesture Recognition
    
    Architecture:
    - Input: 64x64 grayscale image
    - Conv1: 1 -> 16 channels, 5x5 kernel
    - MaxPool: 2x2
    - Conv2: 16 -> 32 channels, 5x5 kernel
    - MaxPool: 2x2
    - Conv3: 32 -> 64 channels, 3x3 kernel
    - MaxPool: 2x2
    - Flatten + Fully Connected layers
    - Output: 2 classes (fist, peace)
    
    This follows the deep learning principles:
    - Convolutional layers detect local features (edges, patterns)
    - Pooling reduces spatial dimensions and provides translation invariance
    - Multiple layers build hierarchical representations
    - Fully-connected layers perform final classification
    """
    
    def __init__(self, num_classes=2):
        super(GestureRecognitionCNN, self).__init__()
        
        # Convolutional block 1
        # Input: 64x64x1 -> Output: 32x32x16
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, 
                               kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional block 2
        # Input: 32x32x16 -> Output: 16x16x32
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, 
                               kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional block 3
        # Input: 16x16x32 -> Output: 8x8x64
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, 
                               kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # After pooling: 8x8x64 = 4096 features
        self.fc1 = nn.Linear(8 * 8 * 64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Dropout for regularization (prevents overfitting)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, 1, 64, 64)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Convolutional block 1: Conv -> ReLU -> Pool
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Convolutional block 2: Conv -> ReLU -> Pool
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Convolutional block 3: Conv -> ReLU -> Pool
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)  # Reshape to (batch_size, 4096)
        
        # Fully connected layers with dropout
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x
    
    def predict(self, x):
        """
        Make prediction with softmax probabilities
        
        Args:
            x: Input tensor
            
        Returns:
            class_idx: Predicted class index
            probabilities: Softmax probabilities for each class
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            class_idx = torch.argmax(probabilities, dim=1)
        return class_idx.item(), probabilities.squeeze().tolist()


def create_model(num_classes=2):
    """
    Factory function to create and initialize the model
    
    Args:
        num_classes: Number of gesture classes (default: 2 for fist/peace)
        
    Returns:
        model: Initialized GestureRecognitionCNN
    """
    model = GestureRecognitionCNN(num_classes=num_classes)
    return model


def count_parameters(model):
    """
    Count the number of trainable parameters in the model
    
    Args:
        model: PyTorch model
        
    Returns:
        total_params: Total number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


if __name__ == "__main__":
    # Test the model
    print("=" * 60)
    print("Gesture Recognition CNN Model")
    print("=" * 60)
    
    model = create_model(num_classes=2)
    print(f"\nModel architecture:")
    print(model)
    
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 1, 64, 64)
    print(f"\nTest input shape: {test_input.shape}")
    
    output = model(test_input)
    print(f"Output shape: {output.shape}")
    print(f"Output logits (sample): {output[0].tolist()}")
    
    # Test prediction
    pred_class, probs = model.predict(test_input[0:1])
    print(f"\nPrediction test:")
    print(f"  Predicted class: {pred_class}")
    print(f"  Class probabilities: {probs}")
