"""
================================================================================
COMPLETE DEEP LEARNING TUTORIAL WITH PYTORCH
================================================================================

This comprehensive educational file covers the fundamentals of deep learning
from basic tensor operations through convolutional neural networks and data
augmentation. Each section builds upon previous concepts to provide a complete
understanding of modern deep learning.

Topics Covered:
1. Tensor Basics and Manipulation
2. Tensor Operations and Reductions
3. Automatic Differentiation and Gradient Descent
4. Linear Regression (Closed-form and Neural Networks)
5. Multi-Layer Perceptrons (MLPs)
6. Classification and Regularization
7. Convolutions for Images
8. Convolutional Neural Networks (CNNs)
9. Data Augmentation Techniques

Author: Compiled from CPS-470 course materials
================================================================================
"""

import csv
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing, fetch_covtype
from sklearn.preprocessing import StandardScaler
import numpy as np


################################################################################
# SECTION 1: TENSOR BASICS AND MANIPULATION
################################################################################
# 
# Tensors are the fundamental data structure in PyTorch. They are 
# multi-dimensional arrays similar to NumPy arrays but with additional
# capabilities like automatic differentiation and GPU acceleration.
#
# Understanding tensor shapes and manipulation is crucial for building
# neural networks where data flows through layers with specific dimensions.
################################################################################

def section_1_tensor_basics():
    """
    SECTION 1: Creating and Manipulating Tensors
    
    Tensors can have different dimensions:
    - 0D tensor (scalar): single number
    - 1D tensor (vector): array of numbers
    - 2D tensor (matrix): table of numbers
    - 3D+ tensor: higher-dimensional arrays
    
    Common use cases:
    - 1D: single data point features, bias terms
    - 2D: batch of data samples (rows) with features (columns)
    - 3D: sequences, time series, or grayscale images
    - 4D: batch of RGB images (batch, channels, height, width)
    """
    print("\n" + "=" * 80)
    print("SECTION 1: TENSOR BASICS AND MANIPULATION")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # 1.1: Creating Tensors
    # -------------------------------------------------------------------------
    print("\n1.1: Creating Tensors")
    print("-" * 40)
    
    # Create tensors of ones with different shapes
    # These are useful for initialization or creating constant tensors
    ones_1d = torch.ones((5))          # Shape: (5,)
    ones_2d = torch.ones((3, 4))       # Shape: (3, 4)
    ones_3d = torch.ones((2, 3, 4))    # Shape: (2, 3, 4)
    
    print(f"1D tensor of ones: {ones_1d}")
    print(f"\n2D tensor of ones:\n{ones_2d}")
    print(f"\n3D tensor shape: {ones_3d.shape}")
    
    # Create a sequence of numbers
    # torch.arange(n) creates integers from 0 to n-1
    seq = torch.arange(12)
    print(f"\nSequence 0-11: {seq}")
    
    # Create random tensors - important for weight initialization
    # torch.randn samples from standard normal distribution (mean=0, std=1)
    rand_2d = torch.randn(3, 4)
    print(f"\nRandom 2D tensor (3x4):\n{rand_2d}")
    
    # -------------------------------------------------------------------------
    # 1.2: Reshaping Tensors
    # -------------------------------------------------------------------------
    # Reshaping allows us to change the dimensions of a tensor without
    # changing its data. This is essential when connecting layers with
    # different expected input/output shapes.
    print("\n1.2: Reshaping Tensors")
    print("-" * 40)
    
    # Reshape our sequence into a matrix
    # Original: 12 elements in 1D
    # Reshaped: same 12 elements arranged as 3x4 matrix
    reshaped = seq.reshape(3, 4)
    print(f"Reshaped to 3x4:\n{reshaped}")
    print(f"Shape: {reshaped.shape}")
    
    # Reshape into a 3D tensor
    # Same 12 elements, now arranged as 2x2x3
    reshaped_3d = seq.reshape(2, 2, 3)
    print(f"\nReshaped to 2x2x3:\n{reshaped_3d}")
    print(f"Shape: {reshaped_3d.shape}")
    
    # -------------------------------------------------------------------------
    # 1.3: Indexing Tensors
    # -------------------------------------------------------------------------
    # Indexing allows us to access specific elements, rows, or slices
    # This is crucial for extracting features, samples, or channels
    print("\n1.3: Indexing Tensors")
    print("-" * 40)
    
    matrix = torch.arange(12).reshape(3, 4)
    print(f"Matrix:\n{matrix}")
    
    # Access first row (all columns)
    first_row = matrix[0]
    print(f"\nFirst row: {first_row}")
    
    # Access specific element at row 1, column 2
    elem = matrix[1, 2]
    print(f"Element at [1,2]: {elem}")
    
    # 3D indexing
    cube = torch.arange(24).reshape(2, 3, 4)
    print(f"\n3D Cube (2x3x4):\n{cube}")
    
    # Get first 3x4 slice (like a single image from a batch)
    first_slice = cube[0]
    print(f"\nFirst slice:\n{first_slice}")
    
    # Get specific element in 3D space
    elem_3d = cube[1, 0, 2]
    print(f"Element at [1,0,2]: {elem_3d}")
    
    # -------------------------------------------------------------------------
    # 1.4: Slicing and Assignment
    # -------------------------------------------------------------------------
    # Slicing extracts sub-regions of tensors
    # Assignment modifies tensor values in-place
    print("\n1.4: Slicing and Assignment")
    print("-" * 40)
    
    data = torch.zeros(4, 4)
    print(f"Starting tensor:\n{data}")
    
    # Set entire second row to 5
    # data[1] selects row index 1 (second row)
    data[1] = 5
    print(f"\nAfter setting row 1 to 5:\n{data}")
    
    # Set entire third column to 7
    # data[:, 2] means "all rows, column 2"
    data[:, 2] = 7
    print(f"\nAfter setting column 2 to 7:\n{data}")
    
    # Set bottom-right 2x2 region to 9
    # data[2:, 2:] means "rows from 2 onward, columns from 2 onward"
    data[2:, 2:] = 9
    print(f"\nAfter setting bottom-right 2x2 to 9:\n{data}")


################################################################################
# SECTION 2: TENSOR OPERATIONS AND REDUCTIONS
################################################################################
#
# Matrix operations are the building blocks of neural networks.
# Understanding dot products, matrix multiplication, and reductions
# is essential for implementing and debugging deep learning models.
#
# Key operations:
# - Dot product: measures similarity between vectors
# - Matrix-vector multiplication: applies linear transformation
# - Matrix-matrix multiplication: composes linear transformations
################################################################################

def section_2_tensor_operations():
    """
    SECTION 2: Tensor Operations and Reductions
    
    Neural networks perform linear transformations using matrix multiplication.
    Understanding these operations is crucial:
    
    - Dot product: sum of element-wise products, used in attention mechanisms
    - Matrix-vector product: how a neural layer transforms input
    - Matrix-matrix product: how layers compose transformations
    """
    print("\n" + "=" * 80)
    print("SECTION 2: TENSOR OPERATIONS AND REDUCTIONS")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # 2.1: Creating Vectors and Matrices
    # -------------------------------------------------------------------------
    print("\n2.1: Creating Vectors and Matrices")
    print("-" * 40)
    
    v1 = torch.arange(4, dtype=torch.float32)
    v2 = torch.tensor([3., 2., 1., 0.])
    
    m1 = v1.reshape(2, 2)
    m2 = v2.reshape(2, 2)
    
    print(f"Vector 1: {v1}")
    print(f"Vector 2: {v2}")
    print(f"\nMatrix 1:\n{m1}")
    print(f"\nMatrix 2:\n{m2}")
    
    # -------------------------------------------------------------------------
    # 2.2: Dot Product
    # -------------------------------------------------------------------------
    # The dot product of two vectors is the sum of element-wise products:
    # v1 · v2 = v1[0]*v2[0] + v1[1]*v2[1] + ... + v1[n-1]*v2[n-1]
    #
    # Geometric interpretation: measures how aligned two vectors are
    # Used in: attention mechanisms, similarity metrics, neural activations
    print("\n2.2: Dot Product")
    print("-" * 40)
    
    dot = torch.dot(v1, v2)
    print(f"v1 · v2 = {dot}")
    print(f"Manual calculation: {v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2] + v1[3]*v2[3]}")
    
    # -------------------------------------------------------------------------
    # 2.3: Matrix-Vector Multiplication
    # -------------------------------------------------------------------------
    # This is how a neural network layer transforms input:
    # y = W x
    # where W is the weight matrix and x is the input vector
    #
    # Each output element is a dot product of a row of W with x
    print("\n2.3: Matrix-Vector Multiplication")
    print("-" * 40)
    
    v3 = torch.tensor([3., 2.])
    print(f"\nMatrix:\n{m1}")
    print(f"Vector: {v3}")
    
    # torch.mv = matrix-vector multiplication
    mv = torch.mv(m1, v3)
    print(f"\nResult (m1 @ v3): {mv}")
    print("This is like a neural layer: each output is a weighted sum of inputs")
    
    # -------------------------------------------------------------------------
    # 2.4: Matrix-Matrix Multiplication
    # -------------------------------------------------------------------------
    # Composing multiple linear transformations:
    # If y = W1 x and z = W2 y, then z = W2(W1 x) = (W2 W1) x
    #
    # This is how information flows through multiple neural network layers
    print("\n2.4: Matrix-Matrix Multiplication")
    print("-" * 40)
    
    print(f"\nMatrix 1:\n{m1}")
    print(f"\nMatrix 2:\n{m2}")
    
    # torch.mm = matrix-matrix multiplication
    # Can also use @ operator: m1 @ m2
    mm = torch.mm(m1, m2)
    print(f"\nResult (m1 @ m2):\n{mm}")
    print("This represents composing two neural network layers")


################################################################################
# SECTION 3: AUTOMATIC DIFFERENTIATION AND GRADIENT DESCENT
################################################################################
#
# Automatic differentiation (autograd) is PyTorch's core feature that enables
# training neural networks. It automatically computes gradients of any
# differentiable function, eliminating the need for manual derivative calculations.
#
# Gradient descent uses these gradients to iteratively update parameters
# to minimize a loss function, enabling the network to learn from data.
################################################################################

def section_3_autograd_and_gradient_descent():
    """
    SECTION 3: Automatic Differentiation and Gradient Descent
    
    Key concepts:
    - requires_grad=True: tells PyTorch to track operations for gradient computation
    - loss.backward(): computes gradients of loss with respect to all parameters
    - w.grad: stores the computed gradient
    - SGD update: w = w - learning_rate * gradient
    
    This is the foundation of training all neural networks.
    """
    print("\n" + "=" * 80)
    print("SECTION 3: AUTOMATIC DIFFERENTIATION AND GRADIENT DESCENT")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # 3.1: Simple Gradient Descent Step
    # -------------------------------------------------------------------------
    print("\n3.1: Single Gradient Descent Step")
    print("-" * 40)
    
    # Problem: learn weights w such that y = w · x approximates a target
    x = torch.tensor([1.0, 2.0])          # Input features
    y = torch.tensor(2.0)                  # Target output
    w = torch.tensor([2.0, 0.5], requires_grad=True)  # Learnable weights
    eta = 0.1                              # Learning rate
    
    # Forward pass: compute prediction
    # y_hat = w[0]*x[0] + w[1]*x[1]
    y_hat = torch.dot(w, x)
    
    # Compute loss: mean squared error
    # Loss measures how far our prediction is from the target
    loss = 0.5 * (y_hat - y) ** 2
    
    print(f"Initial w: {w.tolist()}")
    print(f"Prediction: {y_hat.item():.4f}")
    print(f"Target: {y.item():.4f}")
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass: compute gradients
    # This automatically computes dLoss/dw
    loss.backward()
    
    print(f"Gradient dL/dw: {w.grad.tolist()}")
    
    # Gradient descent update
    # Move weights in direction opposite to gradient (descent)
    with torch.no_grad():  # Don't track this operation
        w -= eta * w.grad
    w.grad.zero_()  # Clear gradients for next iteration
    
    print(f"Updated w: {w.tolist()}")
    
    # -------------------------------------------------------------------------
    # 3.2: Gradient Descent on a Dataset
    # -------------------------------------------------------------------------
    print("\n3.2: Gradient Descent on House Prices Dataset")
    print("-" * 40)
    
    # Load synthetic house data: (sqft, bedrooms) -> price
    x1, x2, y_data = load_simple_house_data()
    
    print(f"Dataset size: {len(y_data)} houses")
    print(f"Features: square feet and number of bedrooms")
    print(f"Target: house price")
    
    # Reset weights
    w = torch.tensor([2.0, 0.5], requires_grad=True)
    
    # Compute predictions for all houses
    # Broadcasting: w[0]*x1 multiplies weight by all sqft values
    y_hat = w[0] * x1 + w[1] * x2
    
    # Compute average loss across all samples
    # This is batch gradient descent (using entire dataset)
    loss = 0.5 * (y_hat - y_data) ** 2
    loss = loss.mean()
    loss_before = loss.item()
    
    print(f"Loss before update: {loss_before:.4f}")
    
    # Compute gradients and update
    loss.backward()
    with torch.no_grad():
        w -= eta * w.grad
    w.grad.zero_()
    
    # Recompute loss after update
    y_hat = w[0] * x1 + w[1] * x2
    loss = 0.5 * (y_hat - y_data) ** 2
    loss = loss.mean()
    loss_after = loss.item()
    
    print(f"Loss after update: {loss_after:.4f}")
    print(f"Improvement: {loss_before - loss_after:.4f}")
    print(f"Updated weights: {w.tolist()}")
    
    # -------------------------------------------------------------------------
    # 3.3: Training Loop
    # -------------------------------------------------------------------------
    print("\n3.3: Full Training Loop (100 steps)")
    print("-" * 40)
    
    # Reset weights
    w = torch.tensor([2.0, 0.5], requires_grad=True)
    
    # Train for multiple iterations
    # Each iteration uses the entire dataset (batch gradient descent)
    for step in range(100):
        # Forward pass
        y_hat = w[0] * x1 + w[1] * x2
        loss = 0.5 * (y_hat - y_data) ** 2
        loss = loss.mean()
        
        # Backward pass
        loss.backward()
        
        # Update weights
        with torch.no_grad():
            w -= eta * w.grad
        w.grad.zero_()
        
        # Print progress
        if step % 20 == 0:
            print(f"Step {step:3d}: loss = {loss.item():.6f}")
    
    print(f"\nFinal weights: {w.tolist()}")
    print("Weight interpretation:")
    print(f"  - Each 1000 sqft increases price by ${w[0].item()*100000:.0f}")
    print(f"  - Each bedroom increases price by ${w[1].item()*100000:.0f}")


################################################################################
# SECTION 4: LINEAR REGRESSION
################################################################################
#
# Linear regression models the relationship between features and a target
# using a linear function: y = w1*x1 + w2*x2 + ... + b
#
# Two approaches:
# 1. Closed-form solution: Directly compute optimal weights using linear algebra
# 2. Neural network: Learn weights using gradient descent
################################################################################

def section_4_linear_regression():
    """
    SECTION 4: Linear Regression
    
    Linear regression is the simplest supervised learning model.
    We'll explore two methods:
    
    1. Least Squares (closed-form): Solves (X^T X) w = X^T y
       - Exact solution in one step
       - Only works for linear models
       - Can be numerically unstable for large datasets
    
    2. Neural Network (gradient descent): Iteratively improves weights
       - Works for any differentiable model
       - Scales to large datasets
       - Foundation for deep learning
    """
    print("\n" + "=" * 80)
    print("SECTION 4: LINEAR REGRESSION")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # 4.1: Least Squares (Closed-Form Solution)
    # -------------------------------------------------------------------------
    print("\n4.1: Least Squares with Closed-Form Solution")
    print("-" * 40)
    
    # Load house price data
    sqft, bedrooms, price = load_house_price_data()
    n = len(price)
    
    print(f"Dataset: {n} houses")
    print(f"Features: square footage, number of bedrooms")
    
    # Build design matrix X with a bias column
    # X shape: (n, 3) where columns are [sqft, bedrooms, 1]
    # The column of 1s allows the model to learn a bias term
    col_ones = torch.ones(n)
    X = torch.stack([sqft, bedrooms, col_ones], dim=1)
    
    # Target vector y: house prices
    y = price.reshape(-1, 1)
    
    print(f"Design matrix X shape: {X.shape}")
    print(f"Target vector y shape: {y.shape}")
    
    # Closed-form solution: w = (X^T X)^(-1) X^T y
    # This minimizes the mean squared error in one step
    XtX = X.T @ X  # (3, 3) matrix
    Xty = X.T @ y  # (3, 1) vector
    
    # Solve the linear system: (X^T X) w = X^T y
    w = torch.linalg.solve(XtX, Xty)
    
    print(f"\nOptimal weights:")
    print(f"  Square feet coefficient: {w[0].item():.3f}")
    print(f"  Bedrooms coefficient: {w[1].item():.3f}")
    print(f"  Bias term: {w[2].item():.3f}")
    
    # Evaluate the model
    y_hat = X @ w
    mse_loss = torch.mean((y_hat - y) ** 2)
    print(f"\nMean Squared Error: {mse_loss.item():.4f}")
    
    # -------------------------------------------------------------------------
    # 4.2: Two-Layer Regression Network
    # -------------------------------------------------------------------------
    print("\n4.2: Two-Layer Neural Network for Regression")
    print("-" * 40)
    
    # Generate nonlinear data (sine wave with noise)
    X_sine, y_sine = make_sine_data()
    n, d = X_sine.shape
    hidden_size = 10
    
    print(f"Nonlinear dataset: {n} samples, {d} features")
    print("Target: noisy sine wave (linear regression would fail)")
    
    # Initialize network parameters
    # Layer 1: input -> hidden
    w1 = (torch.randn(d, hidden_size) * 0.1).requires_grad_(True)
    b1 = torch.zeros(hidden_size).requires_grad_(True)
    
    # Layer 2: hidden -> output
    w2 = (torch.randn(hidden_size, 1) * 0.1).requires_grad_(True)
    b2 = torch.zeros(1).requires_grad_(True)
    
    print(f"\nNetwork architecture:")
    print(f"  Input: {d} -> Hidden: {hidden_size} -> Output: 1")
    
    # Define forward pass
    def forward(X):
        # Layer 1: linear transformation + sigmoid activation
        h = torch.sigmoid(X @ w1 + b1)
        
        # Layer 2: linear transformation (no activation for regression)
        y_hat = h @ w2 + b2
        return y_hat
    
    # Train using Adam optimizer (adaptive learning rate)
    optimizer = torch.optim.Adam([w1, b1, w2, b2], lr=0.1)
    
    print("\nTraining...")
    for step in range(1000):
        y_hat = forward(X_sine)
        loss = ((y_hat - y_sine) ** 2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 200 == 0:
            print(f"  Step {step}: loss = {loss.item():.4f}")
    
    print(f"\nFinal MSE: {loss.item():.4f}")
    print("The two-layer network can fit nonlinear patterns!")


################################################################################
# SECTION 5: MULTI-LAYER PERCEPTRONS (MLPs)
################################################################################
#
# Multi-layer perceptrons are fully-connected neural networks with multiple
# hidden layers. They can approximate any continuous function (universal
# approximation theorem) given enough hidden units.
#
# PyTorch's nn.Module provides convenient building blocks for creating MLPs.
################################################################################

def section_5_multilayer_perceptrons():
    """
    SECTION 5: Multi-Layer Perceptrons
    
    Key components of an MLP:
    - nn.Linear: fully-connected layer (matrix multiplication + bias)
    - nn.ReLU: activation function (adds nonlinearity)
    - nn.Sequential: chains layers together
    
    Design choices:
    - Number of layers (depth): deeper networks can learn more complex patterns
    - Number of neurons per layer (width): wider layers have more capacity
    - Activation functions: ReLU is most common for hidden layers
    
    Trade-offs:
    - More parameters = more capacity but also more risk of overfitting
    - Deeper networks can be harder to train (vanishing gradients)
    """
    print("\n" + "=" * 80)
    print("SECTION 5: MULTI-LAYER PERCEPTRONS")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # 5.1: Load and Normalize Data
    # -------------------------------------------------------------------------
    print("\n5.1: California Housing Dataset")
    print("-" * 40)
    
    X, y = load_california_housing_data()
    
    print(f"Dataset: {X.shape[0]} houses")
    print(f"Features: {X.shape[1]} (median income, house age, rooms, etc.)")
    
    # Normalize features to mean 0, std 1
    # This helps gradient descent converge faster
    mean = X.mean(dim=0)
    std = X.std(dim=0)
    X = (X - mean) / std
    
    print(f"After normalization:")
    print(f"  Feature means: {X.mean(dim=0)[:3].tolist()} ...")
    print(f"  Feature stds: {X.std(dim=0)[:3].tolist()} ...")
    
    # -------------------------------------------------------------------------
    # 5.2: Build MLP with nn.Sequential
    # -------------------------------------------------------------------------
    print("\n5.2: Building MLPs with Different Architectures")
    print("-" * 40)
    
    # Architecture 1: 8 -> 128 -> 64 -> 1
    model1 = nn.Sequential(
        nn.Linear(8, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    
    print("Model 1 architecture: 8 -> 128 -> 64 -> 1")
    print(f"  Total parameters: {sum(p.numel() for p in model1.parameters()):,}")
    
    # Train model 1
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model1.parameters(), lr=0.01)
    
    print("  Training for 100 epochs...")
    for epoch in range(100):
        predictions = model1(X)
        loss = loss_fn(predictions, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        final_loss_1 = loss_fn(model1(X), y).item()
    print(f"  Final MSE: {final_loss_1:.4f}")
    
    # -------------------------------------------------------------------------
    # 5.3: Experiment with Depth and Width
    # -------------------------------------------------------------------------
    print("\n5.3: Comparing Different Architectures")
    print("-" * 40)
    
    # Architecture 2: Increasing width (8 -> 32 -> 64 -> 128 -> 1)
    model2 = nn.Sequential(
        nn.Linear(8, 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )
    
    print("Model 2 (increasing width): 8 -> 32 -> 64 -> 128 -> 1")
    train_and_evaluate_mlp(model2, X, y, lr=0.01, epochs=100)
    
    # Architecture 3: Large pyramid (8 -> 256 -> 128 -> 64 -> 32 -> 1)
    model3 = nn.Sequential(
        nn.Linear(8, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    
    print("Model 3 (large pyramid): 8 -> 256 -> 128 -> 64 -> 32 -> 1")
    train_and_evaluate_mlp(model3, X, y, lr=0.01, epochs=100)
    
    print("\nKey observations:")
    print("  - Deeper networks can learn more complex patterns")
    print("  - Too many parameters may lead to overfitting")
    print("  - Normalization is crucial for training deep networks")


################################################################################
# SECTION 6: CLASSIFICATION AND REGULARIZATION
################################################################################
#
# Classification tasks predict discrete categories rather than continuous values.
# Key differences from regression:
# - Output: class probabilities (via softmax)
# - Loss: cross-entropy instead of MSE
# - Evaluation: accuracy instead of MSE
#
# Regularization techniques prevent overfitting:
# - Dropout: randomly zero out neurons during training
# - L2 regularization: penalize large weights
# - Early stopping: stop before overfitting begins
################################################################################

def section_6_classification():
    """
    SECTION 6: Classification and Regularization
    
    Classification concepts:
    - Logits: raw output scores (before softmax)
    - Softmax: converts logits to probabilities
    - Cross-entropy loss: measures prediction quality for classification
    - Accuracy: percentage of correct predictions
    
    Overfitting happens when:
    - Model has too many parameters relative to data size
    - Training too long without regularization
    - Symptoms: perfect training accuracy, poor test accuracy
    
    Regularization techniques:
    - Dropout: randomly disable neurons to prevent co-adaptation
    - Weight decay (L2): penalize large weights
    - Data augmentation: artificially increase dataset size
    """
    print("\n" + "=" * 80)
    print("SECTION 6: CLASSIFICATION AND REGULARIZATION")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # 6.1: Load and Prepare Classification Dataset
    # -------------------------------------------------------------------------
    print("\n6.1: Forest Cover Type Classification")
    print("-" * 40)
    
    X_train, y_train, X_test, y_test = load_forest_data()
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]} (elevation, slope, soil type, etc.)")
    print(f"Classes: {len(torch.unique(y_train))} forest cover types")
    
    # -------------------------------------------------------------------------
    # 6.2: Build and Train a Classifier
    # -------------------------------------------------------------------------
    print("\n6.2: Training a Basic Classifier")
    print("-" * 40)
    
    # Simple one hidden layer MLP
    model = nn.Sequential(
        nn.Linear(54, 128),
        nn.ReLU(),
        nn.Linear(128, 7)  # 7 output classes
    )
    
    print("Model architecture: 54 -> 128 -> 7")
    
    # Cross-entropy loss for classification
    # Combines softmax and negative log-likelihood
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Training for 200 epochs...")
    for epoch in range(200):
        # Forward pass
        predictions = model(X_train)
        loss = loss_fn(predictions, y_train)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                test_acc = (model(X_test).argmax(dim=1) == y_test).float().mean()
            print(f"  Epoch {epoch+1}: loss = {loss.item():.4f}, "
                  f"test acc = {test_acc:.2%}")
    
    # Final evaluation
    with torch.no_grad():
        train_acc = (model(X_train).argmax(dim=1) == y_train).float().mean()
        test_acc = (model(X_test).argmax(dim=1) == y_test).float().mean()
    
    print(f"\nFinal Results:")
    print(f"  Training accuracy: {train_acc:.2%}")
    print(f"  Test accuracy: {test_acc:.2%}")
    
    # -------------------------------------------------------------------------
    # 6.3: Demonstrating Overfitting
    # -------------------------------------------------------------------------
    print("\n6.3: Overfitting with a Large Model")
    print("-" * 40)
    
    # Much larger model (prone to overfitting)
    big_model = nn.Sequential(
        nn.Linear(54, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 7)
    )
    
    print("Large model: 54 -> 256 -> 256 -> 256 -> 7")
    print(f"Parameters: {sum(p.numel() for p in big_model.parameters()):,}")
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(big_model.parameters(), lr=0.001)
    
    train_losses = []
    test_losses = []
    
    print("Training for 300 epochs (watch for overfitting)...")
    for epoch in range(300):
        # Training
        predictions = big_model(X_train)
        loss = loss_fn(predictions, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Test evaluation
        with torch.no_grad():
            test_loss = loss_fn(big_model(X_test), y_test)
            test_losses.append(test_loss.item())
        
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                train_acc = (big_model(X_train).argmax(dim=1) == y_train).float().mean()
                test_acc = (big_model(X_test).argmax(dim=1) == y_test).float().mean()
            print(f"  Epoch {epoch+1}: train_loss = {loss.item():.4f}, "
                  f"test_loss = {test_loss.item():.4f}")
            print(f"    train_acc = {train_acc:.2%}, test_acc = {test_acc:.2%}")
    
    print("\nObservation: Training loss decreases but test loss may increase")
    print("This indicates overfitting!")
    
    # -------------------------------------------------------------------------
    # 6.4: Using Dropout for Regularization
    # -------------------------------------------------------------------------
    print("\n6.4: Preventing Overfitting with Dropout")
    print("-" * 40)
    
    # Same large model but with dropout
    dropout_model = nn.Sequential(
        nn.Linear(54, 256),
        nn.ReLU(),
        nn.Dropout(0.5),  # Randomly zero 50% of neurons
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 7)
    )
    
    print("Model with 50% dropout after each hidden layer")
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(dropout_model.parameters(), lr=0.001)
    
    print("Training for 300 epochs...")
    for epoch in range(300):
        dropout_model.train()  # Enable dropout
        predictions = dropout_model(X_train)
        loss = loss_fn(predictions, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            dropout_model.eval()  # Disable dropout for evaluation
            with torch.no_grad():
                train_acc = (dropout_model(X_train).argmax(dim=1) == y_train).float().mean()
                test_acc = (dropout_model(X_test).argmax(dim=1) == y_test).float().mean()
            print(f"  Epoch {epoch+1}: train_acc = {train_acc:.2%}, "
                  f"test_acc = {test_acc:.2%}")
    
    print("\nDropout helps reduce gap between train and test performance!")


################################################################################
# SECTION 7: CONVOLUTIONS FOR IMAGES
################################################################################
#
# Convolutional layers are fundamental to processing images. Unlike fully-
# connected layers, convolutions:
# - Preserve spatial structure
# - Share weights across locations (translation invariance)
# - Have far fewer parameters
#
# A convolution slides a small filter (kernel) across the image, computing
# dot products at each position to detect local patterns.
################################################################################

def section_7_convolutions():
    """
    SECTION 7: Convolutions for Images
    
    Convolution operation:
    - Input: image (H x W)
    - Kernel: small filter (K x K), typically 3x3 or 5x5
    - Output: feature map (H-K+1 x W-K+1)
    
    How it works:
    1. Place kernel at top-left of image
    2. Compute element-wise product and sum (dot product)
    3. This gives one output value
    4. Slide kernel right by 1 pixel and repeat
    5. When row is done, move to next row
    
    Different kernels detect different features:
    - Edge detectors: highlight boundaries
    - Blur filters: smooth the image
    - Sharpening filters: enhance details
    """
    print("\n" + "=" * 80)
    print("SECTION 7: CONVOLUTIONS FOR IMAGES")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # 7.1: Load MNIST Dataset
    # -------------------------------------------------------------------------
    print("\n7.1: Loading MNIST Handwritten Digits")
    print("-" * 40)
    
    # Load a few MNIST digits
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(root="./data", train=True, download=True, 
                          transform=transform)
    
    print(f"MNIST dataset: {len(mnist)} training images")
    print("Image size: 28x28 pixels, grayscale")
    
    # Get first image
    image, label = mnist[0]
    image = image.squeeze(0)  # Remove channel dimension
    
    print(f"\nSample image:")
    print(f"  Label: {label}")
    print(f"  Shape: {tuple(image.shape)}")
    print(f"  Value range: [{image.min():.2f}, {image.max():.2f}]")
    
    # -------------------------------------------------------------------------
    # 7.2: Implement Convolution from Scratch
    # -------------------------------------------------------------------------
    print("\n7.2: Implementing 2D Convolution")
    print("-" * 40)
    
    def my_conv2d(image, kernel):
        """
        Perform 2D convolution using nested loops.
        
        Args:
            image: (H, W) tensor
            kernel: (K, K) tensor
        
        Returns:
            output: (H-K+1, W-K+1) tensor
        """
        H, W = image.shape
        K = kernel.shape[0]
        out_h = H - K + 1
        out_w = W - K + 1
        out = torch.zeros((out_h, out_w), dtype=image.dtype)
        
        # Slide kernel across image
        for i in range(out_h):
            for j in range(out_w):
                # Extract image patch
                patch = image[i:i+K, j:j+K]
                # Compute dot product with kernel
                out[i, j] = (patch * kernel).sum()
        
        return out
    
    # Test with a simple kernel
    kernel = torch.tensor([[0.2341, 0.5123, 0.9812],
                          [0.1011, 0.7765, 0.3341],
                          [0.6543, 0.0987, 0.4432]])
    
    print("Testing convolution implementation...")
    my_out = my_conv2d(image, kernel)
    
    # Compare with PyTorch's conv2d
    torch_out = F.conv2d(
        image.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
        kernel.unsqueeze(0).unsqueeze(0),
    ).squeeze()
    
    print(f"  Input shape: {tuple(image.shape)}")
    print(f"  Kernel shape: {tuple(kernel.shape)}")
    print(f"  Output shape: {tuple(my_out.shape)}")
    print(f"  Max difference from PyTorch: {(my_out - torch_out).abs().max().item():.6f}")
    
    # -------------------------------------------------------------------------
    # 7.3: Exploring Different Kernels
    # -------------------------------------------------------------------------
    print("\n7.3: Different Kernels Detect Different Features")
    print("-" * 40)
    
    # Define various kernels
    kernels = {
        "Vertical edges": torch.tensor([[-1., 0., 1.],
                                       [-1., 0., 1.],
                                       [-1., 0., 1.]]),
        
        "Horizontal edges": torch.tensor([[-1., -1., -1.],
                                          [0., 0., 0.],
                                          [1., 1., 1.]]),
        
        "Box blur": torch.ones(3, 3) / 9,
        
        "Gaussian blur": torch.tensor([[1., 2., 1.],
                                       [2., 4., 2.],
                                       [1., 2., 1.]]) / 16,
        
        "Sharpen": torch.tensor([[0., -1., 0.],
                                [-1., 5., -1.],
                                [0., -1., 0.]]),
        
        "45° edges": torch.tensor([[0., 1., 2.],
                                   [-1., 0., 1.],
                                   [-2., -1., 0.]]),
    }
    
    print("Applying different kernels to detect various features:")
    for name, kernel in kernels.items():
        output = my_conv2d(image, kernel)
        print(f"\n  {name}:")
        print(f"    Output range: [{output.min():.2f}, {output.max():.2f}]")
        print(f"    Output mean: {output.mean():.2f}")
    
    print("\nKey insights:")
    print("  - Vertical edge kernel responds strongly to vertical boundaries")
    print("  - Blur kernels smooth out details")
    print("  - Sharpen kernel enhances edges")
    print("  - CNNs learn these kernels automatically from data!")


################################################################################
# SECTION 8: CONVOLUTIONAL NEURAL NETWORKS (CNNs)
################################################################################
#
# CNNs stack convolutional layers with pooling to build hierarchical
# representations:
# - Early layers: detect simple features (edges, corners)
# - Middle layers: combine into parts (eyes, wheels)
# - Late layers: recognize whole objects (faces, cars)
#
# Pooling (max or average) reduces spatial dimensions while keeping
# important features, providing translation invariance.
################################################################################

def section_8_cnns():
    """
    SECTION 8: Convolutional Neural Networks
    
    CNN architecture pattern:
    1. Convolution: detect local features
    2. Activation (ReLU): add nonlinearity
    3. Pooling: downsample, reduce parameters
    4. Repeat: build deeper representations
    5. Flatten: convert to vector
    6. Fully-connected: final classification
    
    LeNet-5 architecture (classic CNN for digits):
    - Input: 28x28 grayscale image
    - Conv1: 1 -> 6 channels, 5x5 kernel -> 28x28x6
    - Pool1: 2x2 max pooling -> 14x14x6
    - Conv2: 6 -> 16 channels, 5x5 kernel -> 14x14x16
    - Pool2: 2x2 max pooling -> 7x7x16
    - Flatten: 7x7x16 = 784 features
    - FC1: 784 -> 120
    - FC2: 120 -> 84
    - FC3: 84 -> 10 (output classes)
    """
    print("\n" + "=" * 80)
    print("SECTION 8: CONVOLUTIONAL NEURAL NETWORKS")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # 8.1: Setting Up MNIST Data Loaders
    # -------------------------------------------------------------------------
    print("\n8.1: Preparing MNIST Dataset")
    print("-" * 40)
    
    transform = transforms.ToTensor()
    full_dataset = datasets.MNIST(root="./data", train=True, download=True, 
                                 transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, 
                                 transform=transform)
    
    # Split into train and validation
    train_dataset, val_dataset = random_split(full_dataset, [50000, 10000])
    
    # Create data loaders for batching
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: 64")
    
    # -------------------------------------------------------------------------
    # 8.2: Building LeNet-5 CNN
    # -------------------------------------------------------------------------
    print("\n8.2: LeNet-5 Architecture")
    print("-" * 40)
    
    model = nn.Sequential(
        # First convolutional block
        nn.Conv2d(1, 6, kernel_size=5, padding=2),  # 28x28x1 -> 28x28x6
        nn.ReLU(),
        nn.MaxPool2d(2),  # 28x28x6 -> 14x14x6
        
        # Second convolutional block
        nn.Conv2d(6, 16, kernel_size=5, padding=2),  # 14x14x6 -> 14x14x16
        nn.ReLU(),
        nn.MaxPool2d(2),  # 14x14x16 -> 7x7x16
        
        # Flatten and fully-connected layers
        nn.Flatten(),  # 7x7x16 = 784
        nn.Linear(784, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10)  # 10 digit classes
    )
    
    print("LeNet-5 architecture:")
    print("  Conv1: 1->6 channels, 5x5 kernel, padding=2")
    print("  ReLU + MaxPool(2x2)")
    print("  Conv2: 6->16 channels, 5x5 kernel, padding=2")
    print("  ReLU + MaxPool(2x2)")
    print("  Flatten -> FC(784->120) -> FC(120->84) -> FC(84->10)")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # -------------------------------------------------------------------------
    # 8.3: Training the CNN
    # -------------------------------------------------------------------------
    print("\n8.3: Training LeNet-5 on MNIST")
    print("-" * 40)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    def train_epoch(model, loader, criterion, optimizer):
        """Train for one epoch."""
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for images, labels in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
        
        return total_loss / total, correct / total
    
    def evaluate(model, loader, criterion):
        """Evaluate on validation/test set."""
        model.eval()
        total_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for images, labels in loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += images.size(0)
        
        return total_loss / total, correct / total
    
    print("Training for 10 epochs...")
    start_time = time.time()
    
    for epoch in range(10):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        print(f"Epoch {epoch+1}/10:")
        print(f"  Train: loss = {train_loss:.4f}, acc = {train_acc:.2%}")
        print(f"  Val:   loss = {val_loss:.4f}, acc = {val_acc:.2%}")
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f} seconds")
    
    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\nFinal Test Accuracy: {test_acc:.2%}")
    
    print("\nKey advantages of CNNs:")
    print("  - Far fewer parameters than fully-connected networks")
    print("  - Translation invariant (recognizes patterns anywhere in image)")
    print("  - Hierarchical feature learning (edges -> parts -> objects)")
    print("  - State-of-the-art for image tasks")


################################################################################
# SECTION 9: DATA AUGMENTATION
################################################################################
#
# Data augmentation artificially increases dataset size by applying
# random transformations to training images. This:
# - Reduces overfitting by exposing model to more variations
# - Improves generalization to new data
# - Makes model more robust to transformations
#
# Common augmentations:
# - Geometric: flips, rotations, scaling, cropping
# - Color: brightness, contrast, saturation, hue
# - Noise: blur, gaussian noise, cutout
################################################################################

def section_9_data_augmentation():
    """
    SECTION 9: Data Augmentation Techniques
    
    Data augmentation applies random transformations during training:
    - Each epoch, images look slightly different
    - Model must learn features robust to these variations
    - Effectively multiplies dataset size
    
    Common transformations:
    - RandomHorizontalFlip: mirrors image left-right
    - RandomRotation: rotates by random angle
    - ColorJitter: changes brightness, contrast, saturation, hue
    - RandomAffine: combines translation, rotation, scale, shear
    - RandomCrop: extracts random sub-region
    
    Important: Only augment training data, not validation/test!
    """
    print("\n" + "=" * 80)
    print("SECTION 9: DATA AUGMENTATION")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # 9.1: Loading CIFAR-10 Dataset
    # -------------------------------------------------------------------------
    print("\n9.1: CIFAR-10 Natural Images")
    print("-" * 40)
    
    # CIFAR-10: 60k 32x32 color images in 10 classes
    CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Load without augmentation first
    basic_transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(root="./data", train=True, download=True,
                              transform=basic_transform)
    
    print(f"CIFAR-10 dataset: {len(dataset)} training images")
    print(f"Image size: 32x32 pixels, RGB (3 channels)")
    print(f"Classes: {', '.join(CLASSES)}")
    
    # Get a batch of images
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    images, labels = next(iter(loader))
    
    print(f"\nSample batch:")
    print(f"  Images shape: {tuple(images.shape)}")  # (16, 3, 32, 32)
    print(f"  Labels: {[CLASSES[l] for l in labels[:8]]}")
    
    # -------------------------------------------------------------------------
    # 9.2: Manual Augmentation Functions
    # -------------------------------------------------------------------------
    print("\n9.2: Implementing Simple Augmentations")
    print("-" * 40)
    
    def hflip(img):
        """Horizontal flip: mirror image left-right."""
        return torch.flip(img, dims=[2])  # Flip width dimension
    
    def pixel_jitter(img, sigma=0.1):
        """Add random Gaussian noise to each pixel."""
        noise = torch.randn_like(img) * sigma
        return (img + noise).clamp(0, 1)
    
    print("Manual augmentation functions:")
    print("  - hflip: mirrors image horizontally")
    print("  - pixel_jitter: adds Gaussian noise")
    
    # Apply augmentations
    sample_img = images[0]
    flipped = hflip(sample_img)
    jittered = pixel_jitter(sample_img, sigma=0.1)
    
    print(f"\nOriginal image shape: {tuple(sample_img.shape)}")
    print(f"Flipped image shape: {tuple(flipped.shape)}")
    print(f"Jittered image range: [{jittered.min():.2f}, {jittered.max():.2f}]")
    
    # -------------------------------------------------------------------------
    # 9.3: Using torchvision Transforms
    # -------------------------------------------------------------------------
    print("\n9.3: Comprehensive Augmentation Pipeline")
    print("-" * 40)
    
    # Define augmentation pipeline
    train_transform = transforms.Compose([
        # Color augmentation
        transforms.ColorJitter(
            brightness=0.5,  # Vary brightness by +/- 50%
            contrast=0.5,    # Vary contrast by +/- 50%
            saturation=0.5,  # Vary saturation by +/- 50%
            hue=0.5          # Vary hue by +/- 50%
        ),
        
        # Geometric augmentation
        transforms.RandomAffine(
            degrees=10,           # Rotate by +/- 10 degrees
            translate=(0.1, 0.1), # Translate by up to 10% of image size
            scale=(0.9, 1.1),     # Scale by 90% to 110%
            shear=10              # Shear by +/- 10 degrees
        ),
        
        # Random horizontal flip
        transforms.RandomHorizontalFlip(p=0.5),
        
        # Convert to tensor
        transforms.ToTensor()
    ])
    
    print("Augmentation pipeline:")
    print("  1. ColorJitter: vary brightness, contrast, saturation, hue")
    print("  2. RandomAffine: rotation, translation, scaling, shearing")
    print("  3. RandomHorizontalFlip: 50% chance to flip")
    print("  4. ToTensor: convert to PyTorch tensor")
    
    # Load augmented dataset
    aug_dataset = datasets.CIFAR10(root="./data", train=True, download=True,
                                  transform=train_transform)
    aug_loader = DataLoader(aug_dataset, batch_size=16, shuffle=True)
    aug_images, aug_labels = next(iter(aug_loader))
    
    print(f"\nAugmented batch shape: {tuple(aug_images.shape)}")
    print("Each training epoch will see different augmented versions!")
    
    # -------------------------------------------------------------------------
    # 9.4: Impact of Data Augmentation
    # -------------------------------------------------------------------------
    print("\n9.4: Why Data Augmentation Works")
    print("-" * 40)
    
    print("Benefits:")
    print("  - Reduces overfitting: model can't memorize specific images")
    print("  - Improves generalization: learns features robust to variations")
    print("  - Increases effective dataset size: each epoch has 'new' data")
    print("  - Makes model invariant to transformations")
    
    print("\nBest practices:")
    print("  - Use realistic augmentations (don't upside-down cars)")
    print("  - More augmentation when data is limited")
    print("  - Only augment training set, not validation/test")
    print("  - Tune augmentation strength as hyperparameter")
    
    print("\nCommon augmentations by domain:")
    print("  Natural images: flips, crops, color jitter, rotation")
    print("  Medical images: rotation, zoom, elastic deformation")
    print("  Text: synonym replacement, back-translation")
    print("  Audio: time stretch, pitch shift, noise injection")


################################################################################
# HELPER FUNCTIONS
################################################################################

def load_simple_house_data():
    """Load synthetic house price data."""
    # Synthetic data: sqft (in 1000s), bedrooms, price (in $100k)
    data = [
        (1.0, 2, 2.0),
        (1.2, 2, 2.3),
        (1.5, 3, 2.85),
        (1.8, 3, 3.4),
        (2.0, 4, 4.0),
        (2.2, 4, 4.5),
        (2.5, 5, 5.2),
        (0.9, 1, 1.7),
    ]
    sqft = torch.tensor([d[0] for d in data])
    bedrooms = torch.tensor([float(d[1]) for d in data])
    price = torch.tensor([d[2] for d in data])
    return sqft, bedrooms, price


def load_house_price_data():
    """Load house price data for least squares."""
    data = [
        (1.0, 2, 2.0),
        (1.2, 2, 2.3),
        (1.5, 3, 2.85),
        (1.8, 3, 3.4),
        (2.0, 4, 4.0),
        (2.2, 4, 4.5),
        (2.5, 5, 5.2),
        (0.9, 1, 1.7),
    ]
    sqft = torch.tensor([d[0] for d in data])
    bedrooms = torch.tensor([float(d[1]) for d in data])
    price = torch.tensor([d[2] for d in data])
    return sqft, bedrooms, price


def make_sine_data(n_samples=50):
    """Generate noisy sine wave data for nonlinear regression."""
    torch.manual_seed(42)
    x = torch.linspace(-3, 3, n_samples).unsqueeze(1)
    y = torch.sin(x) + 0.5 * torch.sin(2 * x) + torch.randn(n_samples, 1) * 0.1
    return x, y


def load_california_housing_data():
    """Load California Housing dataset as float32 tensors."""
    data = fetch_california_housing()
    X = torch.tensor(data.data, dtype=torch.float32)
    y = torch.tensor(data.target, dtype=torch.float32).unsqueeze(1)
    return X, y


def load_forest_data(test_fraction=0.25, n_samples=5000):
    """Load forest cover type dataset, normalize, and split."""
    covtype = fetch_covtype()
    
    # Use subset for faster training
    np.random.seed(42)
    indices = np.random.choice(len(covtype.data), n_samples, replace=False)
    X = covtype.data[indices]
    y = covtype.target[indices] - 1  # Convert to 0-indexed
    
    # Shuffle and split
    perm = np.random.RandomState(42).permutation(len(X))
    X, y = X[perm], y[perm]
    n_test = int(len(X) * test_fraction)
    X_train, X_test = X[:-n_test], X[-n_test:]
    y_train, y_test = y[:-n_test], y[-n_test:]
    
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    return X_train, y_train, X_test, y_test


def train_and_evaluate_mlp(model, X, y, lr=0.01, epochs=100):
    """Helper function to train and evaluate an MLP."""
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        predictions = model(X)
        loss = loss_fn(predictions, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        final_loss = loss_fn(model(X), y).item()
    
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Final MSE: {final_loss:.4f}")


################################################################################
# MAIN EXECUTION
################################################################################

def main():
    """
    Run all sections of the deep learning tutorial.
    
    This comprehensive tutorial covers:
    1. Tensor operations and manipulations
    2. Matrix operations and reductions
    3. Automatic differentiation and gradient descent
    4. Linear regression (closed-form and neural networks)
    5. Multi-layer perceptrons
    6. Classification and regularization
    7. Convolutions for images
    8. Convolutional neural networks
    9. Data augmentation
    
    Each section builds on previous concepts to provide a complete
    understanding of modern deep learning with PyTorch.
    """
    print("\n" + "=" * 80)
    print("COMPLETE DEEP LEARNING TUTORIAL")
    print("=" * 80)
    print("\nThis tutorial covers foundational deep learning concepts from")
    print("basic tensor operations through convolutional neural networks.")
    print("\nNote: Some sections download datasets and may take time to run.")
    print("=" * 80)
    
    # Run all sections
    section_1_tensor_basics()
    section_2_tensor_operations()
    section_3_autograd_and_gradient_descent()
    section_4_linear_regression()
    section_5_multilayer_perceptrons()
    section_6_classification()
    section_7_convolutions()
    section_8_cnns()
    section_9_data_augmentation()
    
    # Final summary
    print("\n" + "=" * 80)
    print("TUTORIAL COMPLETE!")
    print("=" * 80)
    print("\nYou've learned:")
    print("  ✓ Tensor operations and manipulations")
    print("  ✓ Automatic differentiation and gradient descent")
    print("  ✓ Linear and nonlinear regression")
    print("  ✓ Multi-layer perceptrons")
    print("  ✓ Classification and regularization techniques")
    print("  ✓ Convolutional operations")
    print("  ✓ CNN architectures (LeNet-5)")
    print("  ✓ Data augmentation strategies")
    print("\nNext steps:")
    print("  - Experiment with different architectures")
    print("  - Try other datasets (ImageNet, COCO, etc.)")
    print("  - Learn advanced topics (attention, transformers, GANs)")
    print("  - Build your own deep learning projects!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Run the complete tutorial
    main()
