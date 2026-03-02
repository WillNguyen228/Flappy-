"""
Quick Test Script for Gesture Recognition System

This script tests all components to ensure they're working correctly
before you start collecting data and training.
"""

import sys
import torch
import cv2
import pygame


def test_pytorch():
    """Test PyTorch installation and GPU availability"""
    print("=" * 60)
    print("Testing PyTorch...")
    print("=" * 60)
    
    try:
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  Using CPU (training will be slower)")
        
        # Test tensor operations
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = x + y
        print("✓ Tensor operations working")
        
        return True
    except Exception as e:
        print(f"✗ PyTorch error: {e}")
        return False


def test_opencv():
    """Test OpenCV and webcam"""
    print("\n" + "=" * 60)
    print("Testing OpenCV and Webcam...")
    print("=" * 60)
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
        
        # Test webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ Cannot open webcam")
            print("  Check if webcam is connected and not in use")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("✗ Cannot read from webcam")
            cap.release()
            return False
        
        print(f"✓ Webcam working (resolution: {frame.shape[1]}x{frame.shape[0]})")
        cap.release()
        
        return True
    except Exception as e:
        print(f"✗ OpenCV error: {e}")
        return False


def test_pygame():
    """Test Pygame"""
    print("\n" + "=" * 60)
    print("Testing Pygame...")
    print("=" * 60)
    
    try:
        import pygame
        print(f"✓ Pygame version: {pygame.version.ver}")
        
        pygame.init()
        print("✓ Pygame initialized")
        pygame.quit()
        
        return True
    except Exception as e:
        print(f"✗ Pygame error: {e}")
        return False


def test_model_architecture():
    """Test that the model architecture can be created"""
    print("\n" + "=" * 60)
    print("Testing Model Architecture...")
    print("=" * 60)
    
    try:
        from gesture_model import GestureRecognitionCNN, count_parameters
        
        model = GestureRecognitionCNN(num_classes=2)
        params = count_parameters(model)
        
        print(f"✓ Model created successfully")
        print(f"✓ Total parameters: {params:,}")
        
        # Test forward pass
        test_input = torch.randn(1, 1, 64, 64)
        output = model(test_input)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model error: {e}")
        return False


def test_game_assets():
    """Test that game assets exist"""
    print("\n" + "=" * 60)
    print("Testing Game Assets...")
    print("=" * 60)
    
    import os
    
    required_assets = [
        'img/bg.png',
        'img/ground.png',
        'img/bird1.png',
        'img/bird2.png',
        'img/bird3.png',
        'img/pipe.png',
        'img/restart.png'
    ]
    
    missing = []
    for asset in required_assets:
        if os.path.exists(asset):
            print(f"✓ {asset}")
        else:
            print(f"✗ {asset} - MISSING")
            missing.append(asset)
    
    if missing:
        print(f"\n✗ Missing {len(missing)} assets")
        print("  The game may not work properly")
        return False
    else:
        print("\n✓ All assets found")
        return True


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("GESTURE RECOGNITION SYSTEM - SYSTEM CHECK")
    print("=" * 60)
    print("\nThis script will test all components of the system.\n")
    
    results = {
        "PyTorch": test_pytorch(),
        "OpenCV": test_opencv(),
        "Pygame": test_pygame(),
        "Model": test_model_architecture(),
        "Assets": test_game_assets()
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for component, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{component:15s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nYou're ready to:")
        print("  1. Collect training data: python collect_gesture_data.py")
        print("  2. Train the model:       python train_gesture_model.py")
        print("  3. Play the game:         python gesture_flappy_bird.py")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease fix the issues above before continuing.")
        print("See GESTURE_README.md for troubleshooting.")
    print("=" * 60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
