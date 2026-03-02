# 🎮 Gesture-Controlled Flappy Bird with Deep Learning

A real-time hand gesture recognition system that controls Flappy Bird using a Convolutional Neural Network (CNN). This project applies deep learning principles from `complete_deep_learning_tutorial.py` to create an interactive gaming experience.

## 🎯 Project Overview

This project implements:
- **CNN-based hand gesture recognition** (fist vs. peace sign)
- **Real-time webcam processing**
- **Integration with Flappy Bird game**
- Deep learning principles including:
  - Convolutional layers for feature detection
  - Data augmentation for robustness
  - Training with gradient descent
  - Model evaluation and validation

## 🎬 How It Works

1. **Fist** 🤛: No action
2. **Peace Sign** ✌️: Makes the bird jump!

The CNN model processes webcam frames in real-time, detecting your hand gesture and controlling the game accordingly.

## 📁 Project Structure

```
Flappy Bird/
├── gesture_model.py              # CNN model architecture
├── collect_gesture_data.py       # Data collection script
├── train_gesture_model.py        # Training script
├── flappy_bird_gesture.py        # Modified Flappy Bird game
├── gesture_flappy_bird.py        # Main application
├── complete_deep_learning_tutorial.py  # Deep learning reference
├── flappy_bird.py                # Original game
└── img/                          # Game assets
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision opencv-python pygame numpy matplotlib scikit-learn
```

### Step 1: Collect Training Data

Run the data collection script to capture images of your hand gestures:

```bash
python collect_gesture_data.py
```

**Instructions:**
1. Position your hand in the green square
2. Press `0` to capture **fist** images (~100-200 images)
3. Press `1` to capture **peace sign** images (~100-200 images)
4. Vary your hand position, angle, and lighting
5. Press `q` when finished

**Tips:**
- Capture from different angles
- Try different lighting conditions
- Move your hand around the frame
- Use both hands if desired

### Step 2: Train the Model

Train the CNN on your collected data:

```bash
python train_gesture_model.py
```

This will:
- Load your gesture images
- Split into training/validation sets (80/20)
- Apply data augmentation
- Train for 20 epochs
- Save the best model as `gesture_model.pth`
- Generate training history plots

**Expected Results:**
- Training takes 2-5 minutes on CPU
- Should achieve >90% validation accuracy
- Lower accuracy? Collect more diverse training data

### Step 3: Play the Game!

Run the gesture-controlled Flappy Bird:

```bash
python gesture_flappy_bird.py
```

**Controls:**
- Show **peace sign** ✌️ to jump
- Keep **fist** 🤛 to fall
- Or use mouse/UP arrow as backup

## Deep Learning Architecture

### Model: GestureRecognitionCNN

Based on LeNet-5 architecture with modifications for gesture recognition:

```
Input: 64x64 grayscale image
    ↓
Conv1: 1→16 channels, 5x5 kernel + ReLU + MaxPool
    ↓
Conv2: 16→32 channels, 5x5 kernel + ReLU + MaxPool
    ↓
Conv3: 32→64 channels, 3x3 kernel + ReLU + MaxPool
    ↓
Flatten: 8×8×64 = 4,096 features
    ↓
FC1: 4096→256 + ReLU + Dropout(0.5)
    ↓
FC2: 256→128 + ReLU + Dropout(0.5)
    ↓
FC3: 128→2 (fist, peace)
    ↓
Output: Class probabilities
```

**Total Parameters:** ~1.1M trainable parameters

### Key Deep Learning Concepts Applied

1. **Convolutional Layers** (Section 7-8 of tutorial)
   - Detect local features (edges, patterns)
   - Translation invariant
   - Parameter sharing reduces model size

2. **Pooling** (Section 8)
   - Downsamples spatial dimensions
   - Provides translation invariance
   - Reduces overfitting

3. **Data Augmentation** (Section 9)
   - Random flips, rotations, affine transforms
   - Increases effective dataset size
   - Improves generalization

4. **Regularization**
   - Dropout layers (50%)
   - Prevents overfitting
   - Improves test performance

5. **Training Process** (Section 3-6)
   - CrossEntropyLoss for classification
   - Adam optimizer (adaptive learning rate)
   - Learning rate scheduling
   - Train/validation split for model selection

## 📊 Performance Tips

### If Model Accuracy is Low:

1. **Collect more data**
   - Aim for 200+ images per gesture
   - Ensure diverse poses and lighting

2. **Check data quality**
   - Images should be clear
   - Hand should fill most of the frame
   - Consistent lighting helps

3. **Adjust training parameters**
   - Increase epochs: `epochs=30`
   - Adjust learning rate: `learning_rate=0.0005`
   - Modify batch size: `batch_size=16`

4. **Data augmentation**
   - Already implemented in training script
   - Helps with varying conditions

### For Better Real-Time Performance:

1. **Good lighting** - Consistent lighting improves recognition
2. **Clear background** - Avoid cluttered backgrounds
3. **Center your hand** - Keep hand in the green square
4. **Steady movements** - Prediction smoothing is built-in

## 🎓 Educational Value

This project demonstrates:

- **Computer Vision**: Image preprocessing, ROI extraction, histogram equalization
- **Deep Learning**: CNN architecture, training loops, backpropagation
- **Software Engineering**: Modular design, threading, real-time systems
- **Integration**: Combining ML models with interactive applications

## 📝 File Descriptions

### gesture_model.py
Defines the CNN architecture following PyTorch `nn.Module` pattern. Includes:
- `GestureRecognitionCNN` class
- Forward pass implementation
- Prediction utilities

### collect_gesture_data.py
Interactive data collection tool:
- Opens webcam
- Captures and preprocesses images
- Saves to organized directory structure
- Real-time preview of processed images

### train_gesture_model.py
Complete training pipeline:
- Custom `Dataset` loader
- Data augmentation transforms
- Training and validation loops
- Model checkpointing
- Visualization of training history

### flappy_bird_gesture.py
Modified Flappy Bird game:
- Accepts gesture input alongside mouse/keyboard
- Displays gesture information
- Maintains original game mechanics

### gesture_flappy_bird.py
Main application:
- Initializes model and webcam
- Runs gesture detection in background thread
- Updates game state in real-time
- Handles cleanup on exit

## 🔧 Troubleshooting

### "Cannot open webcam"
- Check if webcam is connected
- Close other applications using the camera
- Try changing camera index: `cv2.VideoCapture(1)`

### "gesture_model.pth not found"
- Run training script first
- Ensure training completed successfully
- Check current directory

### Low frame rate
- Close unnecessary applications
- Use GPU if available (CUDA)
- Reduce model size or image resolution

### Game assets missing
- Ensure `img/` folder contains:
  - bg.png
  - ground.png
  - bird1.png, bird2.png, bird3.png
  - pipe.png
  - restart.png

## 🎯 Future Enhancements

- [ ] Add more gestures (thumbs up, pointing, etc.)
- [ ] Multi-class gesture recognition
- [ ] Transfer learning with pre-trained models
- [ ] Mobile deployment
- [ ] Difficulty levels based on gesture accuracy
- [ ] Gesture-based menu navigation

## 📚 Learning Resources

This project is based on concepts from:
- `complete_deep_learning_tutorial.py` - Sections 1-9
- PyTorch documentation
- Computer vision techniques

## 🙏 Credits

- Original Flappy Bird: Dong Nguyen
- Deep Learning Tutorial: CPS-470 course materials
- Implementation: Gesture-controlled version

## 📄 License

Educational project for learning deep learning and computer vision.

---

**Have fun playing Flappy Bird with your hands!** 🎮✌️

If you have questions or issues, review the troubleshooting section or check the inline code comments for detailed explanations.
