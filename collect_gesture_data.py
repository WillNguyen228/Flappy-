"""
Data Collection Script for Hand Gesture Recognition

This script captures images from your webcam to create a training dataset
for the gesture recognition model. It captures two gestures:
- Fist (class 0): closed hand
- Peace sign (class 1): V-sign with two fingers

Usage:
1. Run the script
2. Press '0' to capture fist images
3. Press '1' to capture peace sign images
4. Press 'q' to quit
"""

import cv2
import os
import numpy as np
from datetime import datetime


class GestureDataCollector:
    """
    Collects hand gesture images from webcam for training
    """
    
    def __init__(self, data_dir="gesture_data", img_size=64):
        """
        Initialize the data collector
        
        Args:
            data_dir: Directory to save collected images
            img_size: Size to resize images to (img_size x img_size)
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.class_names = {0: "fist", 1: "peace"}
        
        # Create directories for each class
        for class_idx, class_name in self.class_names.items():
            class_dir = os.path.join(data_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Count images captured per class
        self.image_counts = {0: 0, 1: 0}
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame for gesture recognition
        
        Args:
            frame: Input frame from webcam (BGR)
            
        Returns:
            processed: Grayscale, resized image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Extract region of interest (center square)
        h, w = gray.shape
        size = min(h, w)
        start_h = (h - size) // 2
        start_w = (w - size) // 2
        roi = gray[start_h:start_h+size, start_w:start_w+size]
        
        # Resize to target size
        resized = cv2.resize(roi, (self.img_size, self.img_size))
        
        # Apply histogram equalization for better contrast
        equalized = cv2.equalizeHist(resized)
        
        return equalized
    
    def draw_roi(self, frame):
        """
        Draw region of interest rectangle on frame
        
        Args:
            frame: Input frame
            
        Returns:
            frame with ROI rectangle drawn
        """
        h, w = frame.shape[:2]
        size = min(h, w)
        start_h = (h - size) // 2
        start_w = (w - size) // 2
        
        cv2.rectangle(frame, 
                     (start_w, start_h), 
                     (start_w + size, start_h + size), 
                     (0, 255, 0), 2)
        
        return frame
    
    def draw_instructions(self, frame):
        """
        Draw instructions on the frame
        
        Args:
            frame: Input frame
            
        Returns:
            frame with instructions
        """
        instructions = [
            "Press '0' for FIST",
            "Press '1' for PEACE",
            "Press 'q' to quit",
            "",
            f"Fist images: {self.image_counts[0]}",
            f"Peace images: {self.image_counts[1]}"
        ]
        
        y_offset = 30
        for idx, text in enumerate(instructions):
            cv2.putText(frame, text, (10, y_offset + idx * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def save_image(self, image, class_idx):
        """
        Save captured image to disk
        
        Args:
            image: Preprocessed image
            class_idx: Class index (0 for fist, 1 for peace)
        """
        class_name = self.class_names[class_idx]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{class_name}_{timestamp}.png"
        filepath = os.path.join(self.data_dir, class_name, filename)
        
        cv2.imwrite(filepath, image)
        self.image_counts[class_idx] += 1
        print(f"Saved: {filename} (Total {class_name}: {self.image_counts[class_idx]})")
    
    def collect_data(self):
        """
        Main loop for collecting gesture data
        """
        print("=" * 60)
        print("Gesture Data Collection")
        print("=" * 60)
        print("\nInstructions:")
        print("1. Position your hand in the green square")
        print("2. Press '0' to capture FIST gesture")
        print("3. Press '1' to capture PEACE gesture")
        print("4. Try to capture ~100-200 images per gesture")
        print("5. Vary your hand position, angle, and lighting")
        print("6. Press 'q' to quit when done")
        print("\nStarting webcam...\n")
        
        while True:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Preprocess for preview
            processed = self.preprocess_frame(frame)
            
            # Draw ROI and instructions
            display = self.draw_roi(frame.copy())
            display = self.draw_instructions(display)
            
            # Show preview of processed image
            processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            processed_large = cv2.resize(processed_bgr, (200, 200))
            
            # Place processed preview in corner
            display[10:210, display.shape[1]-210:display.shape[1]-10] = processed_large
            
            # Display
            cv2.imshow('Gesture Data Collection', display)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting data collection...")
                break
            elif key == ord('0'):
                # Capture fist
                self.save_image(processed, 0)
            elif key == ord('1'):
                # Capture peace sign
                self.save_image(processed, 1)
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("Data collection complete!")
        print(f"Total images collected:")
        print(f"  Fist: {self.image_counts[0]}")
        print(f"  Peace: {self.image_counts[1]}")
        print(f"\nImages saved to: {os.path.abspath(self.data_dir)}")
        print("=" * 60)


def main():
    """
    Main function to run data collection
    """
    collector = GestureDataCollector(data_dir="gesture_data", img_size=64)
    collector.collect_data()


if __name__ == "__main__":
    main()
