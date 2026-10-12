"""
Microplastic Detector — Model Training Script
Trains or fine-tunes a YOLO model on the microplastic dataset.
"""

import os
import sys
from ultralytics import YOLO

# Force UTF-8 encoding for Windows stdout console printing
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def main():
    # Define relative path to dataset config
    yaml_path = os.path.abspath(
        os.path.join("microplastic-detection", "microplastic-detection", "data", "data.yaml")
    )
    
    print(f"📁 Loading dataset config from: {yaml_path}")
    
    if not os.path.exists(yaml_path):
        print(f"❌ Error: Dataset config file not found at {yaml_path}")
        return

    # Load base model checkpoint
    base_model = "yolov8n.pt"
    print(f"🧠 Initializing YOLO model from {base_model}...")
    model = YOLO(base_model)
    
    # Train the model
    print("🚀 Starting microplastic detection model training...")
    results = model.train(
        data=yaml_path,
        epochs=30,
        imgsz=640,
        batch=16,
        name="microplastic_yolo_run"
    )
    
    print("\n✅ Training completed successfully!")
    print("🏆 Trained weights are saved in: runs/detect/microplastic_yolo_run/weights/best.pt")

if __name__ == "__main__":
    main()
