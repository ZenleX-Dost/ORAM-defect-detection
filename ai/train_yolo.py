from ultralytics import YOLO
from pathlib import Path

# Paths
DATA_ROOT = Path(r"d:\Github\ORAM-defect-detection\data\datasets\yolo_crack_dataset")
YAML_PATH = DATA_ROOT / "data.yaml"

def train():
    # Load model
    # distinct from classification, we use 'yolov8n.pt' for detection
    model = YOLO("yolov8n.pt") 
    
    print("Starting YOLOv8 training...")
    results = model.train(
        data=str(YAML_PATH),
        epochs=10, 
        imgsz=256,
        batch=32,
        name="yolo_crack_detection",
        device=0, # Use GPU 0
        pretrained=True,
        plots=True
    )
    
    print("Training complete.")
    print(f"Best model saved at: {results.save_dir}")

if __name__ == "__main__":
    train()
