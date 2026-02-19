import shutil
from pathlib import Path
import random
import yaml

# Paths
DATA_ROOT = Path(r"d:\Github\ORAM-defect-detection\data\datasets")
SDNET_ROOT = DATA_ROOT / "sdnet2018"
YOLO_ROOT = DATA_ROOT / "yolo_crack_dataset"

def setup_directories():
    if YOLO_ROOT.exists():
        shutil.rmtree(YOLO_ROOT)
    
    for split in ["train", "val"]:
        (YOLO_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

def convert_dataset():
    print("Converting SDNET to YOLO format...")
    setup_directories()
    
    # Classes in SDNET (post-fix)
    classes = ["cracked", "non-cracked"]
    
    # We only create labels for "cracked". "non-cracked" are background images (empty label file optional, but usually just no label line).
    # Actually for YOLO, if no object is present, we don't need a label file, OR an empty one. 
    # Ultralytics recommends: "If no objects in image, no label file is required." (But creating empty file is safer to avoid warnings)
    
    all_images = []
    
    for cls_name in classes:
        src_dir = SDNET_ROOT / cls_name
        if not src_dir.exists():
            print(f"Directory {src_dir} not found. Skipping.")
            continue
            
        print(f"Processing {cls_name}...")
        label_id = 0 # 'cracked' is class 0.
        
        # Get all images
        images = list(src_dir.glob("*.jpg"))
        
        for img_path in images:
            all_images.append((img_path, cls_name))

    # Shuffle and split
    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.8)
    train_imgs = all_images[:split_idx]
    val_imgs = all_images[split_idx:]
    
    process_split(train_imgs, "train")
    process_split(val_imgs, "val")
    
    create_yaml()
    print("Conversion complete.")

def process_split(image_list, split_name):
    print(f"Processing {split_name} split ({len(image_list)} images)...")
    
    for img_path, cls_name in image_list:
        # Copy image
        dest_img_path = YOLO_ROOT / "images" / split_name / img_path.name
        shutil.copy(img_path, dest_img_path)
        
        # Create label
        # If cracked, create label file with bbox covering full image
        # If non-cracked, create empty label file (or skip, but empty is explicit)
        if cls_name == "cracked":
            label_content = "0 0.5 0.5 1.0 1.0" # class x_center y_center width height
        else:
            label_content = ""
            
        label_path = YOLO_ROOT / "labels" / split_name / (img_path.stem + ".txt")
        with open(label_path, "w") as f:
            f.write(label_content)

def create_yaml():
    data = {
        "path": str(YOLO_ROOT.absolute()),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "crack"
        }
    }
    
    with open(YOLO_ROOT / "data.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"Created data.yaml at {YOLO_ROOT / 'data.yaml'}")

if __name__ == "__main__":
    convert_dataset()
