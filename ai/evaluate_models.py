import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from ai.training import AgentConfig, AgentType, InspectionDataset, ModelBuilder
import json
import time

DATA_ROOT = Path(r"d:\Github\ORAM-defect-detection\data\datasets")
MODEL_ROOT = Path(r"d:\Github\ORAM-defect-detection\data\models")

AGENTS = [
    (AgentType.CRACK, "sdnet2018", "efficientnet_b0", 2),
    (AgentType.CORROSION, "corrosion_mendeley", "mobilenet_v3", 2), 
    (AgentType.GENERAL, "railway_track_fault", "efficientnet_b3", 2),
    (AgentType.LEAK, "synthetic", "resnet18", 2),
]


def find_latest_model(agent_type_val):
    if not MODEL_ROOT.exists():
        return None
    
    candidates = []
    # Scan all job folders
    for job_dir in MODEL_ROOT.iterdir():
        if not job_dir.is_dir(): continue
        
        # Check for agent folder inside
        agent_dir = job_dir / agent_type_val
        if not agent_dir.exists(): continue
        
        # Check for best.pth or final.pth
        p = agent_dir / "best.pth"
        if not p.exists():
             p = agent_dir / "final.pth"
        
        if p.exists():
            candidates.append((p, p.stat().st_mtime))
            
    if not candidates:
        return None
        
    # Sort by mtime descending
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}")

    for agent_type, dataset_key, model_name, num_classes in AGENTS:
        print(f"\n--- Evaluating {agent_type.value} ---")
        
        model_path = find_latest_model(agent_type.value)
        
        if not model_path:
            print(f"Model not found for {agent_type.value} in any job folder.")
            continue
            
        print(f"Loading model: {model_path} (Modified: {time.ctime(model_path.stat().st_mtime)})")

            
        # Check data
        data_dir = DATA_ROOT / dataset_key
        if not data_dir.exists():
             print(f"Dataset not found at {data_dir}")
             continue

        # Load Data
        # Re-use transform logic from training implicitly or simpler one
        from torchvision import transforms
        size = 224 if model_name != "efficientnet_b3" else 300
        tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        ds = InspectionDataset(str(data_dir), transform=tf)
        if len(ds) == 0:
            print("Dataset empty.")
            continue

        # Split (same seed to try and get val set, but without fixed seed in training it's approximate)
        # In real scenario, should have saved val indices. 
        # Here we just evaluate on a subset to get a "sense" of accuracy.
        train_n = int(len(ds) * 0.8)
        val_n = len(ds) - train_n
        _, val_ds = random_split(ds, [train_n, val_n])
        
        val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)
        
        # Load Model
        model = ModelBuilder.build(model_name, num_classes, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f"Accuracy: {100 * correct / total:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    evaluate()
