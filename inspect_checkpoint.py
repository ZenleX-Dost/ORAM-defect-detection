import torch
import sys

def inspect_checkpoint(path):
    print(f"Inspecting {path}...")
    try:
        # map_location='cpu' to avoid OOM if GPU is busy
        state = torch.load(path, map_location='cpu')
        print(f"Type: {type(state)}")
        if isinstance(state, dict):
            print(f"Keys: {list(state.keys())}")
            if "model" in state:
                print("Found 'model' key.")
                if isinstance(state["model"], dict):
                    print(f"Model keys sample: {list(state['model'].keys())[:5]}")
            elif "state_dict" in state:
                 print("Found 'state_dict' key.")
            else:
                 print("No 'model' or 'state_dict' key. Sample keys:", list(state.keys())[:5])
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    inspect_checkpoint(r"d:\Github\ORAM-defect-detection\ai\ai\checkpoints\sam2_hiera_small.pt")
