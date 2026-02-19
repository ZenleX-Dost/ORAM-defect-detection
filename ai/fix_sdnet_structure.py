import shutil
from pathlib import Path

DATA_ROOT = Path(r"d:\Github\ORAM-defect-detection\data\datasets\sdnet2018")
SRC_DIR = DATA_ROOT / "SDNET2018"

def reorganize():
    if not SRC_DIR.exists():
        print(f"Source directory {SRC_DIR} not found.")
        return

    # Target directories
    cracked_dir = DATA_ROOT / "cracked"
    cracked_dir.mkdir(exist_ok=True)
    
    non_cracked_dir = DATA_ROOT / "non-cracked"
    non_cracked_dir.mkdir(exist_ok=True)

    # SDNET Structure:
    # D/CD (Cracked Deck), D/UD (Uncracked Deck)
    # P/CP (Cracked Pavement), P/UP (Uncracked Pavement)
    # W/CW (Cracked Wall), W/UW (Uncracked Wall)
    
    subdirs = ["D", "P", "W"]
    counts = {"cracked": 0, "non-cracked": 0}

    for sub in subdirs:
        s_path = SRC_DIR / sub
        if not s_path.exists():
            continue
            
        print(f"Processing {sub}...")
        
        # Identify cracked vs uncracked folders
        # They usually start with C (Cracked) or U (Uncracked)
        for child in s_path.iterdir():
            if not child.is_dir():
                continue
                
            if child.name.startswith("C"):
                target = cracked_dir
                key = "cracked"
            elif child.name.startswith("U"):
                target = non_cracked_dir
                key = "non-cracked"
            else:
                print(f"Skipping unknown folder: {child.name}")
                continue
                
            # Move files
            for f in child.glob("*.jpg"):
                # Avoid overwriting if names conflict (prepend parent name)
                new_name = f"{sub}_{child.name}_{f.name}"
                try:
                    shutil.move(str(f), str(target / new_name))
                    counts[key] += 1
                except Exception as e:
                    print(f"Error moving {f.name}: {e}")

    print(f" reorganization complete.")
    print(f"Cracked images: {counts['cracked']}")
    print(f"Non-cracked images: {counts['non-cracked']}")
    
    # Clean up empty source folders
    try:
        shutil.rmtree(SRC_DIR)
        print("Removed original SDNET2018 folder.")
    except Exception as e:
        print(f"Could not remove source folder: {e}")

if __name__ == "__main__":
    reorganize()
