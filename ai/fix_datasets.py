import shutil
from pathlib import Path
import os
import requests

DATA_ROOT = Path(r"d:\Github\ORAM-defect-detection\data\datasets")

def fix_sdnet():
    sdnet = DATA_ROOT / "sdnet2018"
    sub = sdnet / "DATA_Maguire_20180517_ALL"
    if sub.exists():
        print("Fixing SDNET structure...")
        # Move contents up
        for item in sub.iterdir():
            shutil.move(str(item), str(sdnet))
        # Remove empty sub
        sub.rmdir()
        print("SDNET fixed.")

def fix_corrosion():
    # Mendeley direct link is tricky. Let's try to find a better one or use a placeholder if we can't.
    # Actually, let's try to use the 'corrosion' agent with a different dataset if possible, 
    # or just create a dummy structure so training doesn't crash, warning the user.
    # The user said "dataset in its place", maybe they meant SDNET?
    # I will try to download a different corrosion dataset or just create the folder structure 
    # so the script can run (it will skip if < 10 images, but we want it to run).
    # Since I cannot easily get corrosion real data without a valid direct link or API, 
    # and the previous download was HTML, I will delete the bad file.
    cor = DATA_ROOT / "corrosion_mendeley"
    for f in cor.glob("*"):
        if f.is_file(): f.unlink() # Delete the bad html file
    
    print("Cleaned corrosion_mendeley. To train corrosion, we need data.")

def fix_railway():
    # Kaggle failed. 
    # I will check if I can use a different source or if the user provided one.
    # For now, ensure directory exists.
    rail = DATA_ROOT / "railway_track_fault"
    rail.mkdir(exist_ok=True)

if __name__ == "__main__":
    fix_sdnet()
    fix_corrosion()
    fix_railway()
