import torch
import numpy as np
import cv2
from pathlib import Path
try:
    from ai.segmentation import SAM2Segmentor
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from ai.segmentation import SAM2Segmentor

def test_sam():
    print("Testing SAM 2 Integration...")
    
    seg = SAM2Segmentor(device="cuda")
    if not seg.available:
        print("SAM 2 or Torch not available.")
        return

    # Create dummy image (checkerboard)
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    # White square in middle
    img[50:200, 50:200] = 255
    
    # Dummy bbox around the square
    bbox = (50, 50, 150, 150) # x, y, w, h
    
    print("Attempting to load SAM 2 model...")
    # This will trigger download if not present
    if not seg.load():
        print("Failed to load SAM 2 model.")
        return

    print("Running segmentation...")
    mask = seg.segment_from_bbox(img, bbox)
    
    if mask and mask.area_pixels > 0:
        print("Success! Mask generated.")
        print(f"Area: {mask.area_pixels}")
        print(f"IoU Prediction: {mask.iou_prediction}")
    else:
        print("Segmentation failed or returned empty mask.")

if __name__ == "__main__":
    test_sam()
