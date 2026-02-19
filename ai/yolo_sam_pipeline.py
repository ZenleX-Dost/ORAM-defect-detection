import argparse
import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
# Add the directory containing the 'ai' package (which is inside 'ai' folder) to sys.path
# 'ai/ai' is the package, so we need to add 'ai' (outer) to path.
# This script is in 'ai' (outer).
sys.path.append(str(Path(__file__).parent))

try:
    from ai.segmentation import get_segmentor
except ImportError:
    # Fallback or alternative import if needed
    try:
        from segmentation import get_segmentor
    except ImportError:
        logger.error("Could not import get_segmentor from ai.segmentation")
        raise

class DefectPipeline:
    def __init__(self, yolo_model_path="yolov8n.pt", sam2_checkpoint=None, device="auto"):
        self.device = device
        logger.info(f"Initializing YOLO from {yolo_model_path}...")
        self.detector = YOLO(yolo_model_path)
        
        logger.info("Initializing SAM 2...")
        self.segmentor = get_segmentor(checkpoint_path=sam2_checkpoint, device=device)
        
    def process_image(self, image_path: Path, output_path: Path = None):
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return
            
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return

        # Prepare RGB image for SAM 2
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        # 1. Detect
        # YOLOv8 expectation: BGR for numpy arrays (default OpenCV format)
        results = self.detector(source=img, conf=0.25, verbose=False)
        
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0].cpu().numpy() # x1, y1, x2, y2
                conf = float(box.conf)
                cls = int(box.cls)
                
                # Convert to x, y, w, h
                x, y = int(b[0]), int(b[1])
                w, h = int(b[2] - b[0]), int(b[3] - b[1])
                
                detections.append({
                    "bbox": (x, y, w, h),
                    "conf": conf,
                    "class": cls,
                    "label": self.detector.names[cls]
                })
        
        logger.info(f"Detected {len(detections)} defects.")
        
        # 2. Segment
        # Load SAM 2 if needed (lazy loading)
        if not self.segmentor._loaded: # Access private attr just to check
             self.segmentor.load()
             
        masks = []
        for det in detections:
            bbox = det["bbox"]
            # Pass correct RGB image
            mask_obj = self.segmentor.segment_from_bbox(img_rgb, bbox)
            if mask_obj:
                det["mask"] = mask_obj.mask
                det["contour"] = mask_obj.contour
                masks.append(mask_obj)
        
        # 3. Visualize
        if output_path:
            vis_img = img.copy()
            # Draw boxes
            for det in detections:
                x, y, w, h = det["bbox"]
                cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(vis_img, f"{det['label']} {det['conf']:.2f}", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Draw mask contour
                if "contour" in det and det["contour"]:
                    pts = np.array(det["contour"], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(vis_img, [pts], True, (0, 255, 0), 2)
                    
                    # Overlay mask
                    if "mask" in det:
                         # Cyan overlay
                         mask = det["mask"]
                         vis_img[mask] = (vis_img[mask] * 0.5 + np.array([255, 255, 0]) * 0.5).astype(np.uint8)

            cv2.imwrite(str(output_path), vis_img)
            logger.info(f"Saved visualization to {output_path}")
            
        return detections

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 + SAM 2 pipeline.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument("--yolo", type=str, default="yolov8n.pt", help="Path to YOLO model.")
    parser.add_argument("--sam", type=str, default=None, help="Path to SAM 2 checkpoint.")
    parser.add_argument("--out", type=str, default="output.jpg", help="Path to output image.")
    args = parser.parse_args()
    
    pipeline = DefectPipeline(yolo_model_path=args.yolo, sam2_checkpoint=args.sam)
    pipeline.process_image(Path(args.image), Path(args.out))

if __name__ == "__main__":
    main()
