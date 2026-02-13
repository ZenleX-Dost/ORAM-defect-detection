"""
ORAM – SAM 2 Segmentation Module
==================================
Wraps Meta's Segment Anything Model 2 (SAM 2) for pixel-level defect
mask generation.  The module takes bounding-box or point prompts from
the detection pipeline and returns precise binary masks + contour
polygons for each defect.

SAM 2 runs on GPU when available, falls back to CPU.
If SAM 2 is not installed the module gracefully degrades,
returning empty masks so the rest of the pipeline keeps working.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

# ── Optional SAM 2 import ────────────────────────────────────────────────
SAM2_AVAILABLE = False
_sam2_build_fn = None
_sam2_predictor_cls = None

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    SAM2_AVAILABLE = True
    _sam2_build_fn = build_sam2
    _sam2_predictor_cls = SAM2ImagePredictor
except ImportError:
    try:
        # Alternative import paths depending on package version
        from segment_anything_2.build_sam import build_sam2
        from segment_anything_2.sam2_image_predictor import SAM2ImagePredictor

        SAM2_AVAILABLE = True
        _sam2_build_fn = build_sam2
        _sam2_predictor_cls = SAM2ImagePredictor
    except ImportError:
        logger.warning(
            "SAM 2 not installed – segmentation will return empty masks. "
            "Install with:  pip install segment-anything-2"
        )

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SegmentationMask:
    """Pixel-level mask for a single detected defect."""

    mask: np.ndarray  # H×W bool array
    contour: List[Tuple[int, int]] = field(default_factory=list)  # polygon
    area_pixels: int = 0
    iou_prediction: float = 0.0  # SAM's own IoU confidence
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contour": self.contour,
            "area_pixels": self.area_pixels,
            "iou_prediction": round(self.iou_prediction, 4),
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# SAM 2 Segmentor
# ---------------------------------------------------------------------------

# Default checkpoint paths (user can override)
DEFAULT_MODEL_CFG = "sam2_hiera_s.yaml"
DEFAULT_CHECKPOINT = "sam2_hiera_small.pt"


class SAM2Segmentor:
    """
    SAM 2 wrapper for defect segmentation.

    Usage
    -----
    >>> seg = SAM2Segmentor()
    >>> seg.load()  # downloads / loads checkpoint
    >>> mask = seg.segment_from_bbox(image, (x, y, w, h))
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_cfg: Optional[str] = None,
        device: str = "auto",
    ):
        self.checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
        self.model_cfg = model_cfg or DEFAULT_MODEL_CFG
        self.device = self._resolve_device(device)
        self.predictor = None
        self._loaded = False

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device

    @property
    def available(self) -> bool:
        return SAM2_AVAILABLE and TORCH_AVAILABLE

    # ── loading ────────────────────────────────────────────────

    def load(self) -> bool:
        """Load SAM 2 model.  Returns True on success."""
        if self._loaded:
            return True

        if not self.available:
            logger.warning("SAM 2 is not available – masks will be empty")
            return False

        try:
            checkpoint = Path(self.checkpoint_path)
            if not checkpoint.exists():
                # Check common locations
                for candidate in [
                    Path(__file__).parent / "checkpoints" / checkpoint.name,
                    Path.home() / ".cache" / "sam2" / checkpoint.name,
                ]:
                    if candidate.exists():
                        checkpoint = candidate
                        break

            if not checkpoint.exists():
                logger.warning(
                    f"SAM 2 checkpoint not found at {checkpoint}. "
                    "Download it from https://github.com/facebookresearch/segment-anything-2"
                )
                return False

            sam2_model = _sam2_build_fn(self.model_cfg, str(checkpoint))
            self.predictor = _sam2_predictor_cls(sam2_model)

            if TORCH_AVAILABLE:
                self.predictor.model = self.predictor.model.to(self.device)

            self._loaded = True
            logger.info(f"SAM 2 loaded on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load SAM 2: {e}")
            return False

    # ── segmentation methods ───────────────────────────────────

    def segment_from_bbox(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        multimask: bool = False,
    ) -> Optional[SegmentationMask]:
        """
        Generate mask from a bounding box prompt.

        Parameters
        ----------
        image : np.ndarray
            RGB image (H, W, 3).
        bbox : tuple
            (x, y, width, height).
        multimask : bool
            If True, return the best of 3 masks.

        Returns
        -------
        SegmentationMask or None
        """
        if not self._loaded:
            if not self.load():
                return self._empty_mask(image.shape[:2])

        t0 = time.perf_counter()

        try:
            # Convert bbox from (x, y, w, h) to (x1, y1, x2, y2)
            x, y, w, h = bbox
            input_box = np.array([x, y, x + w, y + h])

            # Set image
            self.predictor.set_image(image)

            # Predict
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=multimask,
            )

            # Pick highest-scoring mask
            best_idx = int(scores.argmax())
            mask = masks[best_idx].astype(bool)
            score = float(scores[best_idx])

            # Extract contour
            contour = self._mask_to_contour(mask)

            elapsed = (time.perf_counter() - t0) * 1000

            return SegmentationMask(
                mask=mask,
                contour=contour,
                area_pixels=int(mask.sum()),
                iou_prediction=score,
                processing_time_ms=elapsed,
            )

        except Exception as e:
            logger.error(f"SAM 2 segmentation failed: {e}")
            return self._empty_mask(image.shape[:2])

    def segment_from_point(
        self,
        image: np.ndarray,
        point: Tuple[int, int],
        label: int = 1,
    ) -> Optional[SegmentationMask]:
        """
        Generate mask from a single point prompt.

        Parameters
        ----------
        image : np.ndarray
            RGB image.
        point : tuple
            (x, y) pixel coordinate.
        label : int
            1 = foreground, 0 = background.
        """
        if not self._loaded:
            if not self.load():
                return self._empty_mask(image.shape[:2])

        t0 = time.perf_counter()

        try:
            self.predictor.set_image(image)

            input_point = np.array([[point[0], point[1]]])
            input_label = np.array([label])

            masks, scores, _ = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )

            best_idx = int(scores.argmax())
            mask = masks[best_idx].astype(bool)
            score = float(scores[best_idx])
            contour = self._mask_to_contour(mask)
            elapsed = (time.perf_counter() - t0) * 1000

            return SegmentationMask(
                mask=mask,
                contour=contour,
                area_pixels=int(mask.sum()),
                iou_prediction=score,
                processing_time_ms=elapsed,
            )

        except Exception as e:
            logger.error(f"SAM 2 point segmentation failed: {e}")
            return self._empty_mask(image.shape[:2])

    def segment_detections(
        self,
        image: np.ndarray,
        detections: list,
    ) -> List[Optional[SegmentationMask]]:
        """
        Run SAM 2 on every detection that has a bounding box.

        Parameters
        ----------
        image : np.ndarray
            RGB image.
        detections : list
            List of Detection objects (from detection.py).

        Returns
        -------
        list
            SegmentationMask for each detection (None if no bbox).
        """
        masks = []
        for det in detections:
            if det.bbox is not None:
                bbox = (det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height)
                mask_result = self.segment_from_bbox(image, bbox)
                masks.append(mask_result)
            else:
                masks.append(None)
        return masks

    # ── helpers ─────────────────────────────────────────────────

    @staticmethod
    def _mask_to_contour(mask: np.ndarray) -> List[Tuple[int, int]]:
        """Convert binary mask to the largest contour polygon."""
        if not CV2_AVAILABLE:
            return []

        mask_uint8 = (mask.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return []

        # Find largest contour
        largest = max(contours, key=cv2.contourArea)

        # Simplify polygon
        epsilon = 0.005 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)

        return [(int(pt[0][0]), int(pt[0][1])) for pt in approx]

    @staticmethod
    def _empty_mask(shape: Tuple[int, int]) -> SegmentationMask:
        """Return an empty mask when SAM 2 is not available."""
        return SegmentationMask(
            mask=np.zeros(shape, dtype=bool),
            contour=[],
            area_pixels=0,
            iou_prediction=0.0,
        )

    @staticmethod
    def mask_to_overlay(
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Overlay a coloured semi-transparent mask on an image."""
        overlay = image.copy()
        overlay[mask] = (
            (1 - alpha) * overlay[mask] + alpha * np.array(color)
        ).astype(np.uint8)
        return overlay


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_segmentor: Optional[SAM2Segmentor] = None


def get_segmentor(**kwargs) -> SAM2Segmentor:
    """Get or create singleton SAM 2 segmentor."""
    global _segmentor
    if _segmentor is None:
        _segmentor = SAM2Segmentor(**kwargs)
    return _segmentor
