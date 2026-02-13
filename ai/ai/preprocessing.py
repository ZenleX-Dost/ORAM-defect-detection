"""
ORAM – Lighting-Robust Preprocessing Pipeline
===============================================
Adaptive image preprocessing for extreme lighting conditions
found under train carriages: deep shadows, uneven illumination,
strong spotlights, and mixed colour temperatures.

Pipeline stages
---------------
1. **Adaptive Gamma Correction** – auto-detects overall brightness
   and applies inverse-gamma to normalise exposure.
2. **CLAHE** – Contrast-Limited Adaptive Histogram Equalisation on
   the luminance channel preserves colour while boosting local contrast.
3. **White Balance Correction** – grey-world assumption removes
   colour casts from industrial lighting.
4. **Shadow / Highlight Recovery** – sigmoid tone-mapping compresses
   the dynamic range so that both dark corners and bright reflections
   retain usable detail.
"""
from __future__ import annotations

import numpy as np
from loguru import logger

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available – preprocessing will pass through images unchanged")


# ---------------------------------------------------------------------------
# Core normaliser
# ---------------------------------------------------------------------------


class LightingNormalizer:
    """
    End-to-end lighting normaliser for a single frame.

    Parameters
    ----------
    clahe_clip : float
        CLAHE clip limit (higher = more contrast).  Default 3.0.
    clahe_grid : int
        CLAHE tile grid size.  Default 8.
    target_brightness : int
        Target mean brightness (0-255) for gamma correction.  Default 120.
    enable_white_balance : bool
        Apply grey-world white balance.  Default True.
    enable_shadow_recovery : bool
        Apply sigmoid tone mapping.  Default True.
    """

    def __init__(
        self,
        clahe_clip: float = 3.0,
        clahe_grid: int = 8,
        target_brightness: int = 120,
        enable_white_balance: bool = True,
        enable_shadow_recovery: bool = True,
    ):
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.target_brightness = target_brightness
        self.enable_white_balance = enable_white_balance
        self.enable_shadow_recovery = enable_shadow_recovery

        if CV2_AVAILABLE:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip,
                tileGridSize=(self.clahe_grid, self.clahe_grid),
            )

    # ── public API ─────────────────────────────────────────────

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full normalisation pipeline.

        Parameters
        ----------
        image : np.ndarray
            BGR uint8 image (OpenCV convention).

        Returns
        -------
        np.ndarray
            Normalised BGR uint8 image.
        """
        if not CV2_AVAILABLE:
            return image

        img = image.copy()

        # 1. Adaptive gamma correction
        img = self._adaptive_gamma(img)

        # 2. CLAHE on luminance channel
        img = self._apply_clahe(img)

        # 3. White balance
        if self.enable_white_balance:
            img = self._white_balance(img)

        # 4. Shadow / highlight recovery
        if self.enable_shadow_recovery:
            img = self._shadow_highlight_recovery(img)

        return img

    # ── stage implementations ──────────────────────────────────

    def _adaptive_gamma(self, image: np.ndarray) -> np.ndarray:
        """Auto-detect brightness and apply inverse gamma to normalise."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = gray.mean()

        if mean_brightness < 1:
            mean_brightness = 1  # avoid log(0)

        # gamma < 1 brightens, gamma > 1 darkens
        gamma = np.log(self.target_brightness / 255.0) / np.log(
            mean_brightness / 255.0 + 1e-7
        )
        gamma = np.clip(gamma, 0.2, 5.0)  # safety clamp

        table = np.array(
            [((i / 255.0) ** gamma) * 255 for i in range(256)]
        ).astype("uint8")

        return cv2.LUT(image, table)

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """CLAHE on L channel of LAB colour space."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_chan = self._clahe.apply(l_chan)
        lab = cv2.merge([l_chan, a_chan, b_chan])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _white_balance(image: np.ndarray) -> np.ndarray:
        """Grey-world assumption white balance."""
        result = image.copy().astype(np.float32)
        avg_b = result[:, :, 0].mean()
        avg_g = result[:, :, 1].mean()
        avg_r = result[:, :, 2].mean()
        avg_all = (avg_b + avg_g + avg_r) / 3.0

        if avg_b > 0:
            result[:, :, 0] *= avg_all / avg_b
        if avg_g > 0:
            result[:, :, 1] *= avg_all / avg_g
        if avg_r > 0:
            result[:, :, 2] *= avg_all / avg_r

        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _shadow_highlight_recovery(image: np.ndarray) -> np.ndarray:
        """Sigmoid tone-mapping to recover shadows and compress highlights."""
        img_f = image.astype(np.float32) / 255.0

        # Sigmoid centred at 0.5, steepness 8
        img_f = 1.0 / (1.0 + np.exp(-8.0 * (img_f - 0.5)))

        # Re-scale to [0, 255]
        img_f = (img_f - img_f.min()) / (img_f.max() - img_f.min() + 1e-7)
        return (img_f * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Training augmentation helpers
# ---------------------------------------------------------------------------


def build_lighting_augmentations():
    """
    Return a list of torchvision-compatible transforms that simulate
    challenging lighting conditions found under train carriages.

    Includes:
    - Random brightness/contrast shifts
    - Random gamma
    - Gaussian noise (sensor noise in dark environments)
    - Simulated shadow bands via random rectangular darkening
    """
    from torchvision import transforms

    class RandomGamma:
        """Randomly adjust gamma in the range [lo, hi]."""

        def __init__(self, lo: float = 0.5, hi: float = 2.0):
            self.lo = lo
            self.hi = hi

        def __call__(self, img):
            import torch
            gamma = float(torch.empty(1).uniform_(self.lo, self.hi).item())
            return transforms.functional.adjust_gamma(img, gamma)

    class GaussianNoise:
        """Add Gaussian noise to simulate sensor noise in dark scenes."""

        def __init__(self, mean: float = 0.0, std: float = 0.02):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            import torch
            noise = torch.randn_like(tensor) * self.std + self.mean
            return torch.clamp(tensor + noise, 0.0, 1.0)

    class RandomShadowBand:
        """Overlay a random dark horizontal or vertical band."""

        def __call__(self, tensor):
            import torch

            _, h, w = tensor.shape
            if torch.rand(1).item() > 0.5:
                return tensor  # skip half the time

            if torch.rand(1).item() > 0.5:
                # horizontal band
                y0 = int(torch.randint(0, h // 2, (1,)).item())
                band_h = int(torch.randint(h // 8, h // 3, (1,)).item())
                factor = torch.empty(1).uniform_(0.3, 0.7).item()
                tensor[:, y0 : y0 + band_h, :] *= factor
            else:
                # vertical band
                x0 = int(torch.randint(0, w // 2, (1,)).item())
                band_w = int(torch.randint(w // 8, w // 3, (1,)).item())
                factor = torch.empty(1).uniform_(0.3, 0.7).item()
                tensor[:, :, x0 : x0 + band_w] *= factor

            return torch.clamp(tensor, 0.0, 1.0)

    return [RandomGamma(), GaussianNoise(), RandomShadowBand()]


# ---------------------------------------------------------------------------
# Convenience singleton
# ---------------------------------------------------------------------------

_normalizer = None


def get_lighting_normalizer(**kwargs) -> LightingNormalizer:
    """Get or create a singleton LightingNormalizer."""
    global _normalizer
    if _normalizer is None:
        _normalizer = LightingNormalizer(**kwargs)
    return _normalizer
