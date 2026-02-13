"""
ORAM – Dataset Manager
Downloads, indexes and serves datasets for multi-agent training.

Each catalogue entry maps to a publicly downloadable archive or
a Kaggle dataset slug.  The manager keeps everything under
backend/data/datasets/<dataset_key>/.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Dataset Catalogue
# ---------------------------------------------------------------------------

@dataclass
class DatasetInfo:
    key: str
    name: str
    description: str
    category: str                   # crack | corrosion | railway | leak | defect
    size_approx: str                # human-readable
    source: str                     # kaggle | url | roboflow
    source_id: str                  # slug or URL
    classes: List[str] = field(default_factory=list)
    image_count: int = 0
    resolution: str = ""
    license: str = ""
    recommended: bool = False

# The catalogue — add entries as needed
DATASET_CATALOGUE: Dict[str, DatasetInfo] = {
    # ── Cracks ────────────────────────────────────────────────
    "sdnet2018": DatasetInfo(
        key="sdnet2018",
        name="SDNET2018 – Concrete Crack Detection",
        description="56 000+ images (256×256) of cracked / non-cracked concrete surfaces.",
        category="crack",
        size_approx="~2.2 GB",
        source="url",
        source_id="https://digitalcommons.usu.edu/all_datasets/48/",
        classes=["cracked", "non-cracked"],
        image_count=56000,
        resolution="256x256",
        license="CC-BY 4.0",
        recommended=True,
    ),
    "metu_cracks": DatasetInfo(
        key="metu_cracks",
        name="METU Concrete Crack Images",
        description="40 000 images (227×227) – balanced binary crack dataset.",
        category="crack",
        size_approx="~235 MB",
        source="url",
        source_id="https://data.mendeley.com/datasets/5y9wdsg2zt/2",
        classes=["positive", "negative"],
        image_count=40000,
        resolution="227x227",
        license="CC-BY 4.0",
    ),
    "surface_crack_kaggle": DatasetInfo(
        key="surface_crack_kaggle",
        name="Surface Crack Detection (Kaggle)",
        description="General surface crack detection dataset.",
        category="crack",
        size_approx="~120 MB",
        source="kaggle",
        source_id="arunrk7/surface-crack-detection",
        classes=["positive", "negative"],
        image_count=20000,
        resolution="227x227",
        license="Unknown",
    ),
    # ── Railway Specific ──────────────────────────────────────
    "railway_track_fault": DatasetInfo(
        key="railway_track_fault",
        name="Railway Track Fault Detection",
        description="Railway track surface defect images.",
        category="railway",
        size_approx="~80 MB",
        source="kaggle",
        source_id="salmaneunus/railway-track-fault-detection",
        classes=["defective", "non-defective"],
        image_count=1800,
        resolution="224x224",
        license="Unknown",
        recommended=True,
    ),
    "railway_track_resized": DatasetInfo(
        key="railway_track_resized",
        name="Railway Track Fault (Resized 224×224)",
        description="Pre-resized railway track defect images ready for DL.",
        category="railway",
        size_approx="~60 MB",
        source="kaggle",
        source_id="gpiosenka/railway-track-fault-detection-resized-224-x-224",
        classes=["defective", "non-defective"],
        image_count=1800,
        resolution="224x224",
        license="Unknown",
    ),
    # ── Corrosion ─────────────────────────────────────────────
    "corrosion_mendeley": DatasetInfo(
        key="corrosion_mendeley",
        name="Image-based Corrosion Detection",
        description="1 819 images — corrosion vs no-corrosion.",
        category="corrosion",
        size_approx="~110 MB",
        source="url",
        source_id="https://data.mendeley.com/datasets/tbjn6p2gn9/1",
        classes=["corrosion", "no_corrosion"],
        image_count=1819,
        resolution="various",
        license="CC-BY 4.0",
    ),
    # ── General Defects ───────────────────────────────────────
    "neu_surface": DatasetInfo(
        key="neu_surface",
        name="NEU Surface Defect Database",
        description="1 800 grayscale images – 6 steel surface defect types.",
        category="defect",
        size_approx="~25 MB",
        source="url",
        source_id="http://faculty.neu.edu.cn/songkechen/en/zdylm/263265/list/index.htm",
        classes=["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"],
        image_count=1800,
        resolution="200x200",
        license="Research",
    ),
    "mvtec_ad": DatasetInfo(
        key="mvtec_ad",
        name="MVTec Anomaly Detection",
        description="5 000+ images – 15 categories of industrial anomalies.",
        category="defect",
        size_approx="~4.9 GB",
        source="url",
        source_id="https://www.mvtec.com/company/research/datasets/mvtec-ad",
        classes=["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
                 "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"],
        image_count=5354,
        resolution="various",
        license="CC-BY-NC-SA 4.0",
        recommended=True,
    ),
}

# ---------------------------------------------------------------------------
# Dataset Manager
# ---------------------------------------------------------------------------

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "datasets"


class DatasetManager:
    """Manages download, indexing and serving of datasets."""

    def __init__(self, root: Path = DATA_ROOT):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    # ── catalogue ──────────────────────────────────────────────
    def list_datasets(self) -> List[Dict[str, Any]]:
        """Return catalogue with local-availability flag."""
        out: List[Dict[str, Any]] = []
        for ds in DATASET_CATALOGUE.values():
            local_path = self.root / ds.key
            installed = local_path.exists() and any(local_path.iterdir()) if local_path.exists() else False
            sample_count = self._count_images(local_path) if installed else 0
            out.append({
                "key": ds.key,
                "name": ds.name,
                "description": ds.description,
                "category": ds.category,
                "size_approx": ds.size_approx,
                "source": ds.source,
                "classes": ds.classes,
                "image_count": ds.image_count,
                "resolution": ds.resolution,
                "license": ds.license,
                "recommended": ds.recommended,
                "installed": installed,
                "local_samples": sample_count,
            })
        return out

    def get_dataset(self, key: str) -> Optional[Dict[str, Any]]:
        if key not in DATASET_CATALOGUE:
            return None
        ds = DATASET_CATALOGUE[key]
        local_path = self.root / ds.key
        installed = local_path.exists() and any(local_path.iterdir()) if local_path.exists() else False
        return {
            **{
                "key": ds.key, "name": ds.name, "description": ds.description,
                "category": ds.category, "classes": ds.classes,
                "image_count": ds.image_count, "resolution": ds.resolution,
            },
            "installed": installed,
            "local_path": str(local_path) if installed else None,
            "local_samples": self._count_images(local_path) if installed else 0,
        }

    # ── download helpers ───────────────────────────────────────
    async def download_kaggle(self, key: str) -> Dict[str, Any]:
        """Download a Kaggle dataset (requires kaggle CLI configured)."""
        ds = DATASET_CATALOGUE.get(key)
        if not ds or ds.source != "kaggle":
            return {"status": "error", "message": f"'{key}' is not a Kaggle dataset."}

        dest = self.root / key
        dest.mkdir(parents=True, exist_ok=True)

        cmd = f"kaggle datasets download -d {ds.source_id} -p {dest} --unzip"
        logger.info(f"Downloading Kaggle dataset: {cmd}")

        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            msg = stderr.decode(errors="replace")
            logger.error(f"Kaggle download failed: {msg}")
            return {"status": "error", "message": msg}

        samples = self._count_images(dest)
        logger.info(f"Dataset '{key}' downloaded – {samples} images found")
        return {"status": "ok", "key": key, "samples": samples, "path": str(dest)}

    async def download_url(self, key: str) -> Dict[str, Any]:
        """Download a dataset from a direct URL (zip)."""
        ds = DATASET_CATALOGUE.get(key)
        if not ds or ds.source != "url":
            return {"status": "error", "message": f"'{key}' is not a URL dataset."}

        dest = self.root / key
        dest.mkdir(parents=True, exist_ok=True)
        
        # Check if already exists
        if self._count_images(dest) > 100:
             return {"status": "ok", "key": key, "samples": self._count_images(dest), "path": str(dest)}

        url = ds.source_id
        # SDNET2018 direct link override if it's the landing page
        if key == "sdnet2018" and "digitalcommons.usu.edu" in url and "viewcontent" not in url:
             url = "https://digitalcommons.usu.edu/context/all_datasets/article/1047/type/native/viewcontent"
        
        logger.info(f"Downloading URL dataset '{key}' from {url} ...")
        
        # Use simple curl or wget
        zip_path = dest / "download.zip"
        # Try curl first
        cmd = f"curl -L -o \"{zip_path}\" \"{url}\""
        
        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
             return {"status": "error", "message": f"Download failed: {stderr.decode()}"}
             
        # Extract
        try:
            import zipfile
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dest)
            zip_path.unlink()
        except Exception as e:
             return {"status": "error", "message": f"Extraction failed: {e}"}

        samples = self._count_images(dest)
        logger.info(f"Dataset '{key}' downloaded – {samples} images found")
        return {"status": "ok", "key": key, "samples": samples, "path": str(dest)}

    async def create_synthetic(self, key: str, count: int = 500) -> Dict[str, Any]:
        """Create a small synthetic placeholder dataset for quick prototyping."""
        dest = self.root / key
        for cls in ["normal", "crack", "corrosion", "leak", "wear"]:
            (dest / cls).mkdir(parents=True, exist_ok=True)

        try:
            import numpy as np
            from PIL import Image as PILImage
        except ImportError:
            return {"status": "error", "message": "numpy/Pillow required"}

        generated = 0
        for cls in ["normal", "crack", "corrosion", "leak", "wear"]:
            for i in range(count // 5):
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                PILImage.fromarray(img).save(dest / cls / f"{cls}_{i:04d}.jpg")
                generated += 1

        return {"status": "ok", "key": key, "samples": generated, "path": str(dest)}

    # ── custom dataset upload ───────────────────────────────────
    async def add_custom_dataset(
        self,
        name: str,
        category: str,
        description: str,
        classes: List[str],
        file_bytes: bytes,
        filename: str,
    ) -> Dict[str, Any]:
        """Register and extract a user-uploaded dataset (ZIP of class folders)."""
        import re, uuid

        # Build a safe key from the name
        key = re.sub(r"[^a-z0-9_]", "_", name.lower().strip())[:40] or f"custom_{uuid.uuid4().hex[:8]}"
        if key in DATASET_CATALOGUE:
            key = f"{key}_{uuid.uuid4().hex[:6]}"

        dest = self.root / key
        dest.mkdir(parents=True, exist_ok=True)

        # Save the uploaded file
        tmp_file = dest / filename
        tmp_file.write_bytes(file_bytes)

        # If ZIP → extract and remove archive
        if filename.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(tmp_file, "r") as zf:
                    zf.extractall(dest)
                tmp_file.unlink(missing_ok=True)
            except zipfile.BadZipFile:
                return {"status": "error", "message": "Invalid ZIP file"}

        # If no class sub-folders exist, create one from loose images
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        loose = [f for f in dest.iterdir() if f.is_file() and f.suffix.lower() in image_exts]
        if loose and not any(d.is_dir() for d in dest.iterdir()):
            default_cls = dest / "images"
            default_cls.mkdir(exist_ok=True)
            for f in loose:
                shutil.move(str(f), str(default_cls / f.name))
            if not classes:
                classes = ["images"]

        # Detect classes from sub-folders if caller didn't specify
        detected_classes = [d.name for d in sorted(dest.iterdir()) if d.is_dir()]
        if not classes:
            classes = detected_classes or ["unknown"]

        sample_count = self._count_images(dest)

        # Register in catalogue
        ds = DatasetInfo(
            key=key,
            name=name,
            description=description or f"Custom dataset: {name}",
            category=category or "custom",
            size_approx=f"~{len(file_bytes) / (1024*1024):.0f} MB",
            source="upload",
            source_id="user-upload",
            classes=classes,
            image_count=sample_count,
            resolution="various",
            license="Custom",
            recommended=False,
        )
        DATASET_CATALOGUE[key] = ds
        logger.info(f"Custom dataset '{key}' registered – {sample_count} images, classes={classes}")

        return {
            "status": "ok",
            "key": key,
            "name": name,
            "classes": classes,
            "samples": sample_count,
            "path": str(dest),
        }

    async def delete_dataset(self, key: str) -> Dict[str, Any]:
        """Delete a dataset from disk and catalogue."""
        ds_path = self.root / key
        if ds_path.exists():
            shutil.rmtree(ds_path, ignore_errors=True)
        removed = DATASET_CATALOGUE.pop(key, None)
        return {"status": "ok", "key": key, "removed": removed is not None}

    # ── helpers ────────────────────────────────────────────────
    @staticmethod
    def _count_images(path: Path) -> int:
        if not path.exists():
            return 0
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        return sum(1 for f in path.rglob("*") if f.suffix.lower() in exts)

    def get_dataset_path(self, key: str) -> Optional[Path]:
        p = self.root / key
        return p if p.exists() else None


# Singleton
_manager: Optional[DatasetManager] = None


def get_dataset_manager() -> DatasetManager:
    global _manager
    if _manager is None:
        _manager = DatasetManager()
    return _manager
