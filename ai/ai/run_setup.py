"""
ORAM – Setup & Train with REAL HuggingFace Datasets
=====================================================
Uses real image datasets from HuggingFace (no API key needed):

 • Francesco/corrosion-bi3q3 → crack, corrosion, slippage (1144 images)
 • Saves to data/ and trains 4 agents on CUDA

Run:  python ai/run_setup.py
"""
from __future__ import annotations

import json, math, os, random, shutil, sys, time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms, models
from loguru import logger

DATA_ROOT  = ROOT / "data"
MODEL_ROOT = ROOT / "data" / "models"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ═══════════════════════════════════════════════════════════════════════
# 1. Download & Save Real Dataset from HuggingFace
# ═══════════════════════════════════════════════════════════════════════

def download_hf_dataset():
    """
    Download Francesco/corrosion-bi3q3 dataset.
    Contains real images labelled: corrosion, crack, slippage.
    We save them as local image folders for training.
    """
    # Fix: local datasets.py shadows HuggingFace datasets package
    # Temporarily adjust sys.path to import the correct one
    import importlib
    ai_dir = str(Path(__file__).resolve().parent)
    old_path = sys.path.copy()
    # Remove local ai/ directory from path to avoid collision
    sys.path = [p for p in sys.path if os.path.normpath(p) != os.path.normpath(ai_dir)]
    # Also remove any cached import of our local datasets
    if "datasets" in sys.modules:
        saved_datasets = sys.modules.pop("datasets")
    else:
        saved_datasets = None
    try:
        from datasets import load_dataset as hf_load_dataset
    finally:
        sys.path = old_path
        if saved_datasets is not None:
            sys.modules["datasets"] = saved_datasets

    base = DATA_ROOT / "hf_corrosion"
    marker = base / ".downloaded"
    if marker.exists():
        logger.info("  [HF] Dataset already downloaded, skipping")
        return base

    logger.info("  [HF] Downloading Francesco/corrosion-bi3q3 from HuggingFace ...")

    for split_name in ("train", "test"):
        ds = hf_load_dataset("Francesco/corrosion-bi3q3", split=split_name)

        # ClassLabel names: ['Slippage', 'corrosion', 'crack']
        # Category IDs in data are 1-indexed
        LABEL_MAP = {1: "slippage", 2: "corrosion", 3: "crack"}

        logger.info(f"  [HF] {split_name}: {len(ds)} samples")

        for i, row in enumerate(ds):
            img = row["image"]  # PIL Image
            # Get dominant class from category list
            cats = row.get("objects", {}).get("category", [])
            if cats and len(cats) > 0:
                label = LABEL_MAP.get(cats[0], "unlabelled")
            else:
                label = "unlabelled"

            out_dir = base / label
            out_dir.mkdir(parents=True, exist_ok=True)
            img.save(out_dir / f"{split_name}_{i:05d}.jpg", quality=92)

    # Log summary
    for cls_dir in sorted(base.iterdir()):
        if cls_dir.is_dir() and cls_dir.name != ".downloaded":
            n = len(list(cls_dir.glob("*.jpg")))
            logger.info(f"  [HF]   {cls_dir.name}: {n} images")

    marker.touch()
    return base


def prepare_agent_datasets(hf_dir: Path) -> dict:
    """
    From the HF dataset which has crack / corrosion / slippage classes,
    create 4 agent-specific binary datasets.
    """
    data_dirs = {}

    # ── Crack Agent ────────────────────────────────────────────────
    crack_dir = DATA_ROOT / "crack"
    if not (crack_dir / "crack").exists() or len(list((crack_dir / "crack").glob("*"))) < 10:
        logger.info("  Preparing crack dataset ...")
        (crack_dir / "crack").mkdir(parents=True, exist_ok=True)
        (crack_dir / "normal").mkdir(parents=True, exist_ok=True)

        # Positive: crack class from HF
        src = hf_dir / "crack"
        if src.exists():
            for i, f in enumerate(sorted(src.glob("*.jpg"))):
                shutil.copy2(f, crack_dir / "crack" / f"real_{i:05d}.jpg")

        # Also use slippage as defective (structural damage)
        src = hf_dir / "slippage"
        if src.exists():
            for i, f in enumerate(sorted(src.glob("*.jpg"))):
                shutil.copy2(f, crack_dir / "crack" / f"slip_{i:05d}.jpg")

        # Negative: use some corrosion images as "not-crack" + generate clean metal
        pos_count = len(list((crack_dir / "crack").glob("*.jpg")))
        src = hf_dir / "corrosion"
        neg_idx = 0
        if src.exists():
            for f in sorted(src.glob("*.jpg"))[:pos_count // 2]:
                shutil.copy2(f, crack_dir / "normal" / f"cor_{neg_idx:05d}.jpg")
                neg_idx += 1

        # Balance with clean metal
        remaining = pos_count - neg_idx
        for j in range(max(remaining, 0)):
            img = _metal_bg(224)
            img.save(crack_dir / "normal" / f"gen_{j:05d}.jpg", quality=90)

        _log_ds("crack", crack_dir)
    data_dirs["crack"] = crack_dir

    # ── Corrosion Agent ────────────────────────────────────────────
    corrosion_dir = DATA_ROOT / "corrosion"
    if not (corrosion_dir / "corrosion").exists() or len(list((corrosion_dir / "corrosion").glob("*"))) < 10:
        logger.info("  Preparing corrosion dataset ...")
        (corrosion_dir / "corrosion").mkdir(parents=True, exist_ok=True)
        (corrosion_dir / "normal").mkdir(parents=True, exist_ok=True)

        src = hf_dir / "corrosion"
        if src.exists():
            for i, f in enumerate(sorted(src.glob("*.jpg"))):
                shutil.copy2(f, corrosion_dir / "corrosion" / f"real_{i:05d}.jpg")

        pos_count = len(list((corrosion_dir / "corrosion").glob("*.jpg")))

        # Normal: use crack images + clean metal as "not corroded"
        neg_idx = 0
        src = hf_dir / "crack"
        if src.exists():
            for f in sorted(src.glob("*.jpg"))[:pos_count // 2]:
                shutil.copy2(f, corrosion_dir / "normal" / f"crack_{neg_idx:05d}.jpg")
                neg_idx += 1

        remaining = pos_count - neg_idx
        for j in range(max(remaining, 0)):
            img = _metal_bg(224)
            img.save(corrosion_dir / "normal" / f"gen_{j:05d}.jpg", quality=90)

        _log_ds("corrosion", corrosion_dir)
    data_dirs["corrosion"] = corrosion_dir

    # ── Leak Agent ─────────────────────────────────────────────────
    leak_dir = DATA_ROOT / "leak"
    if not (leak_dir / "leak").exists() or len(list((leak_dir / "leak").glob("*"))) < 10:
        logger.info("  Preparing leak dataset (using corrosion stains as proxy + augmented data) ...")
        (leak_dir / "leak").mkdir(parents=True, exist_ok=True)
        (leak_dir / "normal").mkdir(parents=True, exist_ok=True)

        # Use corrosion images tinted darker as leak-like stains
        src = hf_dir / "corrosion"
        idx = 0
        if src.exists():
            for f in sorted(src.glob("*.jpg")):
                img = Image.open(f).convert("RGB")
                # Darken to simulate wet/oily stains
                arr = np.array(img).astype(np.float32)
                arr *= random.uniform(0.4, 0.7)
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                Image.fromarray(arr).save(leak_dir / "leak" / f"leak_{idx:05d}.jpg", quality=90)
                idx += 1

        # Also generate some dark-stain synthetic images
        for k in range(200):
            img = _generate_leak_image(224)
            img.save(leak_dir / "leak" / f"synth_{k:05d}.jpg", quality=90)

        pos_count = len(list((leak_dir / "leak").glob("*.jpg")))

        # Normal: clean metal + some HF images
        neg_idx = 0
        for cls_name in ("crack", "slippage"):
            src = hf_dir / cls_name
            if src.exists():
                for f in sorted(src.glob("*.jpg"))[:pos_count // 4]:
                    shutil.copy2(f, leak_dir / "normal" / f"{cls_name}_{neg_idx:05d}.jpg")
                    neg_idx += 1

        remaining = pos_count - neg_idx
        for j in range(max(remaining, 0)):
            img = _metal_bg(224)
            img.save(leak_dir / "normal" / f"gen_{j:05d}.jpg", quality=90)

        _log_ds("leak", leak_dir)
    data_dirs["leak"] = leak_dir

    # ── General Agent ──────────────────────────────────────────────
    general_dir = DATA_ROOT / "general"
    if not (general_dir / "defective").exists() or len(list((general_dir / "defective").glob("*"))) < 10:
        logger.info("  Preparing general defect dataset ...")
        (general_dir / "defective").mkdir(parents=True, exist_ok=True)
        (general_dir / "non_defective").mkdir(parents=True, exist_ok=True)

        # Defective: all HF classes
        idx = 0
        for cls_name in ("crack", "corrosion", "slippage"):
            src = hf_dir / cls_name
            if src.exists():
                for f in sorted(src.glob("*.jpg")):
                    shutil.copy2(f, general_dir / "defective" / f"{cls_name}_{idx:05d}.jpg")
                    idx += 1

        pos_count = len(list((general_dir / "defective").glob("*.jpg")))

        # Non-defective: clean metal generated
        for j in range(pos_count):
            img = _metal_bg(224)
            img.save(general_dir / "non_defective" / f"gen_{j:05d}.jpg", quality=90)

        _log_ds("general", general_dir)
    data_dirs["general"] = general_dir

    return data_dirs


def _log_ds(name, d):
    """Log dataset size."""
    total = 0
    for sub in d.iterdir():
        if sub.is_dir():
            n = len(list(sub.glob("*.jpg")))
            logger.info(f"    {sub.name}: {n} images")
            total += n
    logger.info(f"  [{name}] Total: {total}")


# ═══════════════════════════════════════════════════════════════════════
# 2. Synthetic Helpers (for normals & leak augmentation)
# ═══════════════════════════════════════════════════════════════════════

def _metal_bg(size=224) -> Image.Image:
    base = random.randint(90, 170)
    arr = np.full((size, size, 3), base, dtype=np.uint8)
    noise = np.random.normal(0, 12, (size, size, 3)).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for _ in range(random.randint(3, 12)):
        y = random.randint(0, size - 1)
        thickness = random.randint(1, 3)
        val = random.randint(-25, 25)
        for dy in range(thickness):
            yy = min(y + dy, size - 1)
            arr[yy, :] = np.clip(arr[yy, :].astype(np.int16) + val, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))
    brightness = random.uniform(0.5, 1.4)
    return Image.fromarray(np.clip(np.array(img).astype(np.float32) * brightness, 0, 255).astype(np.uint8))


def _generate_leak_image(size=224) -> Image.Image:
    img = _metal_bg(size)
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(1, 4)):
        cx, cy = random.randint(20, size-20), random.randint(20, size-20)
        w, h = random.randint(10, 50), random.randint(15, 60)
        color = (random.randint(20, 60), random.randint(20, 50), random.randint(15, 40))
        draw.ellipse([(cx-w, cy-h), (cx+w, cy+h)], fill=color)
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.5, 3.0)))
    arr = np.array(img).astype(np.float32) * random.uniform(0.5, 0.85)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ═══════════════════════════════════════════════════════════════════════
# 3. Training
# ═══════════════════════════════════════════════════════════════════════

AGENTS = {
    "crack": {
        "model": "efficientnet_b0", "classes": ["normal", "crack"],
        "num_classes": 2, "epochs": 12, "batch_size": 32, "lr": 1e-3, "img_size": 224,
    },
    "corrosion": {
        "model": "mobilenet_v3_small", "classes": ["normal", "corrosion"],
        "num_classes": 2, "epochs": 15, "batch_size": 32, "lr": 1e-3, "img_size": 224,
    },
    "leak": {
        "model": "resnet18", "classes": ["normal", "leak"],
        "num_classes": 2, "epochs": 15, "batch_size": 32, "lr": 1e-3, "img_size": 224,
    },
    "general": {
        "model": "efficientnet_b0", "classes": ["non_defective", "defective"],
        "num_classes": 2, "epochs": 12, "batch_size": 32, "lr": 5e-4, "img_size": 224,
    },
}


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, class_names, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        for cls in class_names:
            d = Path(root_dir) / cls
            if d.exists():
                for p in sorted(d.glob("*")):
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                        self.samples.append((str(p), self.class_to_idx[cls]))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_model(name, nc):
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, nc)
    elif name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, nc)
    elif name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, nc)
    else:
        raise ValueError(name)
    return m


def get_transforms(sz, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((sz, sz)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def train_agent(agent_name: str, data_dir: Path) -> dict:
    cfg = AGENTS[agent_name]
    model_dir = MODEL_ROOT / agent_name
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"  TRAINING [{agent_name.upper()}] — {cfg['model']} on {DEVICE}")
    logger.info(f"{'='*60}")

    full_ds = ImageFolderDataset(data_dir, cfg["classes"],
                                  transform=get_transforms(cfg["img_size"], True))
    if len(full_ds) == 0:
        logger.error(f"  No samples for [{agent_name}] in {data_dir}")
        return {"agent": agent_name, "error": "No data"}

    train_n = int(len(full_ds) * 0.8)
    val_n = len(full_ds) - train_n
    train_ds, val_ds = random_split(full_ds, [train_n, val_n])

    # Proper val set without augmentation
    val_ds_clean = ImageFolderDataset(data_dir, cfg["classes"],
                                      transform=get_transforms(cfg["img_size"], False))
    val_ds_clean.samples = [val_ds_clean.samples[i] for i in val_ds.indices]

    # Weighted sampler for class balance
    train_labels = [full_ds.samples[i][1] for i in train_ds.indices]
    counts = np.bincount(train_labels, minlength=cfg["num_classes"]).astype(np.float64)
    w = 1.0 / (counts + 1e-6)
    sampler = WeightedRandomSampler([w[l] for l in train_labels], len(train_labels), replacement=True)

    train_dl = DataLoader(train_ds, cfg["batch_size"], sampler=sampler, num_workers=2,
                          pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(val_ds_clean, cfg["batch_size"], shuffle=False, num_workers=2,
                        pin_memory=True, persistent_workers=True)

    logger.info(f"  Dataset: {train_n} train / {val_n} val ({len(full_ds)} total)")

    model = build_model(cfg["model"], cfg["num_classes"]).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    criterion = FocalLoss(gamma=2.0)

    best_val, patience, pctr, final_ep = 0.0, 6, 0, 0

    for ep in range(1, cfg["epochs"] + 1):
        final_ep = ep
        # Train
        model.train()
        t_loss, t_ok, t_tot = 0.0, 0, 0
        for imgs, labs in train_dl:
            imgs, labs = imgs.to(DEVICE, non_blocking=True), labs.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(imgs)
            loss = criterion(out, labs)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * imgs.size(0)
            t_ok += out.argmax(1).eq(labs).sum().item()
            t_tot += imgs.size(0)
        scheduler.step()

        # Val
        model.eval()
        v_ok, v_tot = 0, 0
        with torch.no_grad():
            for imgs, labs in val_dl:
                imgs, labs = imgs.to(DEVICE, non_blocking=True), labs.to(DEVICE, non_blocking=True)
                v_ok += model(imgs).argmax(1).eq(labs).sum().item()
                v_tot += imgs.size(0)

        t_acc = 100 * t_ok / max(t_tot, 1)
        v_acc = 100 * v_ok / max(v_tot, 1)

        star = ""
        if v_acc > best_val:
            best_val = v_acc
            pctr = 0
            star = "  ★ BEST"
            torch.save({
                "model_state_dict": model.state_dict(),
                "agent": agent_name, "model_name": cfg["model"],
                "num_classes": cfg["num_classes"], "class_names": cfg["classes"],
                "best_val_acc": best_val, "epoch": ep,
            }, model_dir / f"{agent_name}_best.pt")
        else:
            pctr += 1

        logger.info(f"  Epoch {ep:2d}/{cfg['epochs']}  |  "
                     f"Loss: {t_loss/max(t_tot,1):.4f}  |  "
                     f"Train: {t_acc:.1f}%  |  Val: {v_acc:.1f}%{star}")

        if pctr >= patience:
            logger.info(f"  Early stopping at epoch {ep}")
            break

    # Benchmark
    model.eval()
    dummy = torch.randn(1, 3, cfg["img_size"], cfg["img_size"]).to(DEVICE)
    for _ in range(5):
        with torch.no_grad(): model(dummy)
    if DEVICE == "cuda": torch.cuda.synchronize()
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        with torch.no_grad(): model(dummy)
        if DEVICE == "cuda": torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    lat = np.median(times)

    logger.info(f"  ✓ Best val accuracy: {best_val:.1f}%")
    logger.info(f"  ✓ Inference latency: {lat:.1f} ms ({DEVICE})")
    logger.info(f"  ✓ Model saved to: {model_dir / f'{agent_name}_best.pt'}")

    return {"agent": agent_name, "model": cfg["model"],
            "best_val_acc": round(best_val, 2), "latency_ms": round(lat, 1),
            "epochs_trained": final_ep}


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("  ORAM – Full Setup & Training (REAL data)")
    logger.info("=" * 60)
    logger.info(f"  Device:     {DEVICE}")
    if DEVICE == "cuda":
        logger.info(f"  GPU:        {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    logger.info(f"  PyTorch:    {torch.__version__}")
    logger.info(f"  Data root:  {DATA_ROOT}")
    logger.info(f"  Model root: {MODEL_ROOT}")
    logger.info("=" * 60)

    t0 = time.time()

    # Step 1: Download real data
    logger.info("\n[1/3] Downloading real datasets from HuggingFace ...")
    hf_dir = download_hf_dataset()

    # Step 2: Prepare agent datasets
    logger.info("\n[2/3] Preparing agent-specific datasets ...")
    data_dirs = prepare_agent_datasets(hf_dir)

    logger.info("\n  Dataset summary:")
    for name, dd in data_dirs.items():
        total = sum(len(list(sub.glob("*.jpg"))) for sub in dd.iterdir() if sub.is_dir())
        logger.info(f"    [{name:10s}]  {total:5d} images")

    # Step 3: Train
    logger.info("\n[3/3] Training all agents ...")
    results = {}
    for agent in AGENTS:
        results[agent] = train_agent(agent, data_dirs[agent])

    elapsed = time.time() - t0

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)\n")
    for name, r in results.items():
        if "error" in r:
            logger.info(f"  [{name:10s}]  ERROR: {r['error']}")
        else:
            logger.info(f"  [{name:10s}]  acc={r['best_val_acc']:.1f}%  "
                         f"latency={r['latency_ms']:.1f}ms  model={r['model']}")
    logger.info(f"\n  Models saved in: {MODEL_ROOT}")

    # Save JSON
    with open(MODEL_ROOT / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Results JSON:    {MODEL_ROOT / 'training_results.json'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
