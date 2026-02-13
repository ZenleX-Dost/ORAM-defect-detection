# Comprehensive Dataset Guide for TGV Undercarriage Anomaly Detection
## Training Data for Railway Inspection AI Models

---

## 🎯 OVERVIEW

This document provides a curated list of publicly available datasets for training your anomaly detection models for the TGV inspection robot project. The datasets are organized by anomaly type and include download links, descriptions, and usage recommendations.

---

## 📊 DATASET CATEGORIES

### 1. CRACK DETECTION DATASETS

#### 1.1 SDNET2018 - Concrete Crack Detection (★★★★★ HIGHLY RECOMMENDED)
**Description:** One of the most comprehensive crack detection datasets available
- **Size:** 56,000+ images (256x256 pixels)
- **Content:** Cracked and non-cracked concrete surfaces (bridge decks, walls, pavements)
- **Crack Width Range:** 0.06mm to 25mm
- **Features:** Various conditions including shadows, surface roughness, debris, holes
- **Format:** RGB .jpg images
- **Categories:** Organized into Cracked/Non-cracked folders
- **Link:** https://digitalcommons.usu.edu/all_datasets/48/
- **License:** Creative Commons Attribution 4.0

**Why Use This:**
- Large dataset with diverse conditions
- Includes fine cracks similar to train undercarriage
- Well-annotated and widely used in research
- Perfect for transfer learning

**Recommended Usage:**
```python
# Load SDNET2018 for pre-training
base_model = train_on_sdnet2018()
# Fine-tune on your railway-specific images
fine_tuned_model = fine_tune(base_model, railway_images)
```

---

#### 1.2 Concrete Crack Images for Classification (METU Dataset)
**Description:** Binary classification dataset for crack detection
- **Size:** 40,000 images (20,000 positive, 20,000 negative)
- **Resolution:** 227 x 227 pixels RGB
- **Source:** Various METU Campus Buildings
- **Conditions:** Variance in surface finish and illumination
- **Link:** https://data.mendeley.com/datasets/5y9wdsg2zt/2
- **DOI:** http://dx.doi.org/10.17632/5y9wdsg2zt.2

**Ideal For:**
- Binary classification (crack/no-crack)
- Quick prototyping
- Training baseline models

---

#### 1.3 Surface Crack Detection Dataset (Kaggle)
**Description:** General surface crack detection
- **Link:** https://www.kaggle.com/datasets/arunrk7/surface-crack-detection
- **Content:** Mixed surface types including concrete and metal
- **Format:** Ready for immediate use

---

#### 1.4 Crack500 Dataset
**Description:** 500 images of cracked pavement with pixel-level annotations
- **Features:** Challenging real-world scenarios
- **Best For:** Segmentation tasks
- **Note:** Frequently referenced in crack detection research

---

#### 1.5 Rail-5k Dataset
**Description:** Railway-specific crack detection dataset
- **Size:** 5,000+ high-quality images from Chinese railways
- **Annotated:** 1,100 images labeled by railway experts
- **Defect Types:** 13 most common rail defects
- **Link:** https://arxiv.org/abs/2106.14366
- **Format:** Fully-supervised and semi-supervised learning settings

**CRITICAL FOR YOUR PROJECT:**
This is the most relevant dataset for railway applications!

---

### 2. RAILWAY-SPECIFIC DATASETS

#### 2.1 Railway Track Surface Faults Dataset (PMC)
**Description:** Real-world railway track defects
- **Size:** 7 fault conditions
- **Types:** Grooves, Joints, Cracks, Flakings, Shellings, Spallings, Squats
- **Camera:** EKENH9R cameras mounted on inspection vehicle
- **Conditions:** Various environmental and lighting scenarios
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10828558/
- **Purpose:** Low-cost visual inspection systems

**Defect Categories:**
1. Grooves - Surface wear patterns
2. Joints - Connection point defects
3. Cracks - Surface fractures
4. Flakings - Surface material separation
5. Shellings - Shell-like surface damage
6. Spallings - Chipping/fragmentation
7. Squats - Localized plastic deformation

---

#### 2.2 FaultSeg - Train Wheel Defect Dataset
**Description:** Specifically for railway wheel inspection
- **Size:** 829 manually annotated images
- **Defect Types:** Cracks/Scratches, Shelling, Discoloration
- **Model Tested:** YOLOv9 instance segmentation
- **Accuracy:** ~87% baseline
- **Link:** https://www.nature.com/articles/s41597-025-04557-0
- **Published:** February 2025 (very recent!)

**Classes:**
- Cracks/Scratches
- Shelling (surface material loss)
- Discoloration (heat/friction marks)

---

#### 2.3 Railway Track Fault Detection (Kaggle)
**Description:** Ready-to-use railway defect dataset
- **Link:** https://www.kaggle.com/datasets/salmaneunus/railway-track-fault-detection
- **Resolution:** 224x224 pixels (resized version available)
- **Format:** Organized for deep learning

---

### 3. CORROSION DETECTION DATASETS

#### 3.1 Roboflow Universe - Corrosion Datasets (★★★★★)
**Description:** Multiple corrosion detection datasets in one platform
- **Link:** https://universe.roboflow.com/search?q=class:corrosion
- **Datasets Available:**
  - Metal surface corrosion (multiple severity levels)
  - Pipeline corrosion
  - Structural corrosion
  - Ship/marine corrosion
- **Formats:** COCO, YOLO, Pascal VOC, TensorFlow
- **Features:** Pre-annotated with bounding boxes

**Corrosion Classes Available:**
- Low-corrosion
- Medium-corrosion
- High-corrosion
- Pitting corrosion
- Stress corrosion cracking
- Galvanic corrosion
- Crevice corrosion
- Oxidation corrosion

**Easy Integration:**
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace().project("corrosion-detection")
dataset = project.version(1).download("yolov8")
```

---

#### 3.2 Metal Surface Corrosion Dataset (CBG-YOLOv5s)
**Description:** Coastal metal facility corrosion
- **Size:** 6,000 images (600 original + augmentation)
- **Severity Levels:** 3 classes (light, moderate, severe)
- **Features:** Texture, color, and depth variations
- **Link:** https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0300440
- **Model:** Optimized YOLOv5s with CBAM attention

---

#### 3.3 Image-based Corrosion Detection Dataset (Mendeley)
**Description:** General corrosion classification
- **Size:** 1,819 images
- **Classes:** Corrosion (990 images) and No Corrosion (829 images)
- **Sources:** Internet scraping from 8 corrosion categories
- **Link:** https://data.mendeley.com/datasets/tbjn6p2gn9/1
- **DOI:** http://dx.doi.org/10.17632/tbjn6p2gn9.1

**Corrosion Categories:**
1. Steel Corrosion/Rust
2. Ships Corrosion
3. Ship Propellers Corrosion
4. Cars Corrosion
5. Oil & Gas Pipelines Corrosion
6. Concrete Rebar Corrosion
7. Water/Oil Tanks Corrosion
8. Stainless Steel Corrosion

**GitHub Repository:**
https://github.com/pjsun2012/Phase5_Capstone-Project

---

### 4. LEAK/FLUID DETECTION DATASETS

#### 4.1 Oil Leak Detection Dataset (Roboflow)
**Description:** Industrial oil leak detection
- **Size:** 68 open source images
- **Link:** https://universe.roboflow.com/tryout/oil-leak
- **Format:** Object detection annotations
- **License:** MIT

---

#### 4.2 Thermal Liquid Leak Detection Dataset
**Description:** Thermal imaging for leak detection
- **Technology:** YOLOv8 and RT-DETR models
- **Source:** Industry collaboration (confidential dataset)
- **Performance:** 90.8% precision, 89.9% recall
- **Link:** https://arxiv.org/html/2312.10980v1
- **Note:** Dataset not publicly available but methodology is documented

**Alternative Approach:**
Since thermal leak datasets are limited, you can:
1. Use color-based detection (oil has distinct color signatures)
2. Collect your own images with fluorescein tracer
3. Use synthetic data generation

---

#### 4.3 Subsea Pipeline Leak Detection Dataset
**Description:** Underwater leak detection (gas and liquid)
- **Environment:** Water tank experiments
- **Tracer:** Fluorescein dye for visibility
- **Link:** https://www.mdpi.com/2077-1312/13/9/1683
- **Models:** VGG16, ResNet50, InceptionV3, DenseNet121
- **Performance:** F1-scores above 0.80

**Note:** While subsea-focused, the principles apply to detecting fluid leaks

---

### 5. GENERAL DEFECT DETECTION DATASETS

#### 5.1 NEU Surface Defect Database
**Description:** Steel surface defects
- **Size:** 1,800 grayscale images
- **Types:** 6 kinds of typical surface defects
- **Resolution:** 200×200 pixels
- **Use:** General metal defect detection

---

#### 5.2 MVTec Anomaly Detection Dataset
**Description:** Industrial inspection benchmark
- **Categories:** 15 object and texture categories
- **Size:** 5,000+ images
- **Defects:** Scratches, dents, contamination, holes
- **Link:** https://www.mvtec.com/company/research/datasets/mvtec-ad
- **License:** Free for research

**Perfect For:**
- Anomaly detection algorithm development
- Transfer learning base
- Benchmarking your model

---

## 🛠️ PRACTICAL RECOMMENDATIONS

### Strategy 1: Multi-Dataset Training Approach

```python
# Recommended training pipeline

# Phase 1: Pre-train on large general dataset
model = pretrain_on_sdnet2018()  # 56k crack images

# Phase 2: Transfer learning on railway-specific data
model = fine_tune_on_rail5k(model)  # Railway defects

# Phase 3: Fine-tune on your custom data
model = final_tuning_on_tgv_images(model)  # Your collected images

# Phase 4: Ensemble with specialized detectors
crack_model = model
corrosion_model = train_corrosion_detector()
leak_model = train_leak_detector()

# Combine predictions
final_detection = ensemble([crack_model, corrosion_model, leak_model])
```

---

### Strategy 2: Data Augmentation

Since you have limited real TGV undercarriage images, use aggressive augmentation:

```python
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, Rotate, 
    RandomBrightnessContrast, GaussianBlur, 
    GridDistortion, OpticalDistortion, Blur,
    CLAHE, RandomGamma, MotionBlur
)

train_transform = Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.3),
    Rotate(limit=15, p=0.5),
    RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
    GaussianBlur(blur_limit=(3, 7), p=0.3),
    MotionBlur(blur_limit=7, p=0.3),  # Simulate robot movement
    GridDistortion(p=0.2),
    OpticalDistortion(p=0.2),
    CLAHE(p=0.3),  # Enhance contrast for low-light conditions
    RandomGamma(gamma_limit=(80, 120), p=0.3),
])
```

---

### Strategy 3: Synthetic Data Generation

Use Stable Diffusion or DALL-E to generate additional training images:

**Prompts for Synthetic Data:**
```
"Cracked metal surface under train, industrial inspection, dark environment"
"Corroded steel bogie component, railway maintenance, close-up"
"Oil leak on metal surface, industrial equipment, high detail"
"Rust and corrosion on train undercarriage, professional inspection photo"
```

**Tools:**
- Stable Diffusion XL
- Midjourney (commercial license)
- DALL-E 3
- Roboflow's Synthetic Data Generator

---

### Strategy 4: Active Learning Pipeline

```python
# Start with limited labeled data
initial_model = train_baseline(small_labeled_dataset)

# Iterative improvement
for iteration in range(10):
    # Use model to predict on unlabeled data
    predictions = model.predict(unlabeled_pool)
    
    # Select most uncertain/valuable samples
    uncertain_samples = select_uncertain(predictions, method='entropy')
    
    # Manually label these samples
    new_labels = human_annotation(uncertain_samples)
    
    # Add to training set and retrain
    training_data.add(uncertain_samples, new_labels)
    model = retrain(training_data)
```

---

## 📥 DOWNLOAD SCRIPTS

### Script 1: Download Multiple Datasets

```bash
#!/bin/bash

# Create dataset directory
mkdir -p datasets/{cracks,corrosion,railway,leaks}

# SDNET2018 (visit website and download manually)
echo "Download SDNET2018 from: https://digitalcommons.usu.edu/all_datasets/48/"

# Kaggle datasets (requires Kaggle API)
pip install kaggle

# Railway Track Fault Detection
kaggle datasets download -d salmaneunus/railway-track-fault-detection -p datasets/railway/

# Surface Crack Detection
kaggle datasets download -d arunrk7/surface-crack-detection -p datasets/cracks/

# Railway Track Fault Detection Resized
kaggle datasets download -d gpiosenka/railway-track-fault-detection-resized-224-x-224 -p datasets/railway/

# Unzip all
find datasets/ -name "*.zip" -exec unzip {} -d {}_extracted \;

echo "Dataset download complete!"
```

---

### Script 2: Roboflow Downloader

```python
from roboflow import Roboflow

# Initialize Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")

# Download corrosion datasets
corrosion_project = rf.workspace("roboflow-100").project("corrosion-bi3q3")
corrosion_dataset = corrosion_project.version(2).download("yolov8")

# Download oil leak dataset
leak_project = rf.workspace("tryout").project("oil-leak")
leak_dataset = leak_project.version(1).download("yolov8")

print("Roboflow datasets downloaded successfully!")
```

---

## 🎓 DATASET PREPROCESSING TIPS

### 1. Image Normalization

```python
import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Standardize images for training
    """
    img = cv2.imread(image_path)
    
    # Resize to standard size
    img = cv2.resize(img, (640, 640))
    
    # CLAHE for contrast enhancement (important for low-light conditions)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    
    return img
```

---

### 2. Handle Class Imbalance

```python
from imblearn.over_sampling import SMOTE
from collections import Counter

# Check class distribution
print(f"Original distribution: {Counter(y_train)}")

# Balance using SMOTE or class weights
class_weights = compute_class_weight('balanced', 
                                     classes=np.unique(y_train), 
                                     y=y_train)

# Or oversample minority class
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

---

## 📊 DATASET STATISTICS SUMMARY

| Dataset | Size | Type | Railway-Specific | Recommended Use |
|---------|------|------|------------------|-----------------|
| SDNET2018 | 56k | Cracks | ❌ | Pre-training |
| Rail-5k | 5k+ | Rail defects | ✅ | Fine-tuning |
| FaultSeg | 829 | Wheel defects | ✅ | Specialized |
| Railway Track Faults | Varies | Multiple | ✅ | Main training |
| Corrosion (Roboflow) | 1k+ | Corrosion | ❌ | Corrosion detection |
| METU Cracks | 40k | Cracks | ❌ | Augmentation |
| Oil Leak | 68 | Leaks | ❌ | Leak detection |

---

## 🔄 CREATING YOUR CUSTOM DATASET

### Field Data Collection Guide

**Equipment Needed:**
1. **Camera:** 
   - Minimum 12MP resolution
   - Wide-angle lens (110°+)
   - Good low-light performance
   - Recommendation: GoPro Hero 11 or similar

2. **Lighting:**
   - Portable LED panels (400-800 lux)
   - Diffused light to minimize shadows

3. **Stabilization:**
   - Camera mount for consistent positioning
   - Motion blur prevention

**Collection Protocol:**
```
1. Capture images every 0.5 meters along undercarriage
2. Take 3 angles per location: front, side, oblique
3. Ensure lighting consistency
4. Label location metadata (bogie number, position)
5. Minimum 1000 images per defect category
6. Include 3000+ "normal" baseline images
```

---

### Annotation Tools

**Recommended:**

1. **Label Studio** (Free, Open Source)
   - Web-based interface
   - Multiple annotation types
   - Team collaboration
   - Link: https://labelstud.io/

2. **CVAT** (Free, Open Source)
   - Advanced features
   - Video annotation
   - AI-assisted labeling
   - Link: https://cvat.ai/

3. **Roboflow** (Free tier available)
   - Easiest to use
   - Built-in augmentation
   - Model training integrated
   - Link: https://roboflow.com/

4. **Labelbox** (Commercial)
   - Professional features
   - Quality control tools
   - Team management

---

## 🚀 QUICK START TRAINING PIPELINE

```python
# Complete training pipeline

import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
import albumentations as A

# 1. Load multiple datasets
sdnet_data = load_dataset('SDNET2018')
rail_data = load_dataset('Rail-5k')
custom_data = load_dataset('tgv_custom')

# 2. Combine datasets with appropriate sampling
combined_data = CombinedDataset([
    (sdnet_data, 0.3),   # 30% from SDNET
    (rail_data, 0.5),    # 50% from Rail-5k
    (custom_data, 0.2)   # 20% from custom
])

# 3. Data augmentation
train_transform = A.Compose([
    A.Resize(640, 640),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.8),
    A.GaussianBlur(blur_limit=(3,7), p=0.3),
    A.CLAHE(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
])

# 4. Create model
model = models.efficientnet_b3(pretrained=True)
num_classes = 6  # normal, crack, leak, corrosion, missing_part, wear
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.classifier[1].in_features, 512),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(512, num_classes)
)

# 5. Training
train_loader = DataLoader(combined_data, batch_size=32, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Train for 50 epochs with early stopping
best_val_loss = float('inf')
patience = 10

for epoch in range(50):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss = validate(model, val_loader, criterion)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print("Early stopping triggered")
        break

print("Training complete!")
```

---

## 📚 ADDITIONAL RESOURCES

### Research Papers for Reference:

1. **SDNET2018 Paper:**
   "SDNET2018: An annotated image dataset for non-contact concrete crack detection"
   
2. **Rail-5k Paper:**
   "Rail-5k: a Real-World Dataset for Rail Surface Defects Detection"
   arXiv:2106.14366

3. **YOLOv8 for Crack Detection:**
   "Concrete Surface Crack Detection Algorithm Based on Improved YOLOv8"

4. **Transfer Learning:**
   "Comparison of deep convolutional neural networks and edge detectors for image-based crack detection"

### Online Courses:

1. **Deep Learning for Computer Vision** (Stanford CS231n)
2. **Practical Deep Learning for Coders** (fast.ai)
3. **TensorFlow Developer Certificate** (Coursera)

### Communities:

1. **Papers with Code** - Track SOTA models
2. **Roboflow Forum** - Computer vision community
3. **Kaggle Discussions** - Dataset and competition discussions

---

## ⚠️ IMPORTANT NOTES

### Legal Considerations:

1. **Dataset Licenses:** Always check license terms before use
2. **Commercial Use:** Some datasets are research-only
3. **Attribution:** Cite datasets in your documentation
4. **Privacy:** Ensure no personal/confidential data in custom images

### Performance Tips:

1. **Start Small:** Train on 1000 images first, validate approach
2. **Iterative Improvement:** Add data based on model weaknesses
3. **Domain Adaptation:** Railway data performs better than generic
4. **Ensemble Methods:** Combine multiple models for robustness

### Common Pitfalls to Avoid:

❌ Training on one dataset type only
❌ Ignoring class imbalance
❌ Not validating on real TGV images
❌ Over-augmenting (can hurt performance)
❌ Forgetting to normalize images consistently

✅ Use diverse datasets
✅ Balance classes or use weighted loss
✅ Create TGV-specific validation set
✅ Use moderate, realistic augmentation
✅ Standardize preprocessing pipeline

---

## 🎯 FINAL RECOMMENDATIONS

**For Your TGV Project, prioritize:**

1. **Rail-5k** - Most relevant railway dataset
2. **SDNET2018** - Best crack detection dataset
3. **Roboflow Corrosion** - Ready-to-use corrosion data
4. **Custom TGV images** - Collect 500+ annotated images minimum

**Training Strategy:**
```
Week 1-2: Collect custom TGV images (500+)
Week 3: Download and prepare external datasets
Week 4-5: Train baseline on SDNET2018 + Rail-5k
Week 6-7: Fine-tune on custom TGV data
Week 8: Test and iterate
Week 9-10: Optimize and deploy
```

**Success Metrics:**
- Crack Detection: >85% F1-score
- Corrosion Detection: >80% F1-score
- Leak Detection: >75% F1-score
- False Positive Rate: <15%
- Inference Speed: <100ms per frame

---

## 📞 SUPPORT & UPDATES

This guide will be updated as new datasets become available. Check:
- Papers with Code: https://paperswithcode.com/datasets
- Roboflow Universe: https://universe.roboflow.com/
- Kaggle Datasets: https://www.kaggle.com/datasets

**Good luck with your training! 🚀**

---

*Last Updated: February 2026*
*Author: Claude (Anthropic)*
*Project: Prix InnovAM'26 TGV Inspection Robot*
