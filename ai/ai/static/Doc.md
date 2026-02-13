# ORAM – Plateforme d'Inspection Intelligente
*Documentation Technique et Guide Utilisateur*

---

## 1. Vue d'ensemble (Overview)

**ORAM** est une solution complète d'inspection automatisée pour les châssis de TGV, utilisant l'intelligence artificielle pour la détection d'anomalies (fissures, corrosion, fuites) en temps réel. Le système combine la vision par ordinateur classique, le Deep Learning, et la segmentation avancée (SAM 2) pour fournir une analyse précise et robuste, même dans des conditions d'éclairage difficiles.

### Fonctionnalités Clés
- **Détection Multi-Agents** : Modèles spécialisés pour Fissures (Crack), Corrosion, et Fuites (Leak).
- **Segmentation SAM 2** : Isolation précise des défauts au pixel près (Segment Anything Model 2).
- **Robustesse à l'Éclairage** : Prétraitement adaptatif (CLAHE, correction gamma, balance des blancs) pour les environnements sous-châssis sombres.
- **Interface Temps Réel** : Dashboard web moderne pour le contrôle caméra, le streaming vidéo, et la visualisation des alertes.
- **Entraînement Continu** : Pipeline intégré pour réentraîner les modèles sur de nouvelles données annotées.

---

## 2. Architecture Technique

Le projet est structuré en deux parties principales : un backend Python performant et un frontend web réactif.

### 2.1 Backend (Python / FastAPI)
- **API REST** : Gestion du système, upload d'images, configuration (`/api/status`, `/api/analyze`).
- **WebSocket** : Streaming vidéo temps réel avec latence minimale (`/api/stream`).
- **Orchestrateur IA** : Gère le chargement des modèles, l'inférence asynchrone, et le pipeline de traitement.
- **Pipeline de Données** : Téléchargement automatique des datasets (HuggingFace) et préparation des lots d'entraînement via CUDA.

### 2.2 Frontend (HTML5 / CSS3 / JS)
- **Design Industriel** : Thème inspiré des interfaces HMI/SCADA (palette SNCF : Terracotta, Blanc Cassé, Gris Acier).
- **Visualisation** : Canvas interactif pour l'affichage des bounding boxes et masques de segmentation.
- **Contrôles** : Paramétrage des seuils de confiance, activation des filtres, et gestion des caméras (RTSP/MJPEG/USB).

### 2.3 Stack Technologique
- **Langage** : Python 3.9+, JavaScript (ES6+)
- **IA/ML** : PyTorch, TorchVision, Ultralytics (YOLO/SAM), HuggingFace Datasets
- **Web** : FastAPI, Uvicorn, WebSockets
- **Traitement Image** : OpenCV, Pillow, NumPy
- **Matériel** : Optimisé pour GPU NVIDIA (CUDA) avec fallback CPU.

---

## 3. Installation et Configuration

### Prérequis
- Python 3.8 ou supérieur
- Carte graphique NVIDIA (recommandé pour l'entraînement)
- `pip` mis à jour

### Installation
1. **Cloner le projet**
   ```bash
   git clone https://github.com/votre-repo/oram.git
   cd oram
   ```

2. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```
   *Note : Le script installera automatiquement PyTorch, FastAPI, et les bibliothèques de traitement d'image.*

3. **Lancer le serveur**
   ```bash
   python -m ai.server
   ```
   L'interface sera accessible sur `http://localhost:8000`.

---

## 4. Pipeline d'Entraînement (Training)

Le système inclut un module d'entraînement automatisé capable de télécharger des données réelles et d'entraîner des modèles spécifiques.

### Script de Configuration (`run_setup.py`)
Ce script exécute le pipeline complet :
1. **Téléchargement** : Récupère le dataset `Francesco/corrosion-bi3q3` depuis HuggingFace (images réelles de corrosion/fissures).
2. **Préparation** : Crée 4 jeux de données binaires équilibrés :
   - **Crack Agent** : Fissures vs Métal sain
   - **Corrosion Agent** : Rouille vs Métal sain
   - **Leak Agent** : Fuites (simulées/réelles) vs Autre
   - **General Agent** : Tout défaut vs Sain
3. **Entraînement** : Lance l'entraînement sur GPU avec :
   - *Focal Loss* pour gérer le déséquilibre des classes.
   - *Data Augmentation* (rotations, éclairage, bruit).
   - *Early Stopping* pour éviter le surapprentissage.

### Résultats des Modèles (Benchmark RTX 4050)
| Agent | Modèle | Précision (Val) | Latence |
|-------|--------|:--------------:|:-------:|
| **Crack** | EfficientNet-B0 | **96.8%** | 28.8ms |
| **Corrosion** | MobileNet V3 | **94.5%** | 17.8ms |
| **Leak** | ResNet18 | **98.1%** | 8.6ms |
| **General** | EfficientNet-B0 | **100%** | 30.5ms |

---

## 5. Guide d'Utilisation

### 5.1 Connexion Caméra
1. Allez dans le panneau **Camera Connection**.
2. Entrez l'URL du flux (ex: `rtsp://192.168.1.50:554/stream` ou `0` pour webcam locale).
3. Sélectionnez le protocole ou laissez "Auto Detect".
4. Cliquez sur **Connect**.

### 5.2 Analyse Temps Réel
1. Une fois connecté, cliquez sur **Start Inspection**.
2. Le flux vidéo s'affiche avec les détections en surimpression.
   - **Cadres colorés** : Indiquent la zone du défaut et sa sévérité (Rouge=Critique, Orange=Élevé).
   - **Masques** : Si SAM 2 est activé, la forme exacte du défaut est détourée.
3. Ajustez le curseur **Confidence Threshold** pour filtrer les détections incertaines.

### 5.3 Paramètres Avancés
- **SAM 2 Segmentation** : Active/désactive la segmentation fine (plus lent mais plus précis).
- **Light Preprocessing** : Active les filtres d'amélioration d'image (utile pour les zones sombres).

---

## 6. Structure du Code

```
ai/
├── server.py            # Point d'entrée du backend FastAPI
├── run_setup.py         # Script d'entraînement et de téléchargement des données
├── detection.py         # Logique de détection et pipelines
├── training.py          # Gestionnaire d'entraînement (Trainer)
├── preprocessing.py     # Algorithmes de correction d'image
├── segmentation.py      # Wrapper pour SAM 2
├── datasets.py          # Gestion des données et chargeurs
└── static/              # Frontend (HTML/CSS/JS)
    ├── index.html       # Interface utilisateur
    ├── style.css        # Thème industriel (CSS)
    └── app.js           # Logique frontend (WebSocket, UI)
```

---

*Généré par l'Assistant IA ORAM - Février 2026*
