# YOLOv8 Object Detector

Détection d'objets en temps réel avec YOLOv8. Supporte trois modes : image, webcam et vidéo.
Interface graphique intégrée (Tkinter), overlay de statistiques temps réel, code structuré et commenté.

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Sommaire

- [Fonctionnalités](#fonctionnalités)
- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Architecture interne](#architecture-interne)
- [Principes mathématiques et physiques](#principes-mathématiques-et-physiques)
- [Description des fonctions](#description-des-fonctions)
- [Modèles disponibles](#modèles-disponibles)
- [Cas d'usage](#cas-dusage)
- [Références](#références)
- [Licence](#licence)

---

## Fonctionnalités

- Interface graphique de sélection du mode et de la source (Tkinter)
- Détection sur image, webcam ou vidéo
- Overlay temps réel : FPS, latence, nombre d'objets détectés
- Sauvegarde optionnelle du résultat annoté
- Filtrage par classe COCO (ex. : uniquement les personnes)
- Support CPU / CUDA / MPS (Apple Silicon)
- Compteur FPS par fenêtre glissante

---

## Structure du projet

```
yolo-detector/
├── src/
│   └── detector.py          # Moteur principal (YOLODetecteur, FPSCompteur, main)
├── outils/
│   ├── logpy.py             # Configuration des logs
│   └── visualizer.py        # Overlay statistiques (dessin_stats_overlay)
├── models/                  # Modèles .pt téléchargés automatiquement
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/votre-user/yolo-detector.git
cd yolo-detector

python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate

pip install -r requirements.txt
```

**Dépendances principales :**

```
ultralytics
opencv-python
numpy
tkinter   # inclus dans Python standard
```

---

## Utilisation

Au lancement, une interface graphique vous demande :

1. Le **mode** : `image`, `webcam`, ou `video`
2. La **source** : chemin vers un fichier (image ou vidéo), ou caméra `0`
3. Le **seuil de confiance** (défaut : `0.40`)
4. Le **seuil IoU NMS** (défaut : `0.45`)

```bash
python -m src.detecteur
```

Pour quitter la détection en flux (webcam ou vidéo) : appuyez sur `q`.

---

## Architecture interne

```
┌──────────────────────────────────────────────┐
│          Interface Tkinter (ask_parameters)  │
└──────────────────┬───────────────────────────┘
                   │
         ┌─────────▼──────────┐
         │   YOLODetecteur    │
         │  ┌──────────────┐  │
         │  │  YOLO model  │  │  
         │  └──────────────┘  │
         │  detect_image()    │
         │  detect_flux()     │
         └─────────┬──────────┘
                   │
       ┌───────────▼───────────┐
       │  dessin_stats_overlay │
       │  FPSCompteur          │
       └───────────────────────┘
```

---

## Principes mathématiques et physiques

### 1. Détection d'objets — YOLO ("You Only Look Once")

YOLO est une architecture de réseau de neurones convolutionnel (CNN) qui traite l'image entière
en une seule passe, contrairement aux méthodes à deux étages (R-CNN) qui proposent d'abord des
régions d'intérêt.

**Principe de la grille :**
L'image est divisée en une grille S x S. Chaque cellule prédit B boîtes englobantes et leurs
scores de confiance. Chaque boîte est définie par :

```
(x, y, w, h, confidence)
```

Où `(x, y)` est le centre de la boîte, `(w, h)` ses dimensions, et `confidence = P(objet) * IoU`.

**Références :**
- Article fondateur : https://arxiv.org/abs/1506.02640 (Redmon et al., 2015)
- YOLOv8 (Ultralytics) : https://docs.ultralytics.com
- Explication visuelle : https://www.v7labs.com/blog/yolo-object-detection


### 2. Non-Maximum Suppression (NMS)

Le modèle génère de nombreuses boîtes candidates qui se chevauchent pour un même objet.
La NMS élimine les doublons en conservant uniquement la boîte ayant le score le plus élevé
parmi celles dont l'IoU dépasse un seuil.

**Intersection over Union (IoU) :**

```
IoU = Aire(Intersection) / Aire(Union)
```

Un IoU élevé (proche de 1) indique deux boîtes qui se superposent fortement.
Le paramètre `iou` dans le code contrôle ce seuil (défaut : 0.45).

**Références :**
- https://en.wikipedia.org/wiki/Jaccard_index
- https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/

---

### 3. Seuil de confiance

Chaque détection est associée à un score de confiance :

```
score = P(classe | objet) * IoU(boite_predite, boite_reelle)
```

Seules les détections dont le score dépasse le paramètre `conf` (défaut : 0.40) sont conservées.
Un seuil trop bas augmente les faux positifs ; trop haut, les faux négatifs.

**Référence :**
- https://developers.google.com/machine-learning/crash-course/classification/thresholding

---

### 4. FPS et latence

Le compteur FPS utilise une fenêtre glissante de N frames pour lisser la mesure :

```
FPS = (N - 1) / (t_derniere_frame - t_premiere_frame)
```

La latence correspond au temps d'inférence du modèle sur une seule frame, mesurée en millisecondes
avec `time.perf_counter()`.

**Référence :**
- https://docs.python.org/3/library/time.html#time.perf_counter

---

### 5. Décodage des classes (COCO)

Le modèle est entraîné sur le dataset COCO (Common Objects in Context), qui contient 80 classes
d'objets courants (personne, voiture, chien, etc.). Chaque identifiant numérique est converti
en nom de classe via le dictionnaire `model.names`.

**Référence :**
- https://cocodataset.org
- Liste des 80 classes : https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/

---

## Description des fonctions

### `YOLODetecteur.__init__`

Initialise le modèle YOLO et les paramètres de détection.

```python
YOLODetecteur(model_path, conf, iou, classes, device)
```

| Paramètre | Description | Défaut |
|-----------|-------------|--------|
| `model_path` | Fichier du modèle `.pt` | `yolov8n.pt` |
| `conf` | Seuil de confiance minimum | `0.40` |
| `iou` | Seuil IoU pour la NMS | `0.45` |
| `classes` | Liste d'IDs COCO à détecter | `None` (tous) |
| `device` | Matériel d'inférence | `cpu` |

Le modèle est chargé depuis le hub Ultralytics ou en local si le fichier `.pt` est présent.

**Références :**
- https://docs.ultralytics.com/modes/predict/
- https://pytorch.org/docs/stable/tensor_attributes.html#torch.device

---

### `YOLODetecteur._predict`

Appelle le moteur d'inférence YOLOv8 sur une frame NumPy.

L'image traverse le réseau neuronal (backbone + neck + head) en une seule passe forward.
Le résultat contient les boîtes brutes avant filtrage NMS.

**Référence :**
- https://docs.ultralytics.com/reference/engine/results/


### `YOLODetecteur._annotation`

Génère une image BGR annotée (boîtes, labels, scores) à partir des résultats de détection,
en utilisant la méthode `.plot()` d'Ultralytics.

**Référence :**
- https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Results.plot



### `YOLODetecteur.detect_image`

Charge une image depuis le disque, exécute la détection, affiche le résultat annoté
et le sauvegarde optionnellement.

```python
detect_image(path: str, save: bool = False) -> np.ndarray
```

La lecture utilise OpenCV (`cv2.imread`) qui charge l'image au format BGR (Blue-Green-Red),
contrairement à la convention RGB de la plupart des bibliothèques.

**Références :**
- https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html
- https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html


### `YOLODetecteur.detect_flux`

Capture et traite un flux vidéo frame par frame (webcam ou fichier vidéo).

```python
detect_flux(source: int | str = 0, save: bool = False, output_path: str = None)
```

La capture est réalisée via `cv2.VideoCapture`, qui s'appuie sur les drivers V4L2 (Linux),
DirectShow (Windows) ou AVFoundation (macOS) selon la plateforme.

Pour la sauvegarde, `cv2.VideoWriter` encode le flux annoté avec le codec `mp4v` (MPEG-4 Part 2).

La boucle se termine proprement grâce au bloc `try / finally`, qui garantit la libération des
ressources (`cap.release()`, `writer.release()`) même en cas d'interruption.

**Références :**
- https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
- https://docs.opencv.org/4.x/dd/d9e/classcv_1_1VideoWriter.html
- https://fourcc.org/codecs.php


### `FPSCompteur`

Calcule le nombre de frames par seconde avec une fenêtre glissante de 30 frames.

```python
fps = FPSCompteur(window=30)
fps.tick()  # retourne le FPS courant
```

La fenêtre glissante évite les variations brutales dues à des frames occasionnellement lentes,
en moyennant le débit sur les N dernières frames plutôt que de mesurer frame à frame.

**Référence :**
- https://en.wikipedia.org/wiki/Moving_average


### `ask_parameters`

Interface graphique Tkinter permettant à l'utilisateur de saisir le mode, la source et les
paramètres de détection sans ligne de commande.

```python
mode, source, conf, iou = ask_parameters()
```

Utilise `simpledialog` pour les valeurs numériques et `filedialog` pour la sélection de fichiers.

**Référence :**
- https://docs.python.org/3/library/tkinter.html
- https://docs.python.org/3/library/dialog.html


### `main`

Point d'entrée principal. Enchaîne : récupération des paramètres, initialisation du détecteur,
et lancement du mode choisi.

```python
if __name__ == "__main__":
    main()
```

**Référence :**
- https://docs.python.org/3/library/__main__.html


## Modèles disponibles

| Modèle | Taille | mAP50-95 | Vitesse (CPU) |
|--------|--------|----------|---------------|
| yolov8n | 6 MB | 37.3 | rapide |
| yolov8s | 22 MB | 44.9 | moyen |
| yolov8m | 50 MB | 50.2 | lent |
| yolov8l | 83 MB | 52.9 | très lent |
| yolov8x | 130 MB | 53.9 | très très lent |

Le suffixe indique la taille : **n** (nano), **s** (small), **m** (medium), **l** (large), **x** (extra-large).
mAP50-95 est la métrique standard COCO : précision moyenne sur des IoU de 0.50 à 0.95.

**Référence :**
- https://docs.ultralytics.com/models/yolov8/#performance-metrics


## Cas d'usage

- Surveillance vidéo et comptage de personnes
- Inspection industrielle automatisée
- Aide à la conduite autonome
- Analyse de flux sportifs
- Robotique et vision embarquée


## Références

| Sujet | Lien |
|-------|------|
| Article YOLO original | https://arxiv.org/abs/1506.02640 |
| Documentation Ultralytics YOLOv8 | https://docs.ultralytics.com |
| Dataset COCO | https://cocodataset.org |
| Non-Maximum Suppression | https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/ |
| Intersection over Union | https://en.wikipedia.org/wiki/Jaccard_index |
| OpenCV VideoCapture | https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html |
| Tkinter (interface graphique) | https://docs.python.org/3/library/tkinter.html |
| NumPy ndarray | https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html |
| time.perf_counter (latence) | https://docs.python.org/3/library/time.html#time.perf_counter |

 

## Licence
voir [LICENSE](LICENSE)

