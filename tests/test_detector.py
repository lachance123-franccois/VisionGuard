"""
tests/test_detector.py — Tests unitaires du détecteur YOLO.
Utilise pytest. Aucune dépendance à un GPU ni à une caméra réelle.
"""

from __future__ import annotations

import types
import numpy as np
import pytest

# Stubs pour éviter d'importer ultralytics dans les tests CI

class _FausseBox:
    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n


class _FauxResultat:
    def __init__(self, n: int) -> None:
        self.boxes = _FausseBox(n)

    def plot(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)


class _FauxModel:
    def __call__(self, frame, **kwargs):
        return [_FauxResultat(3)]


# Patch ultralytics avant l'import du module
import sys
ultralytics_stub = types.ModuleType("ultralytics")
ultralytics_stub.YOLO = lambda *a, **kw: _FauxModel()
sys.modules.setdefault("ultralytics", ultralytics_stub)

# Patch cv2 minimal pour les tests sans affichage
import unittest.mock as mock
sys.modules.setdefault("cv2", mock.MagicMock())

from src.detecteur import FPSCompteur, YOLODetecteur  # noqa: E402

# Tests FPSCounter

class TestFPSCompteur:
    def test_initial_fps_is_zero(self):
        compteur = FPSCompteur()
        assert compteur.tick() == 0.0

    def test_fps_positive_after_ticks(self):
        import time
        compteur = FPSCompteur()
        for _ in range(5):
            compteur.tick()
            time.sleep(0.01)
        fps = compteur.tick()
        assert fps > 0

    def test_window_capping(self):
        compteur= FPSCompteur(window=5)
        for _ in range(20):
            compteur.tick()
        assert len(compteur._times) <= 5


# Tests YOLODetector sansfichiers réels

class TestYOLODetecteur:
    def setup_methode(self):
        self.detecteur = YOLODetecteur(model_path="yolov8n.pt")

    def test_model_load(self):
        assert self.detecteur.model is not None

    def test_predict_r(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        resultats = self.detecteur._predict(frame)
        assert len(resultats) == 1
        assert len(resultats[0].boxes) == 3

    def test_annotatation_r(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        resultats = self.detecteur._predict(frame)
        out = self.detecteur._annotatation(resultats)
        assert isinstance(out, np.ndarray)
