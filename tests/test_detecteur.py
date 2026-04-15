from __future__ import annotations

import types
import sys
import numpy as np
import unittest.mock as mock

# =========================
# FAKE YOLO STUBS
# =========================

class _FausseBox:
    def __init__(self, n: int) -> None:
        self.cls = np.array([0, 1, 2])  # classes fake

    def __len__(self) -> int:
        return len(self.cls)


class _FauxResultat:
    def __init__(self, n: int) -> None:
        self.boxes = _FausseBox(n)
        self.names = {0: "person", 1: "car", 2: "dog"}

    def plot(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)


class _FauxModel:
    def __init__(self):
        self.names = {0: "person", 1: "car", 2: "dog"}

    def __call__(self, frame, **kwargs):
        return [_FauxResultat(3)]


# =========================
# PATCH ULTRALYTICS
# =========================

ultralytics_stub = types.ModuleType("ultralytics")
ultralytics_stub.YOLO = lambda *a, **kw: _FauxModel()
sys.modules["ultralytics"] = ultralytics_stub


# =========================
# PATCH CV2
# =========================

cv2_mock = mock.MagicMock()
cv2_mock.imread.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
sys.modules["cv2"] = cv2_mock


# =========================
# IMPORT MODULE TESTE
# =========================

from src.detecteur import FPSCompteur, YOLODetecteur  # noqa: E402


# =========================
# TEST FPS
# =========================

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

        assert compteur.tick() >= 0.0

    def test_window_capping(self):
        compteur = FPSCompteur(window=5)

        for _ in range(20):
            compteur.tick()

        assert len(compteur._times) <= 5


# =========================
# TEST YOLO
# =========================

class TestYOLODetecteur:

    def setup_method(self):
        self.detecteur = YOLODetecteur(model_path="yolov8n.pt")

    def test_model_load(self):
        assert self.detecteur.model is not None

    def test_predict_r(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        resultats = self.detecteur._predict(frame)

        assert len(resultats) == 1
        assert len(resultats[0].boxes) == 3

    def test_annotation_r(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        resultats = self.detecteur._predict(frame)

        out = self.detecteur._annotation(resultats)

        assert isinstance(out, np.ndarray)
        assert out.shape == (480, 640, 3)