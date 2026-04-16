from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from outils.logpy import connexion
from outils.affichage import dessin_stats_overlay

import tkinter as tk
from tkinter import filedialog, simpledialog


DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_CONF = 0.40
DEFAULT_IOU = 0.45

log = logging.getLogger(__name__)


class YOLODetecteur:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
        classes: Optional[list[int]] = None,
        device: str = "cpu",
    ) -> None:
        log.info("Chargement du modele : %s", model_path)
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.classes = classes
        self.device = device

        log.info("Modele pret (device=%s, conf=%.2f, iou=%.2f)", device, conf, iou)

    def _predict(self, frame: np.ndarray):
        return self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )

    def _annotation(self, resultats) -> np.ndarray:
        return resultats[0].plot()

    def _get_names(self):
        return self.model.names

    def detect_image(self, path: str, save: bool = False) -> np.ndarray:
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"Image introuvable : {path}")

        frame = cv2.imread(str(src))
        if frame is None:
            raise ValueError(f"Impossible de lire l'image : {path}")

        t0 = time.perf_counter()
        resultats = self._predict(frame)

        names = self._get_names()

        if len(resultats[0].boxes) > 0:
            cls = resultats[0].boxes.cls
            
            if hasattr(cls, "cpu"):
                cls_ids = cls.cpu().numpy()
            else:
                cls_ids = np.array(cls)
            detecte = [names[int(i)] for i in cls_ids]
        else:
            detecte = []

        elapse = (time.perf_counter() - t0) * 1000

        annotation = self._annotation(resultats)
        n_object = len(resultats[0].boxes)

        log.info("Detection image : %d objet(s) en %.1f ms", n_object, elapse)
        log.info("Objets détectés : %s", detecte)

        dessin_stats_overlay(annotation, fps=None, n_objects=n_object, latency_ms=elapse)

        if save:
            out_path = src.parent / f"{src.stem}_detected{src.suffix}"
            cv2.imwrite(str(out_path), annotation)
            log.info("Image sauvegardee : %s", out_path)

        cv2.imshow("YOLOv8 — Détection image", annotation)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return annotation

    def detect_flux(
        self,
        source: int | str = 0,
        save: bool = False,
        output_path: Optional[str] = None,
    ) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise IOError(f"Impossible d'ouvrir la source : {source}")

        fps_cap = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if save:
            dest = output_path or "output_detected.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(dest, fourcc, fps_cap, (w, h))
            log.info("Enregistrement active : %s", dest)

        names = self._get_names()

        fps_compteur = FPSCompteur()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                t0 = time.perf_counter()
                resultats = self._predict(frame)

                if len(resultats[0].boxes) > 0:
                    cls = resultats[0].boxes.cls
                    
                    if hasattr(cls, "cpu"):
                        cls_ids = cls.cpu().numpy()
                    else:
                        cls_ids = np.array(cls)
                    detecte = [names[int(i)] for i in cls_ids]
                else:
                    detecte = []

                latency = (time.perf_counter() - t0) * 1000
                fps = fps_compteur.tick()

                annotate = self._annotation(resultats)
                n_objects = len(resultats[0].boxes)

                log.info("Objets détectés : %s", detecte)

                dessin_stats_overlay(
                    annotate,
                    fps=fps,
                    n_objects=n_objects,
                    latency_ms=latency,
                )

                if writer:
                    writer.write(annotate)

                cv2.imshow("YOLOv8", annotate)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()


class FPSCompteur:
    def __init__(self, window: int = 30) -> None:
        self._times: list[float] = []
        self._window = window

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


def ask_parameters():
    root = tk.Tk()
    root.withdraw()

    mode = simpledialog.askstring("Mode YOLO", "image / webcam / video")
    mode = (mode or "webcam").lower()

    source = None

    if mode in ["image", "video"]:
        source = filedialog.askopenfilename()
    elif mode == "webcam":
        source = 0

    conf = simpledialog.askfloat("Confiance", "0.4", initialvalue=0.4)
    iou = simpledialog.askfloat("IoU", "0.45", initialvalue=0.45)
    
    save = simpledialog.askstring("Sauvegarder ?", "oui / non", initialvalue="non")
    save = save.strip().lower() == "oui"


    return mode, source, conf, iou,save


def main() -> None:
    mode, source, conf, iou,save = ask_parameters()

    connexion(verbose=False)

    detecteur = YOLODetecteur(
        model_path=DEFAULT_MODEL,
        conf=conf,
        iou=iou,
        classes=None,
        device="cpu",
    )

    if mode == "image":
        detecteur.detect_image(path=source, save=save)

    elif mode == "webcam":
        detecteur.detect_flux(source=0, save=save)

    elif mode == "video":
        detecteur.detect_flux(source=source, save=save)


if __name__ == "__main__":
    main()