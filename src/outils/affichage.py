
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def dessin_stats_overlay(
    frame: np.ndarray,
    fps: Optional[float],
    n_objects: int,
    latency_ms: float,
) -> None:

    h, w = frame.shape[:2]
    bar_h = 36
    overlay = frame.copy()

    # Fond semi-transparent
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Texte
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.55
    color = (220, 220, 220)
    thick = 1
    y = h - 10

    fps_str = f"FPS: {fps:.1f}" if fps is not None else "FPS: —"
    lat_str = f"Latence: {latency_ms:.1f} ms"
    obj_str = f"Objets: {n_objects}"

    cv2.putText(frame, fps_str, (10, y), font, scale, color, thick, cv2.LINE_AA)
    cv2.putText(frame, lat_str, (160, y), font, scale, color, thick, cv2.LINE_AA)
    cv2.putText(frame, obj_str, (360, y), font, scale, color, thick, cv2.LINE_AA)
