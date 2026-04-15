import numpy as np
from ultralytics import YOLO

class YOLOService:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, image: np.ndarray):
        results = self.model(image)[0]
        names = self.model.names

        detections = []

        if len(results.boxes) > 0:
            cls = results.boxes.cls
            conf = results.boxes.conf

            cls = cls.cpu().numpy() if hasattr(cls, "cpu") else np.array(cls)
            conf = conf.cpu().numpy() if hasattr(conf, "cpu") else np.array(conf)

            for c, p in zip(cls, conf):
                detections.append({
                    "class": names[int(c)],
                    "confidence": float(p)
                })

        annotated = results.plot()

        return detections, annotated