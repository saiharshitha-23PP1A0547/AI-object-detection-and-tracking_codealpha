import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

class YOLODetector:
    """YOLO Object Detector wrapper for CodeAlpha AI Internship."""
    def __init__(self, model_name: str = "yolo11n.pt"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[YOLODetector] Initializing model '{model_name}' on device: {self.device.upper()}")
        print("[YOLODetector] Pretrained weights download automatically on first run.")
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", model_name) if not os.path.exists(model_name) else model_name
        try:
            self.model = YOLO(model_path if os.path.exists(model_path) else model_name)
        except Exception:
            self.model = YOLO(model_name)
        self.classes = self.model.names if hasattr(self.model, "names") else {}

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.35, iou_threshold: float = 0.45, allowed_classes: list = None):
        if frame is None or frame.size == 0:
            return []
        results = self.model.predict(source=frame, conf=conf_threshold, iou=iou_threshold, device=self.device, verbose=False)
        detections = []
        if not results or len(results) == 0 or results[0].boxes is None:
            return detections
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        for i in range(len(xyxy)):
            box = xyxy[i].tolist()
            score = float(confs[i])
            class_id = int(cls_ids[i])
            class_name = self.classes.get(class_id, f"class_{class_id}")
            if allowed_classes is not None and class_name not in allowed_classes:
                continue
            detections.append({
                "box": [round(coord, 1) for coord in box],
                "score": round(score, 3),
                "class_id": class_id,
                "class_name": class_name
            })
        return detections
