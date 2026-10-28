"""Ayudas para usar Ultralytics YOLO (se importa solo cuando hace falta)."""
from __future__ import annotations

import numpy as np


def to_np(x):
    """Convierte tensores de torch (o listas) en arrays de NumPy."""
    if x is None:
        return None
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x)


class Runner:
    """Carga un modelo YOLO y lo ejecuta con seguimiento (IDs estables) si es posible."""

    def __init__(self, name):
        from ultralytics import YOLO       # lanza ImportError si no está instalado
        from core.utils import MODELS_DIR
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(MODELS_DIR / name))   # descarga los pesos la primera vez
        self.name = name
        self.track_ok = True

    @property
    def names(self):
        return dict(self.model.names)

    def __call__(self, frame, conf=0.35, classes=None, imgsz=640):
        if self.track_ok:
            try:
                return self.model.track(frame, persist=True, conf=conf, classes=classes, imgsz=imgsz,
                                        verbose=False, tracker="bytetrack.yaml")[0]
            except Exception:
                self.track_ok = False       # p. ej. falta la librería del tracker: seguimos sin IDs
        return self.model.predict(frame, conf=conf, classes=classes, imgsz=imgsz, verbose=False)[0]

    def reset_tracking(self):
        try:
            pred = getattr(self.model, "predictor", None)
            if pred is not None and hasattr(pred, "trackers"):
                for t in pred.trackers:
                    t.reset()
        except Exception:
            try:
                self.model.predictor = None
            except Exception:
                pass
