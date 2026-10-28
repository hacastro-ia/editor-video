"""Prueba rapida del entorno de compilacion: nms de torchvision y YOLO.

Se ejecuta en CI antes de empaquetar. Si falla aqui, el exe tambien fallaria
con "operator torchvision::nms does not exist".
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import nms

cajas = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]])
puntuaciones = torch.tensor([0.9, 0.8])
print("nms:", nms(cajas, puntuaciones, 0.5).tolist())

from ultralytics import YOLO  # noqa: E402

print("ultralytics importado")

# Igual que la app: ruta absoluta escribible, no el directorio actual.
pesos = Path(tempfile.mkdtemp()) / "yolov8n.pt"
modelo = YOLO(str(pesos))
marco = np.zeros((64, 64, 3), dtype=np.uint8)
modelo(marco, verbose=False)
print("inferencia OK")

print("SMOKE OK")
sys.exit(0)
