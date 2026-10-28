"""Censura automática de caras, con selección de cuáles censurar."""
from __future__ import annotations

import cv2
import numpy as np

from core.facecore import DETECTOR, FaceState
from core.params import Effect, F, I, C
from core.pipeline import register


def censor_region(img, box, style, intensity, margin, shape):
    ih, iw = img.shape[:2]
    x, y, w, h = [float(v) for v in box]
    cx, cy = x + w / 2, y + h / 2
    w2, h2 = w * (1 + margin), h * (1 + margin)
    x0, y0 = int(max(0, cx - w2 / 2)), int(max(0, cy - h2 / 2))
    x1, y1 = int(min(iw, cx + w2 / 2)), int(min(ih, cy + h2 / 2))
    rw, rh = x1 - x0, y1 - y0
    if rw < 4 or rh < 4:
        return
    roi = img[y0:y1, x0:x1]
    if style == "pixelado":
        n = max(2, int((1.0 - intensity) * 28) + 3)                # menos bloques = más censura
        small = cv2.resize(roi, (n, max(2, int(n * rh / rw))), interpolation=cv2.INTER_AREA)
        cens = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    elif style == "desenfoque":
        k = int(max(rw, rh) * (0.15 + 0.6 * intensity)) | 1
        cens = cv2.GaussianBlur(roi, (k, k), 0)
        cens = cv2.GaussianBlur(cens, (k, k), 0)
    elif style == "blanco":
        cens = np.full_like(roi, 255)
    else:                                                          # negro
        cens = np.zeros_like(roi)
    if shape == "elipse":
        mask = np.zeros((rh, rw), np.uint8)
        cv2.ellipse(mask, (rw // 2, rh // 2), (max(1, rw // 2), max(1, rh // 2)), 0, 0, 360, 255, -1)
        roi[mask > 0] = cens[mask > 0]
    else:
        roi[:] = cens


@register
class FaceCensor(Effect):
    name = "Censura de caras"
    category = "Caras"
    pin_first = True        # se aplica antes que cualquier otro efecto: nada puede "des-censurar"
    NO_MIX = True           # la censura siempre es total: sin mezcla con la imagen original
    PARAMS = [C("modo", "Modo", ["todas", "todas menos las elegidas", "solo las elegidas"]),
              C("estilo", "Estilo", ["pixelado", "desenfoque", "negro", "blanco"]),
              F("intensidad", "Intensidad", 0.7, 0.05, 1.0), F("margen", "Margen alrededor", 0.25, 0.0, 1.0),
              C("forma", "Forma", ["elipse", "rectángulo"]),
              F("umbral", "Umbral de detección (bajo = más caras)", 0.5, 0.3, 0.95),
              I("cada", "Detectar cada N fotogramas", 1, 1, 6),
              I("memoria", "Mantener censura tras perder la cara (fotogramas)", 20, 0, 90)]

    def reset(self):
        self._needs_reset = True

    def process(self, frame, ctx):
        fs = ctx.faces
        if fs is None:
            ctx.faces = fs = FaceState()
        if getattr(self, "_needs_reset", False):
            fs.clear_tracks()
            self._needs_reset = False
        fs.mode = self.value("modo", ctx)
        if ctx.frame_index % self.value("cada", ctx) == 0 or not fs.tracks:
            dets = DETECTOR.detect(frame, self.value("umbral", ctx))
            fs.update(dets, frame, ctx.frame_index, self.value("memoria", ctx))
            if DETECTOR.error:
                self.error = DETECTOR.error
        out = frame.copy()
        style, inten = self.value("estilo", ctx), self.value("intensidad", ctx)
        margin, shape = self.value("margen", ctx), self.value("forma", ctx)
        with fs.lock:
            for t in fs.tracks.values():
                if fs.should_censor(t):
                    censor_region(out, t.box, style, inten, margin, shape)
        return out
