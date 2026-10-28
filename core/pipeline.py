"""Cadena de efectos y registro de efectos disponibles."""
from __future__ import annotations

import traceback

import cv2
import numpy as np

REGISTRY: dict = {}
CATEGORY_ORDER = ["Básicos", "Estilo TouchDesigner", "Cinta analógica", "Audio", "Datamosh", "Nube de puntos",
                  "Psicodélico", "Detección", "Personas", "Biblioteca", "Caras"]


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


def by_category():
    cats = {c: [] for c in CATEGORY_ORDER}
    for cls in REGISTRY.values():
        cats.setdefault(cls.category, []).append(cls)
    return {c: v for c, v in cats.items() if v}


def blend(base, top, mode, mix):
    """Mezcla `top` sobre `base` con un modo tipo Photoshop / TouchDesigner (Composite) y una opacidad."""
    b = base.astype(np.float32) * (1.0 / 255.0)
    t = top.astype(np.float32) * (1.0 / 255.0)
    if mode == "sumar":
        r = np.minimum(b + t, 1.0)
    elif mode == "pantalla":
        r = 1.0 - (1.0 - b) * (1.0 - t)
    elif mode == "multiplicar":
        r = b * t
    elif mode == "superponer":
        r = np.where(b < 0.5, 2.0 * b * t, 1.0 - 2.0 * (1.0 - b) * (1.0 - t))
    elif mode == "diferencia":
        r = np.abs(b - t)
    elif mode == "aclarar":
        r = np.maximum(b, t)
    elif mode == "oscurecer":
        r = np.minimum(b, t)
    else:
        r = t
    r = b + (r - b) * float(mix)
    return (np.clip(r, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


class Chain:
    def __init__(self):
        self.effects = []

    def add(self, effect):
        if effect.pin_first:
            idx = sum(1 for e in self.effects if e.pin_first)
            self.effects.insert(idx, effect)
        else:
            self.effects.append(effect)
        return self.effects.index(effect)

    def remove(self, index):
        if 0 <= index < len(self.effects):
            self.effects.pop(index)

    def move(self, index, delta):
        j = index + delta
        if 0 <= index < len(self.effects) and 0 <= j < len(self.effects):
            a, b = self.effects[index], self.effects[j]
            if a.pin_first != b.pin_first:      # las pinned se quedan primero
                return index
            self.effects[index], self.effects[j] = b, a
            return j
        return index

    def clear(self):
        self.effects = []

    def reset(self):
        for e in self.effects:
            try:
                e.reset()
            except Exception:
                traceback.print_exc()

    def process(self, frame, ctx):
        for e in self.effects:
            if not e.enabled:
                continue
            try:
                out = e.process(frame, ctx)
                if out is not None:
                    if out.shape != frame.shape:
                        out = cv2.resize(out, (frame.shape[1], frame.shape[0]))
                    if not e.NO_MIX:
                        mix, mode = e.value("_mix", ctx), e.params["_blend"].value
                        if mix < 0.999 or mode != "normal":
                            out = blend(frame, out, mode, mix)
                    frame = out
            except Exception as ex:  # un efecto roto no debe tumbar el programa
                e.error = f"{type(ex).__name__}: {ex}"
                traceback.print_exc()
        return frame
