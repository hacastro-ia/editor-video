"""Ruido fractal suave que evoluciona en el tiempo (para deformaciones fluidas tipo «domain warping»)."""
from __future__ import annotations

import cv2
import numpy as np


class FlowNoise:
    def __init__(self, seed=0):
        self.seed = int(seed)
        self.cache = {}

    def _grid(self, o, k, gw, gh, ch):
        key = (o, k, gw, gh, ch)
        g = self.cache.get(key)
        if g is None:
            if len(self.cache) > 300:
                self.cache.clear()
            rng = np.random.default_rng((self.seed * 1000003 + o * 7919 + k * 104729 + ch * 31) & 0xFFFFFFFF)
            g = rng.standard_normal((gh, gw)).astype(np.float32)
            self.cache[key] = g
        return g

    def field(self, h, w, t, scale=3.0, octaves=3, speed=0.35, ch=0, lowres=2):
        """Campo suave (-1..1 aprox.) de tamaño h×w. scale = celdas de ruido a lo ancho."""
        hh, ww = max(8, h // lowres), max(8, w // lowres)
        out = np.zeros((hh, ww), np.float32)
        amp, tot = 1.0, 0.0
        for o in range(int(octaves)):
            gs = scale * (2 ** o)
            gw, gh = int(gs) + 4, max(4, int(gs * h / w) + 4)
            tt = t * speed * (1.0 + 0.45 * o)
            k = int(np.floor(tt))
            f = tt - k
            f = f * f * (3 - 2 * f)
            g = self._grid(o, k, gw, gh, ch) * (1 - f) + self._grid(o, k + 1, gw, gh, ch) * f
            out += cv2.resize(g, (ww, hh), interpolation=cv2.INTER_CUBIC) * amp
            tot += amp
            amp *= 0.5
        out /= tot * 1.1                                           # aprox. rango -1..1
        if lowres > 1:
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        return out
