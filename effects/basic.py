"""Efectos básicos: color, nitidez, degradados y estilos gráficos."""
from __future__ import annotations

import cv2
import numpy as np

from core import palettes
from core.params import Effect, F, I, B, C
from core.pipeline import register
from core.utils import grid, scale_saturation, shift_hue


def _mix(orig, new, amount):
    if amount >= 0.999:
        return new
    if amount <= 0.001:
        return orig
    return cv2.addWeighted(orig, 1.0 - amount, new, amount, 0)


_vig_cache: dict = {}


def vignette_map(h, w, power=1.5):
    key = (h, w, round(power, 2))
    m = _vig_cache.get(key)
    if m is None:
        if len(_vig_cache) > 6:
            _vig_cache.clear()
        X, Y = grid(w, h)
        r = np.sqrt(((X - w / 2) / (w / 2)) ** 2 + ((Y - h / 2) / (h / 2)) ** 2) / 1.4142
        m = np.clip(r, 0, 1) ** power
        _vig_cache[key] = m.astype(np.float32)
    return m


@register
class ColorAdjust(Effect):
    name = "Ajuste de color"
    category = "Básicos"
    PARAMS = [F("brillo", "Brillo", 0, -100, 100), F("contraste", "Contraste", 1.0, 0.2, 3.0),
              F("gamma", "Gamma", 1.0, 0.3, 3.0), F("saturacion", "Saturación", 1.0, 0.0, 3.0),
              F("tono", "Tono (°)", 0, -180, 180), F("temperatura", "Temperatura (frío ↔ cálido)", 0.0, -1.0, 1.0),
              F("viñeta", "Viñeta", 0.0, 0.0, 1.0), F("nitidez", "Nitidez", 0.0, 0.0, 3.0),
              F("desenfoque", "Desenfoque", 0, 0, 30)]

    def process(self, frame, ctx):
        out = frame
        g = self.value("gamma", ctx)
        if abs(g - 1.0) > 0.01:
            lut = (np.power(np.arange(256) / 255.0, 1.0 / g) * 255).astype(np.uint8)
            out = cv2.LUT(out, lut)
        out = cv2.convertScaleAbs(out, alpha=self.value("contraste", ctx), beta=self.value("brillo", ctx))
        out = shift_hue(out, self.value("tono", ctx))
        out = scale_saturation(out, self.value("saturacion", ctx))
        t = self.value("temperatura", ctx)
        if abs(t) > 0.01:
            gains = np.array([1.0 - 0.28 * t, 1.0, 1.0 + 0.28 * t], np.float32)       # B, G, R
            out = np.clip(out.astype(np.float32) * gains, 0, 255).astype(np.uint8)
        k = int(self.value("desenfoque", ctx))
        if k > 0:
            out = cv2.GaussianBlur(out, (2 * k + 1, 2 * k + 1), 0)
        n = self.value("nitidez", ctx)
        if n > 0.02:
            out = cv2.addWeighted(out, 1.0 + n, cv2.GaussianBlur(out, (0, 0), 1.6), -n, 0)
        v = self.value("viñeta", ctx)
        if v > 0.01:
            m = 1.0 - v * vignette_map(*out.shape[:2])[..., None]
            out = (out.astype(np.float32) * m).astype(np.uint8)
        return out


@register
class Grayscale(Effect):
    name = "Blanco y negro"
    category = "Básicos"
    PARAMS = [F("mezcla", "Intensidad", 1.0, 0, 1), F("contraste", "Contraste", 1.0, 0.3, 3.0),
              F("brillo", "Brillo", 0, -100, 100),
              C("tinte", "Tinte", ["neutro", "sepia", "azulado", "verde fósforo", "ámbar (monitor)"])]

    def process(self, frame, ctx):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.convertScaleAbs(g, alpha=self.value("contraste", ctx), beta=self.value("brillo", ctx))
        tint = self.value("tinte", ctx)
        if tint == "neutro":
            out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        elif tint == "azulado":
            out = palettes.apply(g, "hielo")
        else:
            out = palettes.apply(g, tint)
        return _mix(frame, out, self.value("mezcla", ctx))


@register
class Sepia(Effect):
    name = "Sepia"
    category = "Básicos"
    PARAMS = [F("mezcla", "Intensidad", 1.0, 0, 1)]
    _M = np.array([[0.131, 0.534, 0.272], [0.168, 0.686, 0.349], [0.189, 0.769, 0.393]], np.float32)

    def process(self, frame, ctx):
        return _mix(frame, cv2.transform(frame, self._M), self.value("mezcla", ctx))


@register
class Negative(Effect):
    name = "Negativo"
    category = "Básicos"
    PARAMS = [F("mezcla", "Intensidad", 1.0, 0, 1), B("solo_luz", "Invertir solo la luminosidad", False)]

    def process(self, frame, ctx):
        if self.value("solo_luz", ctx):
            ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            ycc[..., 0] = 255 - ycc[..., 0]
            neg = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
        else:
            neg = 255 - frame
        return _mix(frame, neg, self.value("mezcla", ctx))


@register
class Threshold(Effect):
    name = "Umbral / Duotono"
    category = "Básicos"
    PARAMS = [F("umbral", "Umbral", 0.5, 0.0, 1.0), F("suavidad", "Suavidad del corte", 0.04, 0.0, 0.5),
              C("paleta", "Paleta", palettes.NAMES, "neón rosa y cian"),
              F("desplazamiento", "Desplazar la paleta", 0.0, 0.0, 1.0), B("ciclar", "Ciclar la paleta con el tiempo", False),
              F("ruido", "Ruido (tramado)", 0.0, 0.0, 0.5)]

    def process(self, frame, ctx):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        n = self.value("ruido", ctx)
        if n > 0.01:
            g = g + (np.random.default_rng(ctx.frame_index).random(g.shape, dtype=np.float32) - 0.5) * n
        s = max(self.value("suavidad", ctx), 1e-3)
        x = np.clip((g - self.value("umbral", ctx)) / s * 0.5 + 0.5, 0, 1)
        shift = self.value("desplazamiento", ctx) + (ctx.time * 0.1 if self.value("ciclar", ctx) else 0)
        return palettes.apply((x * 255).astype(np.uint8), self.value("paleta", ctx), shift, cyclic=self.value("ciclar", ctx))


@register
class Cartoon(Effect):
    name = "Caricatura"
    category = "Básicos"
    PARAMS = [I("niveles", "Niveles de color", 6, 2, 12), F("lineas", "Umbral de líneas", 7, 2, 15),
              I("grosor", "Grosor de línea", 1, 1, 4), F("suavizado", "Suavizado del color", 1.0, 0.0, 3.0),
              F("saturacion", "Saturación", 1.2, 0.0, 2.5), F("opacidad_linea", "Opacidad de las líneas", 1.0, 0.0, 1.0)]

    def process(self, frame, ctx):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(2, w // 2), max(2, h // 2)))
        for _ in range(int(round(self.value("suavizado", ctx)))):
            small = cv2.bilateralFilter(small, 7, 50, 50)
        smooth = scale_saturation(cv2.resize(small, (w, h)), self.value("saturacion", ctx))
        step = 256 // self.value("niveles", ctx)
        poster = (smooth // step * step + step // 2).astype(np.uint8)
        gray = cv2.medianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9,
                                      self.value("lineas", ctx))
        k = self.value("grosor", ctx)
        if k > 1:
            edges = cv2.erode(edges, np.ones((k, k), np.uint8))
        a = self.value("opacidad_linea", ctx)
        dark = (edges == 0)[..., None]
        return np.where(dark, (poster * (1.0 - a)).astype(np.uint8), poster)


@register
class Sketch(Effect):
    name = "Boceto a lápiz"
    category = "Básicos"
    PARAMS = [F("suavidad", "Grosor del trazo", 10, 2, 40), B("color", "Conservar color", False),
              F("contraste", "Contraste del trazo", 1.0, 0.5, 3.0), F("papel", "Textura de papel", 0.0, 0.0, 1.0)]

    def process(self, frame, ctx):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(255 - gray, (0, 0), self.value("suavidad", ctx))
        sk = cv2.divide(gray, 255 - blur, scale=256)
        c = self.value("contraste", ctx)
        if abs(c - 1) > 0.02:
            sk = np.clip((sk.astype(np.float32) - 128) * c + 128, 0, 255).astype(np.uint8)
        p = self.value("papel", ctx)
        if p > 0.01:
            h, w = sk.shape
            tex = cv2.GaussianBlur(np.random.default_rng(3).random((h, w), dtype=np.float32), (0, 0), 1.2)
            sk = np.clip(sk.astype(np.float32) * (1 - p * 0.25 + p * 0.25 * (tex * 2 - 0.5)), 0, 255).astype(np.uint8)
        sk = cv2.cvtColor(sk, cv2.COLOR_GRAY2BGR)
        if self.value("color", ctx):
            sk = cv2.multiply(frame, sk, scale=1 / 255.0)
        return sk


@register
class Pixelate(Effect):
    name = "Pixelado"
    category = "Básicos"
    PARAMS = [I("bloque", "Tamaño del bloque", 12, 2, 80), C("forma", "Forma", ["cuadrado", "círculo", "rombo", "rejilla"]),
              F("espacio", "Separación entre bloques", 0.15, 0.0, 0.8), F("brillo_bloque", "Brillo de cada bloque", 1.0, 0.3, 2.0)]

    def process(self, frame, ctx):
        h, w = frame.shape[:2]
        b = self.value("bloque", ctx)
        cw, ch = max(1, w // b), max(1, h // b)
        small = cv2.resize(frame, (cw, ch), interpolation=cv2.INTER_AREA)
        forma = self.value("forma", ctx)
        if forma == "cuadrado":
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        small = cv2.convertScaleAbs(small, alpha=self.value("brillo_bloque", ctx))
        big = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = np.zeros((h, w), np.uint8)
        bw, bh = w / cw, h / ch
        gap = self.value("espacio", ctx)
        r = max(1, int(min(bw, bh) * 0.5 * (1.0 - gap)))
        for j in range(ch):
            cy = int((j + 0.5) * bh)
            for i in range(cw):
                cx = int((i + 0.5) * bw)
                if forma == "círculo":
                    cv2.circle(mask, (cx, cy), r, 255, -1)
                elif forma == "rombo":
                    cv2.fillPoly(mask, [np.array([[cx, cy - r], [cx + r, cy], [cx, cy + r], [cx - r, cy]], np.int32)], 255)
                else:
                    cv2.rectangle(mask, (cx - r, cy - r), (cx + r, cy + r), 255, -1)
        return cv2.bitwise_and(big, big, mask=mask)
