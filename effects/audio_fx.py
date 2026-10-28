"""Efectos que reaccionan al audio (solo alteran la imagen, nunca el sonido)."""
from __future__ import annotations

import cv2
import numpy as np

from core.params import Effect, F, I, C
from core.pipeline import register
from core.utils import grid, hue_color, scale_saturation, shift_hue
from effects.touch import GlitchRGB


@register
class AudioDistort(Effect):
    name = "Audio: Distorsión ondulada"
    category = "Audio"
    PARAMS = [F("amplitud", "Amplitud de onda", 0.004, 0.0, 0.08, mod=("medios", 0.6)),
              F("frecuencia", "Frecuencia", 6, 1, 30), F("velocidad", "Velocidad", 2, 0, 10),
              F("abultamiento", "Abultamiento (lente)", 0.0, 0, 1, mod=("graves", 0.7))]

    def process(self, frame, ctx):
        h, w = frame.shape[:2]
        X, Y = grid(w, h)
        amp = self.value("amplitud", ctx) * w
        f = self.value("frecuencia", ctx)
        ph = ctx.time * self.value("velocidad", ctx) * 2 * np.pi
        mx = X + amp * np.sin(2 * np.pi * f * Y / h + ph)
        my = Y + amp * 0.6 * np.cos(2 * np.pi * f * X / w + ph * 0.8)
        b = self.value("abultamiento", ctx)
        if b > 0.01:
            cx, cy = w / 2, h / 2
            dx, dy = mx - cx, my - cy
            r = np.sqrt(dx * dx + dy * dy) / (0.5 * np.hypot(w, h))
            k = 1.0 - b * 0.45 * np.clip(1.0 - r, 0, 1)
            mx, my = cx + dx * k, cy + dy * k
        return cv2.remap(frame, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)


@register
class AudioGlitch(GlitchRGB):
    name = "Audio: Glitch (beat)"
    category = "Audio"
    OVERRIDES = {"intensidad": 0.0}
    DEFAULT_MODS = {"intensidad": ("beat", 0.9), "desplazamiento": ("graves", 0.4)}


@register
class AudioZoom(Effect):
    name = "Audio: Pulso de zoom"
    category = "Audio"
    PARAMS = [F("zoom", "Zoom", 1.0, 1.0, 1.8, mod=("graves", 0.35)),
              F("giro", "Giro (°)", 0.0, -30, 30, mod=("beat", 0.15))]

    def process(self, frame, ctx):
        h, w = frame.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), self.value("giro", ctx), self.value("zoom", ctx))
        return cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_REFLECT)


@register
class AudioHue(Effect):
    name = "Audio: Matiz y saturación"
    category = "Audio"
    PARAMS = [F("matiz", "Cambio de matiz (°)", 0, 0, 180, mod=("agudos", 0.8)),
              F("saturacion", "Saturación", 1.0, 0, 3, mod=("volumen", 0.5))]

    def process(self, frame, ctx):
        return scale_saturation(shift_hue(frame, self.value("matiz", ctx)), self.value("saturacion", ctx))


@register
class AudioSpectrum(Effect):
    name = "Audio: Espectro"
    category = "Audio"
    PARAMS = [C("estilo", "Estilo", ["barras", "radial"]), F("altura", "Tamaño", 0.25, 0.05, 0.6),
              F("opacidad", "Opacidad", 0.85, 0.1, 1.0), F("velocidad_color", "Velocidad del color (°/s)", 40, 0, 360)]

    def process(self, frame, ctx):
        spec = ctx.audio.get("spectrum")
        if spec is None:
            return frame
        h, w = frame.shape[:2]
        n = len(spec)
        over = np.zeros_like(frame)
        size = self.value("altura", ctx)
        hue0 = ctx.time * self.value("velocidad_color", ctx)
        if self.value("estilo", ctx) == "barras":
            bw = w / n
            for i, v in enumerate(spec):
                x0, x1 = int(i * bw), int((i + 1) * bw) - 2
                y0 = h - int(float(v) * h * size * 2.2)
                cv2.rectangle(over, (x0, y0), (max(x0 + 1, x1), h - 1), hue_color(hue0 + i * 360 / n), -1)
        else:
            cx, cy = w // 2, h // 2
            r0 = int(min(w, h) * 0.18)
            for i, v in enumerate(spec):
                a = 2 * np.pi * i / n + ctx.time * 0.3
                r1 = r0 + int(float(v) * min(w, h) * size * 1.8)
                p0 = (int(cx + r0 * np.cos(a)), int(cy + r0 * np.sin(a)))
                p1 = (int(cx + r1 * np.cos(a)), int(cy + r1 * np.sin(a)))
                cv2.line(over, p0, p1, hue_color(hue0 + i * 360 / n), max(2, w // 160))
        mask = over.max(axis=2) > 0
        out = frame.copy()
        a = self.value("opacidad", ctx)
        out[mask] = cv2.addWeighted(frame, 1 - a, over, a, 0)[mask]
        return out
