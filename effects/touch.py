"""Efectos al estilo TouchDesigner: bordes, glitch, espejo, feedback y mapas de color."""
from __future__ import annotations

import cv2
import numpy as np

from core import palettes
from core.params import Effect, F, I, B, C
from core.pipeline import register
from core.utils import fast_blur, grid, shift_hue, slice_glitch
from effects.basic import _mix


@register
class NeonEdges(Effect):
    name = "Bordes de neón"
    category = "Estilo TouchDesigner"
    PARAMS = [F("umbral", "Umbral de bordes", 80, 20, 200), F("suavizado", "Suavizado antes de detectar", 1.0, 0.0, 4.0),
              I("grosor", "Grosor", 1, 1, 6), F("resplandor", "Resplandor", 8, 0, 30), F("halo", "Halo amplio", 0.0, 0.0, 1.0),
              F("intensidad", "Intensidad del neón", 1.0, 0.2, 2.5), F("fondo", "Imagen de fondo", 0.0, 0, 1),
              C("color", "Color", ["arcoíris", "degradado (paleta)", "blanco"]),
              C("paleta", "Paleta (degradado)", palettes.NAMES, "neón rosa y cian"),
              F("velocidad_color", "Velocidad del color (°/s)", 40, 0, 360)]

    def process(self, frame, ctx):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        s = self.value("suavizado", ctx)
        if s > 0.1:
            gray = cv2.GaussianBlur(gray, (0, 0), s)
        t = self.value("umbral", ctx)
        edges = cv2.Canny(gray, t, t * 2.2)
        k = self.value("grosor", ctx)
        if k > 1:
            edges = cv2.dilate(edges, np.ones((k, k), np.uint8))
        gx, _ = grid(w, h)
        phase = ctx.time * self.value("velocidad_color", ctx)
        mode = self.value("color", ctx)
        if mode == "arcoíris":
            hue = np.mod(gx / w * 90 + phase / 2, 180).astype(np.uint8)
            neon = cv2.cvtColor(np.dstack([hue, np.full_like(hue, 255), edges]), cv2.COLOR_HSV2BGR)
        else:
            if mode == "blanco":
                col = np.full((h, w, 3), 255, np.uint8)
            else:
                idx = (np.mod(gx / w * 0.6 + phase / 360.0, 1.0) * 255).astype(np.uint8)
                col = palettes.lut(self.value("paleta", ctx), 256, 0, cyclic=True)[idx]
            neon = (col.astype(np.float32) * (edges.astype(np.float32) / 255.0)[..., None]).astype(np.uint8)
        gain = self.value("intensidad", ctx)
        out = cv2.addWeighted(frame, self.value("fondo", ctx), neon, min(gain, 1.0), 0)
        glow_s, halo = self.value("resplandor", ctx), self.value("halo", ctx)
        glow = None
        if glow_s > 0.5:
            glow = cv2.convertScaleAbs(fast_blur(neon, glow_s), alpha=1.6 * gain)
        if halo > 0.02:
            hl = cv2.convertScaleAbs(fast_blur(neon, 22 * w / 640), alpha=2.4 * halo * gain)
            glow = hl if glow is None else cv2.add(glow, hl)
        return cv2.add(out, glow) if glow is not None else out


@register
class GlitchRGB(Effect):
    name = "Glitch RGB"
    category = "Estilo TouchDesigner"
    PARAMS = [F("intensidad", "Intensidad", 0.4, 0, 1), I("desplazamiento", "Desplazamiento RGB (px)", 8, 0, 40),
              I("franjas", "Franjas rotas", 8, 0, 30), C("modo", "Tipo de glitch", ["franjas", "bloques", "ondas"]),
              I("bloque", "Tamaño de bloque (px)", 32, 8, 128), F("vertical", "Saltos verticales", 0.0, 0.0, 1.0),
              F("ruido", "Ruido digital", 0.0, 0.0, 1.0), F("lineas", "Líneas de barrido", 0.0, 0.0, 1.0)]

    def reset(self):
        self.rng = np.random.default_rng(1234)

    def _blocks(self, frame, amt):
        h, w = frame.shape[:2]
        b = self.value("bloque", self._ctx)
        out = frame.copy()
        for _ in range(int(amt * 36)):
            bw, bh = b * int(self.rng.integers(1, 4)), b
            x, y = int(self.rng.integers(0, max(1, w - bw))), int(self.rng.integers(0, max(1, h - bh)))
            dx = int(self.rng.integers(-3, 4)) * b
            dy = int(self.rng.integers(-1, 2)) * b
            sx, sy = int(np.clip(x + dx, 0, w - bw)), int(np.clip(y + dy, 0, h - bh))
            out[y:y + bh, x:x + bw] = frame[sy:sy + bh, sx:sx + bw]
        return out

    def _waves(self, frame, amt, t):
        h, w = frame.shape[:2]
        y = np.arange(h, dtype=np.float32)
        dx = amt * w * 0.05 * np.sin(y / h * 2 * np.pi * 6 + t * 7) * (0.4 + self.rng.random(h)).astype(np.float32)
        X, Y = grid(w, h)
        return cv2.remap(frame, X + dx[:, None], Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    def process(self, frame, ctx):
        self._ctx = ctx
        amt = float(np.clip(self.value("intensidad", ctx), 0, 1))
        if amt <= 0:
            return frame
        h, w = frame.shape[:2]
        modo = self.value("modo", ctx)
        if modo == "bloques":
            out = self._blocks(frame, amt)
        elif modo == "ondas":
            out = self._waves(frame, amt, ctx.time)
        else:
            out = slice_glitch(frame, amt, 0, self.value("franjas", ctx), self.rng)
        s = int(round(self.value("desplazamiento", ctx) * amt))
        if s > 0:
            out = out.copy()
            out[..., 0] = np.roll(out[..., 0], s, axis=1)
            out[..., 2] = np.roll(out[..., 2], -s, axis=1)
        v = self.value("vertical", ctx)
        if v > 0.01 and self.rng.random() < v * 0.6:
            out = np.roll(out, int(self.rng.integers(-h // 6, h // 6) * v), axis=0)
        n = self.value("ruido", ctx)
        if n > 0.01:
            g = (self.rng.standard_normal((h, w)).astype(np.float32) * 40 * n * amt).astype(np.int16)
            out = np.clip(out.astype(np.int16) + g[..., None], 0, 255).astype(np.uint8)
        ln = self.value("lineas", ctx)
        if ln > 0.01:
            m = np.ones((h, 1, 1), np.float32)
            m[::2] = 1.0 - 0.55 * ln
            out = (out.astype(np.float32) * m).astype(np.uint8)
        return out


@register
class Mirror(Effect):
    name = "Espejo"
    category = "Estilo TouchDesigner"
    PARAMS = [C("modo", "Modo", ["un eje", "dos ejes"]), F("angulo", "Ángulo del eje (°)", 90, -90, 90),
              F("posicion", "Posición del eje", 0.0, -0.5, 0.5), F("giro", "Giro automático (°/s)", 0.0, -90, 90),
              B("invertir", "Invertir lado", False)]

    def process(self, frame, ctx):
        h, w = frame.shape[:2]
        X, Y = grid(w, h)
        ang = np.deg2rad(self.value("angulo", ctx) + self.value("giro", ctx) * ctx.time)
        pos = self.value("posicion", ctx)
        sign = -1.0 if self.value("invertir", ctx) else 1.0
        mx, my = X, Y
        axes = 2 if self.value("modo", ctx) == "dos ejes" else 1
        for k in range(axes):
            a = ang + k * np.pi / 2
            nx, ny = np.cos(a), np.sin(a)
            cx = w / 2 + pos * w * nx * (1 if k == 0 else 0)
            cy = h / 2 + pos * h * ny * (1 if k == 0 else 0)
            d = ((mx - cx) * nx + (my - cy) * ny) * sign
            flip = d < 0
            mx = np.where(flip, mx - 2 * d * nx * sign, mx)
            my = np.where(flip, my - 2 * d * ny * sign, my)
        return cv2.remap(frame, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)


_BORDERS = {"reflejar": cv2.BORDER_REFLECT, "repetir": cv2.BORDER_REPLICATE, "negro": cv2.BORDER_CONSTANT}


@register
class Feedback(Effect):
    name = "Estela (feedback)"
    category = "Estilo TouchDesigner"
    PARAMS = [F("mezcla", "Persistencia", 0.85, 0.0, 0.99), F("zoom", "Zoom por fotograma", 1.0, 0.8, 1.2),
              F("rotacion", "Rotación (°/fotograma)", 0.0, -15, 15), F("matiz", "Cambio de matiz (°)", 0, 0, 20),
              F("desplaza_x", "Deriva horizontal", 0.0, -0.03, 0.03), F("desplaza_y", "Deriva vertical", 0.0, -0.03, 0.03),
              F("desenfoque", "Desenfoque de la estela", 0.0, 0.0, 8.0), F("brillo", "Brillo de la estela", 1.0, 0.9, 1.08),
              C("modo", "Modo", ["mezclar", "aclarar", "sumar", "diferencia"]),
              C("bordes", "Bordes", ["reflejar", "repetir", "negro"])]

    def reset(self):
        self.prev = None

    def process(self, frame, ctx):
        h, w = frame.shape[:2]
        if self.prev is None or self.prev.shape != frame.shape:
            self.prev = frame.copy()
            return frame
        m = cv2.getRotationMatrix2D((w / 2, h / 2), self.value("rotacion", ctx), self.value("zoom", ctx))
        m[0, 2] += self.value("desplaza_x", ctx) * w
        m[1, 2] += self.value("desplaza_y", ctx) * h
        warped = cv2.warpAffine(self.prev, m, (w, h), borderMode=_BORDERS[self.value("bordes", ctx)])
        warped = shift_hue(warped, self.value("matiz", ctx))
        bl = self.value("desenfoque", ctx)
        if bl > 0.2:
            warped = cv2.GaussianBlur(warped, (0, 0), bl)
        g = self.value("brillo", ctx)
        if abs(g - 1) > 0.002:
            warped = cv2.convertScaleAbs(warped, alpha=g)
        p = self.value("mezcla", ctx)
        mode = self.value("modo", ctx)
        if mode == "aclarar":
            out = np.maximum(frame, cv2.convertScaleAbs(warped, alpha=p))
        elif mode == "sumar":
            out = cv2.add(frame, cv2.convertScaleAbs(warped, alpha=p))
        elif mode == "diferencia":
            out = cv2.addWeighted(cv2.absdiff(frame, cv2.convertScaleAbs(warped, alpha=p)), 0.7, frame, 0.3, 0)
        else:
            out = cv2.addWeighted(frame, 1.0 - p, warped, p, 0)
        self.prev = out.copy()
        return out


@register
class FractalDroste(Feedback):
    name = "Fractal (Droste)"
    category = "Psicodélico"
    OVERRIDES = {"mezcla": 0.82, "zoom": 0.93, "rotacion": 4.0, "matiz": 3}


_CV_MAPS = {"inferno (OpenCV)": cv2.COLORMAP_INFERNO, "jet (OpenCV)": cv2.COLORMAP_JET,
            "turbo (OpenCV)": cv2.COLORMAP_TURBO, "rainbow (OpenCV)": cv2.COLORMAP_RAINBOW}


@register
class HeatMap(Effect):
    name = "Mapa de calor"
    category = "Estilo TouchDesigner"
    PARAMS = [C("paleta", "Paleta", palettes.NAMES + list(_CV_MAPS.keys()), "magma"),
              F("contraste", "Contraste", 1.0, 0.3, 3.0), F("gamma", "Gamma", 1.0, 0.3, 3.0),
              F("desplazamiento", "Desplazar la paleta", 0.0, 0.0, 1.0), F("ciclo", "Ciclar la paleta (vueltas/s)", 0.0, 0.0, 1.0),
              B("invertir", "Invertir", False), F("mezcla", "Intensidad", 1.0, 0, 1)]

    def process(self, frame, ctx):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        g = np.clip((g - 0.5) * self.value("contraste", ctx) + 0.5, 0, 1) ** (1.0 / self.value("gamma", ctx))
        if self.value("invertir", ctx):
            g = 1.0 - g
        g8 = (g * 255).astype(np.uint8)
        name = self.value("paleta", ctx)
        if name in _CV_MAPS:
            heat = cv2.applyColorMap(g8, _CV_MAPS[name])
        else:
            cyc = self.value("ciclo", ctx) > 0
            heat = palettes.apply(g8, name, self.value("desplazamiento", ctx) + ctx.time * self.value("ciclo", ctx), cyclic=cyc)
        return _mix(frame, heat, self.value("mezcla", ctx))
