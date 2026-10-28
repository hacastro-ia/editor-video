"""Efectos de acabado profesional: bloom, slit-scan, lente, pantalla CRT y halftone."""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from core import palettes
from core.params import Effect, F, I, B, C, Btn
from core.pipeline import register
from core.utils import fast_blur, grid, grid_lo, remap_lo


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# --------------------------------------------------------------------------- bloom
@register
class Bloom(Effect):
    name = "Bloom (resplandor)"
    category = "Estilo TouchDesigner"
    PARAMS = [F("umbral", "Umbral de brillo", 0.75, 0.0, 1.0), F("suavidad", "Suavidad del umbral", 0.2, 0.0, 0.5),
              F("intensidad", "Intensidad", 0.8, 0.0, 3.0, mod=("graves", 0.3)), F("radio", "Radio (px)", 18.0, 2.0, 80.0),
              I("niveles", "Niveles de resplandor", 3, 1, 4), C("tinte", "Tinte", ["natural", "cálido", "frío", "neón rosa y cian", "fuego"]),
              B("solo_brillo", "Ver solo el resplandor", False)]

    def process(self, frame, ctx):
        h, w = frame.shape[:2]
        sm = cv2.resize(frame, (max(8, w // 2), max(8, h // 2)), interpolation=cv2.INTER_AREA)
        f = sm.astype(np.float32) * (1.0 / 255.0)
        lum = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        thr, knee = self.value("umbral", ctx), max(self.value("suavidad", ctx), 1e-3)
        m = _smoothstep(thr - knee, thr + knee, lum)[..., None]
        bright = (f * m * 255).astype(np.uint8)
        glow = np.zeros(bright.shape, np.float32)
        r = self.value("radio", ctx) * w / 640.0 / 2.0
        n = self.value("niveles", ctx)
        for k in range(n):
            glow += fast_blur(bright, r * (2 ** k) / (2 ** (n - 1)) + 1.5).astype(np.float32) / n
        tint = self.value("tinte", ctx)
        if tint == "cálido":
            glow *= np.array([0.65, 0.9, 1.15], np.float32)
        elif tint == "frío":
            glow *= np.array([1.2, 1.0, 0.75], np.float32)
        elif tint in ("neón rosa y cian", "fuego"):
            g = cv2.cvtColor(np.clip(glow, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            glow = palettes.apply(g, tint).astype(np.float32)
        glow *= self.value("intensidad", ctx) * 1.6
        glow_u8 = cv2.resize(np.clip(glow, 0, 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
        return glow_u8 if self.value("solo_brillo", ctx) else cv2.add(frame, glow_u8)


# --------------------------------------------------------------------------- slit-scan
@register
class SlitScan(Effect):
    name = "Slit-scan (Time Machine)"
    category = "Estilo TouchDesigner"
    HELP = {"modo": {
        "vertical (cada fila en un tiempo)": "Cada fila de la imagen viene de un instante distinto: el movimiento se curva en el espacio.",
        "horizontal (cada columna)": "Igual, pero por columnas.",
        "radial (del centro hacia fuera)": "El centro es el presente y los bordes el pasado: ondas de tiempo.",
        "diagonal": "El retardo cambia en diagonal.",
    }}
    PARAMS = [C("modo", "Modo", list(HELP["modo"].keys())), I("profundidad", "Profundidad (fotogramas)", 40, 4, 90),
              F("curva", "Curva del retardo", 1.0, 0.3, 3.0), B("invertir", "Invertir el sentido", False),
              F("onda", "Ondulación del retardo", 0.0, 0.0, 1.0), F("velocidad_onda", "Velocidad de la ondulación", 0.5, 0.0, 3.0)]

    def reset(self):
        self.buf = deque(maxlen=90)

    def process(self, frame, ctx):
        h, w = frame.shape[:2]
        if self.buf and self.buf[-1].shape != frame.shape:
            self.buf.clear()
        cap = max(4, min(self.value("profundidad", ctx), int(150e6 / (h * w * 3))))
        if self.buf.maxlen != cap:
            self.buf = deque(self.buf, maxlen=cap)
        self.buf.append(frame.copy())
        n = len(self.buf)
        if n < 2:
            return frame
        modo = self.value("modo", ctx)
        curva, inv = self.value("curva", ctx), self.value("invertir", ctx)
        onda, vel = self.value("onda", ctx), self.value("velocidad_onda", ctx)
        out = frame.copy()

        def delays(t):                                      # t en 0..1 -> índice de fotograma pasado (0 = el más viejo)
            t = np.clip(t, 0, 1) ** curva
            t = 1.0 - t if inv else t
            return np.clip(((1.0 - t) * (n - 1)).round().astype(int), 0, n - 1)

        if modo.startswith("vertical") or modo.startswith("horizontal"):
            vert = modo.startswith("vertical")
            m = h if vert else w
            t = np.arange(m, dtype=np.float32) / max(m - 1, 1)
            if onda > 0.01:
                t = t + onda * 0.15 * np.sin(t * 9 + ctx.time * vel * 3)
            idx = delays(t)
            for k in np.unique(idx):
                sel = np.where(idx == k)[0]
                if vert:
                    out[sel] = self.buf[k][sel]
                else:
                    out[:, sel] = self.buf[k][:, sel]
            return out
        X, Y = grid_lo(w, h, 4)
        if modo.startswith("radial"):
            t = np.sqrt(((X - w / 2) / (w / 2)) ** 2 + ((Y - h / 2) / (h / 2)) ** 2) / 1.4142
        else:
            t = (X / w + Y / h) / 2.0
        if onda > 0.01:
            t = t + onda * 0.12 * np.sin(t * 14 - ctx.time * vel * 3)
        key = (modo, n, round(curva, 2), inv, w, h)
        if onda > 0.01 or getattr(self, "_mkey", None) != key:
            idx = cv2.resize(delays(t).astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(int)
            step = int(np.ceil(len(np.unique(idx)) / 22))     # limitar el trabajo: se agrupan los retardos
            if step > 1:
                idx = (idx // step * step).clip(0, n - 1)
            masks = [(int(k), (idx == k).astype(np.uint8)) for k in np.unique(idx)]
            if onda <= 0.01:
                self._mkey, self._masks = key, masks
        else:
            masks = self._masks
        for k, msk in masks:
            cv2.copyTo(self.buf[k], msk, out)
        return out


# --------------------------------------------------------------------------- lente
@register
class Lens(Effect):
    name = "Lente cinematográfica"
    category = "Estilo TouchDesigner"
    PARAMS = [F("aberracion", "Aberración cromática", 0.35, 0.0, 1.0), F("barril", "Distorsión (barril / cojín)", 0.0, -0.5, 0.5),
              F("vineta", "Viñeta", 0.4, 0.0, 1.0), F("halacion", "Halación (resplandor cálido)", 0.3, 0.0, 1.0),
              F("destello", "Destello anamórfico", 0.0, 0.0, 1.0), F("grano", "Grano de película", 0.25, 0.0, 1.0),
              F("nitidez", "Nitidez", 0.0, 0.0, 2.0)]

    def reset(self):
        self.rng = np.random.default_rng(17)
        self._vig = None
        self._gn = None

    def process(self, frame, ctx):
        h, w = frame.shape[:2]
        bar, ab = self.value("barril", ctx), self.value("aberracion", ctx)
        out = frame
        if abs(bar) > 0.005 or ab > 0.01:
            X, Y = grid_lo(w, h, 2)
            dx, dy = (X - w / 2) / (w / 2), (Y - h / 2) / (h / 2)
            r2 = dx * dx + dy * dy
            b, g, r = cv2.split(frame)
            chans = []
            for ch, k in ((b, 1.0 - 0.012 * ab), (g, 1.0), (r, 1.0 + 0.012 * ab)):
                s = k * (1.0 + bar * r2)
                chans.append(remap_lo(ch, w / 2 + dx * s * w / 2, h / 2 + dy * s * h / 2))
            out = cv2.merge(chans)
        n = self.value("nitidez", ctx)
        if n > 0.02:
            out = cv2.addWeighted(out, 1.0 + n, cv2.GaussianBlur(out, (0, 0), 1.5), -n, 0)
        hal, dst = self.value("halacion", ctx), self.value("destello", ctx)
        if hal > 0.02 or dst > 0.02:                       # se calcula a media resolución
            sm = cv2.resize(out, (max(8, w // 2), max(8, h // 2)), interpolation=cv2.INTER_AREA)
            lum = np.clip((cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0 - 0.7) / 0.3, 0, 1)
            bright = (lum[..., None] * sm).astype(np.uint8)
            add = np.zeros(sm.shape, np.float32)
            if hal > 0.02:
                add += fast_blur(bright, 8 * w / 640.0).astype(np.float32) * np.array([0.35, 0.8, 1.3], np.float32) * hal
            if dst > 0.02:
                st = cv2.resize(bright, (max(8, w // 16), sm.shape[0]), interpolation=cv2.INTER_AREA)
                st = cv2.GaussianBlur(st, (0, 0), sigmaX=60 * w / 640.0 / 16, sigmaY=1.0)
                add += cv2.resize(st, (sm.shape[1], sm.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32) * np.array(
                    [1.2, 0.8, 0.5], np.float32) * dst * 1.2
            out = cv2.add(out, cv2.resize(np.clip(add, 0, 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR))
        v = self.value("vineta", ctx)
        if v > 0.01:
            key = (h, w, round(v, 2))
            if self._vig is None or self._vig[0] != key:
                X, Y = grid_lo(w, h, 2)
                r2 = ((X - w / 2) / (w / 2)) ** 2 + ((Y - h / 2) / (h / 2)) ** 2
                fall = cv2.resize(1.0 - v * np.clip(r2 / 2.0, 0, 1) ** 1.2, (w, h), interpolation=cv2.INTER_LINEAR)
                self._vig = (key, cv2.merge([(fall * 255).astype(np.uint8)] * 3))
            out = cv2.multiply(out, self._vig[1], scale=1 / 255.0)
        gr = self.value("grano", ctx)
        if gr > 0.01:
            if self._gn is None or self._gn.shape != (2 * h, 2 * w):
                self._gn = self.rng.standard_normal((2 * h, 2 * w)).astype(np.float32)
            oy, ox = int(self.rng.integers(0, h)), int(self.rng.integers(0, w))
            g = self._gn[oy:oy + h, ox:ox + w] * (14 * gr)
            pos, neg = np.clip(g, 0, 255).astype(np.uint8), np.clip(-g, 0, 255).astype(np.uint8)
            out = cv2.subtract(cv2.add(out, cv2.merge([pos] * 3)), cv2.merge([neg] * 3))
        return out


# --------------------------------------------------------------------------- pantalla CRT
CRT_PRESETS = {
    "TV de tubo": dict(curvatura=0.45, esquinas=0.6, lineas=0.5, lineas_pantalla=240, mascara="máscara de sombra", mascara_fuerza=0.35,
                       resplandor=0.5, convergencia=1.6, desenfoque=0.9, parpadeo=0.15, barra=0.25, ruido=0.12, persistencia=0.25,
                       fosforo="color", saturacion=1.1, brillo=1.1, jitter=0.2),
    "Monitor de PC": dict(curvatura=0.15, esquinas=0.3, lineas=0.3, lineas_pantalla=480, mascara="rejilla de apertura (RGB)", mascara_fuerza=0.25,
                          resplandor=0.3, convergencia=0.6, desenfoque=0.4, parpadeo=0.05, barra=0.0, ruido=0.03, persistencia=0.1,
                          fosforo="color", saturacion=1.0, brillo=1.05, jitter=0.0),
    "Arcade": dict(curvatura=0.3, esquinas=0.5, lineas=0.7, lineas_pantalla=224, mascara="rejilla de apertura (RGB)", mascara_fuerza=0.45,
                   resplandor=0.8, convergencia=1.0, desenfoque=0.7, parpadeo=0.1, barra=0.0, ruido=0.05, persistencia=0.3,
                   fosforo="color", saturacion=1.35, brillo=1.2, jitter=0.1),
    "Trinitron": dict(curvatura=0.1, esquinas=0.35, lineas=0.2, lineas_pantalla=480, mascara="rejilla de apertura (RGB)", mascara_fuerza=0.4,
                      resplandor=0.45, convergencia=0.4, desenfoque=0.35, parpadeo=0.04, barra=0.0, ruido=0.02, persistencia=0.12,
                      fosforo="color", saturacion=1.2, brillo=1.1, jitter=0.0),
    "Terminal verde": dict(curvatura=0.3, esquinas=0.55, lineas=0.55, lineas_pantalla=300, mascara="ninguna", mascara_fuerza=0.0,
                           resplandor=0.9, convergencia=0.3, desenfoque=0.9, parpadeo=0.2, barra=0.35, ruido=0.08, persistencia=0.45,
                           fosforo="verde fósforo", saturacion=1.0, brillo=1.15, jitter=0.1),
    "Terminal ámbar": dict(curvatura=0.3, esquinas=0.55, lineas=0.5, lineas_pantalla=300, mascara="ninguna", mascara_fuerza=0.0,
                           resplandor=0.8, convergencia=0.3, desenfoque=0.9, parpadeo=0.2, barra=0.3, ruido=0.08, persistencia=0.4,
                           fosforo="ámbar (monitor)", saturacion=1.0, brillo=1.15, jitter=0.1),
    "TV vieja y gastada": dict(curvatura=0.6, esquinas=0.8, lineas=0.65, lineas_pantalla=200, mascara="puntos", mascara_fuerza=0.4,
                               resplandor=0.6, convergencia=3.0, desenfoque=1.4, parpadeo=0.4, barra=0.5, ruido=0.35, persistencia=0.35,
                               fosforo="color", saturacion=0.85, brillo=1.0, jitter=0.5),
}


@register
class CRTScreen(Effect):
    name = "Pantalla CRT"
    category = "Cinta analógica"
    LIVE_WHEN_PAUSED = False
    PARAMS = [C("preset", "Preset de pantalla", list(CRT_PRESETS.keys())), Btn("aplicar", "Aplicar el preset"),
              F("curvatura", "Curvatura del cristal", 0.35, 0.0, 1.0), F("esquinas", "Esquinas redondeadas y viñeta", 0.5, 0.0, 1.0),
              F("lineas", "Líneas de barrido", 0.45, 0.0, 1.0), I("lineas_pantalla", "Líneas de la pantalla", 240, 120, 720),
              C("mascara", "Máscara de fósforo", ["ninguna", "rejilla de apertura (RGB)", "máscara de sombra", "puntos"]),
              F("mascara_fuerza", "Fuerza de la máscara", 0.3, 0.0, 1.0), F("resplandor", "Resplandor del fósforo", 0.4, 0.0, 1.5),
              F("convergencia", "Error de convergencia RGB (px)", 1.0, 0.0, 6.0), F("desenfoque", "Desenfoque del haz", 0.6, 0.0, 3.0),
              F("persistencia", "Persistencia del fósforo", 0.2, 0.0, 0.9), F("parpadeo", "Parpadeo de brillo", 0.15, 0.0, 1.0),
              F("barra", "Barra de interferencia", 0.0, 0.0, 1.0), F("ruido", "Ruido / nieve", 0.1, 0.0, 1.0),
              F("jitter", "Temblor horizontal", 0.0, 0.0, 1.0),
              C("fosforo", "Color del fósforo", ["color", "verde fósforo", "ámbar (monitor)", "blanco y negro"]),
              F("saturacion", "Saturación", 1.0, 0.0, 2.0), F("brillo", "Brillo", 1.05, 0.5, 1.8)]

    HELP = {"preset": {k: "Aplica un conjunto de valores; después puedes ajustarlos a mano." for k in CRT_PRESETS}}

    def reset(self):
        self.rng = np.random.default_rng(4)
        self.prev = None
        self.n = 0

    def on_button(self, key):
        if key == "aplicar":
            for k, v in CRT_PRESETS[self.params["preset"].value].items():
                self.params[k].value = v

    def process(self, frame, ctx):
        h, w = frame.shape[:2]
        sc = w / 640.0
        self.n += 1
        rng = self.rng
        img = frame
        # --- geometría: cristal curvo, temblor horizontal
        cur = self.value("curvatura", ctx)
        jit = self.value("jitter", ctx)
        X, Y = grid_lo(w, h, 2)
        dx, dy = (X - w / 2) / (w / 2), (Y - h / 2) / (h / 2)
        mx, my = X.copy(), Y.copy()
        if cur > 0.01:
            r2 = dx * dx + dy * dy
            k = 1.0 + cur * 0.12 * r2
            mx, my = w / 2 + dx * k * w / 2 / (1 + cur * 0.06), h / 2 + dy * k * h / 2 / (1 + cur * 0.06)
        if jit > 0.01:
            rows = rng.standard_normal(mx.shape[0]).astype(np.float32)
            rows = cv2.GaussianBlur(rows.reshape(-1, 1), (0, 0), 1.2).reshape(-1) * jit * 3.0 * sc
            mx = mx + rows[:, None]
            if rng.random() < 0.04 * jit:
                mx = mx + rng.uniform(-8, 8) * sc
        if cur > 0.01 or jit > 0.01:
            img = remap_lo(frame, mx, my, cv2.BORDER_CONSTANT)
        # --- convergencia RGB y desenfoque del haz
        conv = self.value("convergencia", ctx) * sc
        bl = self.value("desenfoque", ctx)
        if bl > 0.05:
            img = cv2.GaussianBlur(img, (0, 0), sigmaX=bl * sc, sigmaY=max(0.3, bl * sc * 0.35))
        if conv > 0.1:
            b, g, r = cv2.split(img)
            s = int(round(conv))
            b = np.roll(b, -s, axis=1)
            r = np.roll(r, s, axis=1)
            img = cv2.merge([b, g, r])
        f = img.astype(np.float32) * (1.0 / 255.0)
        # --- color del fósforo
        fos = self.value("fosforo", ctx)
        if fos != "color":
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            f = palettes.apply(g, "blanco y negro" if fos == "blanco y negro" else fos).astype(np.float32) * (1.0 / 255.0)
        else:
            sat = self.value("saturacion", ctx)
            if abs(sat - 1) > 0.02:
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[..., None]
                f = gray + (f - gray) * sat
        # --- líneas de barrido y máscara de fósforo
        ln = self.value("lineas", ctx)
        if ln > 0.01:
            period = h / float(self.value("lineas_pantalla", ctx))
            y = np.arange(h, dtype=np.float32)
            beam = 0.5 + 0.5 * np.cos(2 * np.pi * (y + (0.5 * period if self.n % 2 else 0) * 0.3) / period)
            f *= (1.0 - ln * (1.0 - beam ** 0.8))[:, None, None]
        mk, ms = self.value("mascara", ctx), self.value("mascara_fuerza", ctx)
        if mk != "ninguna" and ms > 0.01:
            cols = np.arange(w)
            if mk == "rejilla de apertura (RGB)":
                m = np.full((1, w, 3), 1.0 - ms, np.float32)
                for c in range(3):
                    m[0, (cols + c) % 3 == 0, c] = 1.0 + ms * 0.35
                f *= m
            elif mk == "máscara de sombra":
                yy = np.arange(h)[:, None]
                idx = (cols[None, :] + (yy % 2) * 1 + (yy // 2 % 2)) % 3
                m = np.full((h, w, 3), 1.0 - ms, np.float32)
                for c in range(3):
                    m[..., c][idx == c] = 1.0 + ms * 0.35
                f *= m
            else:
                yy = np.arange(h)[:, None]
                dots = (0.5 + 0.5 * np.cos(cols[None, :] * 2.1) * np.cos(yy * 2.1)) * 0.7 + 0.3
                f *= (1.0 - ms + ms * dots)[..., None]
        # --- resplandor del fósforo (bloom)
        rs = self.value("resplandor", ctx)
        if rs > 0.01:
            lum = np.clip((cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) - 0.55) / 0.45, 0, 1)[..., None]
            bright = (f * lum * 255).astype(np.uint8)
            f = f + fast_blur(bright, 7 * sc).astype(np.float32) * (1.0 / 255.0) * rs * 1.4
        f *= self.value("brillo", ctx)
        # --- parpadeo, barra de interferencia, ruido
        fl = self.value("parpadeo", ctx)
        if fl > 0.01:
            f *= 1.0 + fl * (0.04 * (1 if self.n % 2 else -1) + 0.03 * np.sin(ctx.time * 2.0))
        bar = self.value("barra", ctx)
        if bar > 0.01:
            pos = (ctx.time * 0.18) % 1.4 - 0.2
            yy = np.arange(h, dtype=np.float32) / h
            f *= (1.0 - bar * 0.35 * np.exp(-((yy - pos) / 0.07) ** 2))[:, None, None]
        nz = self.value("ruido", ctx)
        if nz > 0.01:
            f += (rng.standard_normal((h, w)).astype(np.float32) * 0.06 * nz)[..., None]
        # --- viñeta y esquinas redondeadas
        co = self.value("esquinas", ctx)
        if co > 0.01:
            r2 = np.clip((np.abs(dx) ** 4 + np.abs(dy) ** 4) ** 0.5, 0, 1.6)
            vig = np.clip(1.0 - co * 1.1 * _smoothstep(0.55, 1.15, r2), 0, 1)
            f *= cv2.resize(vig, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
        out = np.clip(f * 255.0, 0, 255).astype(np.uint8)
        pe = self.value("persistencia", ctx)
        if pe > 0.01:
            if self.prev is not None and self.prev.shape == out.shape:
                out = np.maximum(out, cv2.convertScaleAbs(self.prev, alpha=pe))
            self.prev = out.copy()
        return out


# --------------------------------------------------------------------------- halftone
@register
class Halftone(Effect):
    name = "Halftone (retícula)"
    category = "Estilo TouchDesigner"
    PARAMS = [I("tamano", "Tamaño de la retícula (px)", 10, 4, 40), F("angulo", "Ángulo (°)", 45, 0, 90),
              C("modo", "Modo", ["puntos negros sobre blanco", "puntos de color sobre negro", "tintas CMY (imprenta)"]),
              F("contraste", "Contraste", 1.0, 0.5, 2.5), F("suavidad", "Suavidad del borde", 0.15, 0.02, 0.6),
              F("giro", "Giro automático (°/s)", 0.0, -30, 30)]

    def _tile(self):
        if getattr(self, "_t", None) is None:
            u = (np.arange(64, dtype=np.float32) + 0.5) / 64.0
            self._t = ((np.cos(2 * np.pi * u)[None, :] + np.cos(2 * np.pi * u)[:, None]) * 0.5).astype(np.float32)
        return self._t

    def process(self, frame, ctx):
        h, w = frame.shape[:2]
        s = float(self.value("tamano", ctx)) * w / 640.0
        base = np.deg2rad(self.value("angulo", ctx) + self.value("giro", ctx) * ctx.time)
        modo = self.value("modo", ctx)
        cst, soft = self.value("contraste", ctx), self.value("suavidad", ctx)

        tile = self._tile()

        def pattern(angle):
            k = 64.0 / s
            m = np.array([[np.cos(angle) * k, np.sin(angle) * k, 0.0], [-np.sin(angle) * k, np.cos(angle) * k, 0.0]], np.float32)
            return cv2.warpAffine(tile, m, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_WRAP)

        def ink(dark, angle):
            thr = 1.0 - 2.0 * np.clip(dark * cst, 0, 1)
            return np.clip((pattern(angle) - thr) / soft * 0.5 + 0.5, 0, 1)

        f = frame.astype(np.float32) * (1.0 / 255.0)
        if modo.startswith("puntos negros"):
            lum = f @ np.array([0.114, 0.587, 0.299], np.float32)
            out = np.repeat((1.0 - ink(1.0 - lum, base))[..., None], 3, axis=2)
        elif modo.startswith("puntos de color"):
            lum = f @ np.array([0.114, 0.587, 0.299], np.float32)
            out = f * ink(lum, base)[..., None]
        else:
            out = np.ones_like(f)
            for c, ang in enumerate((base + np.deg2rad(15), base + np.deg2rad(45), base + np.deg2rad(75))):
                out[..., c] = 1.0 - ink(1.0 - f[..., c], ang)
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)
