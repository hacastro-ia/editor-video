"""Datamosh en tiempo real, simulado con flujo óptico.

Imita lo que ocurre al romper la compresión de un video: sin fotogramas clave (I-frames), el movimiento del video
nuevo (los «vectores») se aplica sobre la imagen anterior y los píxeles se arrastran y se funden.
Los modos están inspirados en las técnicas de Datamosher Pro:
  · Remover I-frame  → «arrastre» (la imagen retenida se mueve con el vídeo nuevo)
  · Repetir fotogramas P → «Bloom» (el mismo movimiento se repite una y otra vez)
  · Modos de vectores de movimiento (FFglitch) → «stretch», «fluid» y «pack»
Para un datamosh REAL sobre un archivo (manipulando el códec) usa Herramientas → Datamosh real.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.params import Effect, F, I, B, C, Btn
from core.pipeline import register
from core.utils import grid


def _small_gray(frame, width=320):
    h, w = frame.shape[:2]
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if w > width:
        g = cv2.resize(g, (width, max(2, int(h * width / w))), interpolation=cv2.INTER_AREA)
    return g


def _farneback(cur, prev):
    """Flujo de cur hacia prev (mapa «hacia atrás»: para cada píxel de cur, dónde estaba en prev)."""
    return cv2.calcOpticalFlowFarneback(cur, prev, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def _to_full(flow, w, h, small_w):
    return cv2.resize(flow, (w, h)) * (w / small_w)


def _blockify(flow, block):
    """Promedia el flujo por macrobloques (como los vectores de un códec) y lo amplía por vecino más cercano."""
    h, w = flow.shape[:2]
    bw, bh = max(1, w // block), max(1, h // block)
    return cv2.resize(cv2.resize(flow, (bw, bh), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_NEAREST)


def _beat_edge(effect, ctx):
    b = ctx.audio.get("beat", 0.0)
    hit = b > 0.95 and effect._last_beat <= 0.95
    effect._last_beat = b
    return hit


@register
class DatamoshBend(Effect):
    name = "Datamosh: Pixel-bending"
    category = "Datamosh"
    HELP = {"modo": {
        "arrastre (hold + vectores)": "Clásico: la imagen retenida se arrastra con el movimiento del video nuevo.",
        "stretch (vector promedio)": "Todo el cuadro se estira en la dirección del movimiento medio.",
        "fluid (suave y persistente)": "Vectores muy suaves con inercia: las formas se derriten y fluyen.",
        "pack (bloques de códec)": "Vectores por macrobloques, con el aspecto cuadriculado de un códec roto.",
    }}
    PARAMS = [C("modo", "Modo", list(HELP["modo"].keys())),
              F("fuerza", "Fuerza de arrastre", 1.0, 0.0, 4.0),
              F("retencion", "Inercia del movimiento", 0.0, 0.0, 0.98),
              I("bloque", "Tamaño de macrobloque (px)", 16, 4, 64),
              F("cuantizar", "Cuantizar vectores", 0.0, 0.0, 1.0),
              F("suavizado", "Suavizado del movimiento", 4.0, 0.0, 25.0),
              F("ruido", "Ruido de vectores", 0.0, 0.0, 1.0),
              F("refresco", "Refresco de imagen nueva", 0.02, 0.0, 1.0),
              I("iframe", "Fotograma clave cada N (0 = nunca)", 0, 0, 600),
              B("iframe_beat", "Fotograma clave con el beat", False),
              Btn("reiniciar", "Reiniciar ahora (nuevo I-frame)")]

    def reset(self):
        self.acc = None
        self.prev = None
        self.prev_flow = None
        self.since = 0
        self._last_beat = 0.0
        self.force = False
        self.rng = np.random.default_rng(3)

    def on_button(self, key):
        if key == "reiniciar":
            self.force = True

    def process(self, frame, ctx):
        h, w = frame.shape[:2]
        small = _small_gray(frame)
        if self.acc is None or self.acc.shape[:2] != (h, w) or self.prev is None or self.prev.shape != small.shape:
            self.acc = frame.astype(np.float32)
            self.prev = small
            self.prev_flow = None
            self.since = 0
            return frame
        modo = self.value("modo", ctx)
        flow = _to_full(_farneback(small, self.prev), w, h, small.shape[1])
        s = self.value("suavizado", ctx)
        if modo.startswith("fluid"):
            s = max(s, 12.0)
        if s > 0.3:
            flow = cv2.GaussianBlur(flow, (0, 0), s)
        if modo.startswith("stretch"):
            flow = flow.mean(axis=(0, 1), keepdims=True) * 1.6 + flow * 0.25
            flow = np.broadcast_to(flow, (h, w, 2)).astype(np.float32)
        elif modo.startswith("pack"):
            flow = _blockify(flow, self.value("bloque", ctx))
        q = self.value("cuantizar", ctx)
        if q > 0.01:
            step = 1.0 + 7.0 * q
            flow = np.round(flow / step) * step
        flow = flow * self.value("fuerza", ctx)
        ret = self.value("retencion", ctx)
        if modo.startswith("fluid"):
            ret = max(ret, 0.85)
        if ret > 0.01:
            if self.prev_flow is not None and self.prev_flow.shape == flow.shape:
                flow = flow + ret * self.prev_flow
            lim = 0.3 * w
            flow = np.clip(flow, -lim, lim)
            self.prev_flow = flow
        nz = self.value("ruido", ctx)
        if nz > 0.01:
            b = self.value("bloque", ctx)
            gw, gh = max(2, w // b), max(2, h // b)
            hit = self.rng.random((gh, gw)) < nz * 0.12
            vec = self.rng.normal(0, 6 + 30 * nz, (gh, gw, 2)).astype(np.float32) * hit[..., None]
            flow = flow + cv2.resize(vec, (w, h), interpolation=cv2.INTER_NEAREST)
        gx, gy = grid(w, h)
        self.acc = cv2.remap(self.acc, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        self.since += 1
        n = self.value("iframe", ctx)
        key = self.force or (n > 0 and self.since >= n) or (self.value("iframe_beat", ctx) and _beat_edge(self, ctx))
        if key:
            self.acc = frame.astype(np.float32)
            self.since = 0
            self.force = False
            self.prev_flow = None
        else:
            r = self.value("refresco", ctx)
            self.acc = self.acc * (1 - r) + frame.astype(np.float32) * r
        self.prev = small
        return np.clip(self.acc, 0, 255).astype(np.uint8)


@register
class DatamoshBloom(Effect):
    name = "Datamosh: Bloom (repetir delta)"
    category = "Datamosh"
    HELP = {"disparo": {
        "botón": "Pulsa «Disparar bloom» (o la tecla B) cuando quieras.",
        "beat": "Se dispara solo en cada golpe de la música.",
        "cada N fotogramas": "Se dispara de forma periódica.",
        "movimiento fuerte": "Se dispara cuando hay mucho movimiento en la imagen.",
    }}
    PARAMS = [C("disparo", "Disparo", list(HELP["disparo"].keys()), "beat"),
              I("duracion", "Duración (fotogramas)", 40, 5, 240), F("fuerza", "Fuerza", 1.0, 0.0, 3.0),
              F("crecimiento", "Crecimiento por fotograma", 1.0, 0.9, 1.15), I("bloque", "Tamaño de macrobloque (px)", 16, 4, 64),
              F("suavizado", "Suavizado", 2.0, 0.0, 15.0), F("salida", "Velocidad de regreso al video", 0.08, 0.01, 0.4),
              I("intervalo", "Intervalo (modo periódico)", 120, 20, 600), F("umbral", "Umbral de movimiento (px)", 3.0, 0.5, 12.0),
              Btn("disparar", "Disparar bloom")]

    def reset(self):
        self.acc = None
        self.prev = None
        self.left = 0
        self.vec = None
        self.gain = 1.0
        self.since = 0
        self._last_beat = 0.0
        self.pending = False

    def on_button(self, key):
        if key == "disparar":
            self.pending = True

    def process(self, frame, ctx):
        h, w = frame.shape[:2]
        small = _small_gray(frame)
        if self.prev is None or self.prev.shape != small.shape:
            self.prev = small
            return frame
        modo = self.value("disparo", ctx)
        fire = self.pending
        self.pending = False
        self.since += 1
        flow = None
        if modo == "beat":
            fire = fire or _beat_edge(self, ctx)
        elif modo == "cada N fotogramas":
            fire = fire or self.since >= self.value("intervalo", ctx)
        elif modo == "movimiento fuerte" and self.left == 0:
            flow = _farneback(small, self.prev)
            fire = fire or float(np.abs(flow).mean()) * (w / small.shape[1]) > self.value("umbral", ctx)
        if fire and self.left == 0:
            if flow is None:
                flow = _farneback(small, self.prev)
            v = _to_full(flow, w, h, small.shape[1])
            if self.value("suavizado", ctx) > 0.3:
                v = cv2.GaussianBlur(v, (0, 0), self.value("suavizado", ctx))
            self.vec = _blockify(v, self.value("bloque", ctx)) * self.value("fuerza", ctx) * 2.0
            self.acc = frame.astype(np.float32)
            self.left, self.gain, self.since = self.value("duracion", ctx), 1.0, 0
        self.prev = small
        if self.acc is None:
            return frame
        gx, gy = grid(w, h)
        if self.left > 0:                                  # repetir el mismo movimiento una y otra vez
            self.acc = cv2.remap(self.acc, gx + self.vec[..., 0] * self.gain, gy + self.vec[..., 1] * self.gain,
                                 cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            self.gain *= self.value("crecimiento", ctx)
            self.left -= 1
            return np.clip(self.acc, 0, 255).astype(np.uint8)
        r = self.value("salida", ctx)                       # termina: se funde de vuelta al video
        self.acc = self.acc * (1 - r) + frame.astype(np.float32) * r
        if float(np.abs(self.acc - frame).mean()) < 1.5:
            self.acc = None
            return frame
        return np.clip(self.acc, 0, 255).astype(np.uint8)


@register
class DatamoshFreeze(Effect):
    name = "Datamosh: Congelar y contagiar"
    category = "Datamosh"
    PARAMS = [F("sensibilidad", "Sensibilidad al movimiento", 22, 4, 80), I("expansion", "Expansión del contagio", 4, 0, 20),
              F("suavidad", "Suavidad de los bordes", 2.0, 0.0, 10.0), F("limpieza", "Regreso a la imagen real", 0.0, 0.0, 0.3),
              B("invertir", "Invertir (congelar lo que se mueve)", False),
              I("refresco_cada", "Reiniciar cada N (0 = nunca)", 0, 0, 900), B("refresco_beat", "Reiniciar con el beat", False),
              Btn("reiniciar", "Reiniciar ahora")]

    def reset(self):
        self.frozen = None
        self.prev = None
        self.since = 0
        self._last_beat = 0.0
        self.force = False

    def on_button(self, key):
        if key == "reiniciar":
            self.force = True

    def process(self, frame, ctx):
        if self.frozen is None or self.frozen.shape != frame.shape or self.prev is None:
            self.frozen = frame.copy()
            self.prev = frame
            self.since = 0
            return frame
        diff = cv2.absdiff(frame, self.prev).max(axis=2)
        mask = (diff > self.value("sensibilidad", ctx)).astype(np.uint8)
        k = self.value("expansion", ctx)
        if k > 0:
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
        m = mask.astype(np.float32)
        sm = self.value("suavidad", ctx)
        if sm > 0.3:
            m = cv2.GaussianBlur(m, (0, 0), sm)
        if self.value("invertir", ctx):
            m = 1.0 - m
        m = m[..., None]
        self.frozen = (frame * m + self.frozen * (1 - m)).astype(np.uint8)
        self.since += 1
        n = self.value("refresco_cada", ctx)
        if self.force or (n > 0 and self.since >= n) or (self.value("refresco_beat", ctx) and _beat_edge(self, ctx)):
            self.frozen = frame.copy()
            self.since = 0
            self.force = False
        lim = self.value("limpieza", ctx)
        if lim > 0:
            self.frozen = cv2.addWeighted(self.frozen, 1 - lim, frame, lim, 0)
        self.prev = frame
        return self.frozen


@register
class CorruptBlocks(Effect):
    name = "Datamosh: Bloques corruptos"
    category = "Datamosh"
    PARAMS = [I("bloque", "Tamaño del bloque", 32, 8, 128), F("cantidad", "Cantidad de bloques", 0.05, 0, 0.6, mod=("beat", 0.4)),
              I("duracion", "Duración (fotogramas)", 8, 1, 40),
              C("modo", "Modo", ["congelar", "desplazar", "franjas", "intercambiar"]),
              B("solo_color", "Corromper solo un canal de color", False)]

    def reset(self):
        self.rng = np.random.default_rng(99)
        self.active = {}

    def process(self, frame, ctx):
        h, w = frame.shape[:2]
        b = self.value("bloque", ctx)
        gh, gw = max(1, h // b), max(1, w // b)
        modo = self.value("modo", ctx)
        rng = self.rng
        dur = self.value("duracion", ctx)
        n_new = int(round(gh * gw * self.value("cantidad", ctx) / max(1, dur) * 2))
        for _ in range(n_new):
            gy, gx = int(rng.integers(0, gh)), int(rng.integers(0, gw))
            y, x = gy * b, gx * b
            life = int(rng.integers(2, dur + 2))
            if modo == "congelar":
                data = frame[y:y + b, x:x + b].copy()
            elif modo == "franjas":
                data = frame[y:y + max(2, b // 3), :].copy()
            elif modo == "intercambiar":
                data = (int(rng.integers(0, gh)) * b, int(rng.integers(0, gw)) * b)
            else:
                data = (int(rng.integers(-3, 4)) * b, int(rng.integers(-3, 4)) * b)
            self.active[(gy, gx)] = [life, data]
        out = frame.copy()
        ch = int(rng.integers(0, 3)) if self.value("solo_color", ctx) else None
        dead = []
        for (gy, gx), item in self.active.items():
            y, x = gy * b, gx * b
            bh, bw = min(b, h - y), min(b, w - x)
            if modo == "congelar":
                src = item[1][:bh, :bw]
            elif modo == "franjas":
                bh = min(item[1].shape[0], h - y)
                src = item[1][:bh, :]
                x, bw = 0, w
            else:
                dy, dx = item[1]
                if modo == "intercambiar":
                    sy, sx = int(np.clip(dy, 0, h - b)), int(np.clip(dx, 0, w - b))
                else:
                    sy, sx = int(np.clip(y + dy, 0, h - b)), int(np.clip(x + dx, 0, w - b))
                src = frame[sy:sy + bh, sx:sx + bw]
            if ch is None:
                out[y:y + bh, x:x + bw] = src[:bh, :bw]
            else:
                out[y:y + bh, x:x + bw, ch] = src[:bh, :bw, ch]
            item[0] -= 1
            if item[0] <= 0:
                dead.append((gy, gx))
        for k in dead:
            self.active.pop(k, None)
        return out
