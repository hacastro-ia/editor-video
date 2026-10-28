"""Nube de puntos: partículas, contornos de profundidad, aberración cromática y desintegración."""
from __future__ import annotations

import cv2
import numpy as np

from core.params import Effect, F, I, B, C
from core.pipeline import register
from core.utils import grid, slice_glitch

# (color de puntos, color de fondo) en BGR
PALETAS = {
    "crema": ((208, 226, 238), (46, 40, 38)),
    "blanco": ((245, 245, 245), (20, 20, 20)),
    "verde fósforo": ((140, 255, 120), (8, 15, 5)),
    "ámbar": ((60, 176, 255), (4, 10, 20)),
    "azul hielo": ((255, 225, 190), (28, 14, 8)),
}


def _blur_lowres(img, sigma, f=4):
    """Desenfoque grande y barato: se calcula a menor resolución y se vuelve a escalar."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(2, w // f), max(2, h // f)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(0.5, sigma / f))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


FUENTES_PROF = ["luminancia", "luminancia invertida", "detalle", "modelo MiDaS (más lento)"]


@register
class PointCloud(Effect):
    name = "Nube de puntos"
    category = "Nube de puntos"
    PARAMS = [
        F("densidad", "Densidad de puntos", 0.9, 0.2, 3.0, mod=("agudos", 0.35)),
        I("tamano", "Tamaño del punto", 1, 1, 4),
        F("contraste", "Contraste de profundidad", 1.4, 0.5, 3.0),
        F("dispersion", "Dispersión con el movimiento", 1.0, 0.0, 4.0, mod=("graves", 0.5)),
        F("estela", "Persistencia de partículas", 0.75, 0.0, 0.97),
        F("contornos", "Líneas de contorno", 0.5, 0.0, 1.0, mod=("volumen", 0.3)),
        I("niveles", "Niveles de contorno", 10, 3, 24),
        F("aberracion", "Aberración cromática (px)", 3.0, 0.0, 25.0, mod=("beat", 0.3)),
        F("glitch", "Glitch", 0.0, 0.0, 1.0, mod=("beat", 0.5)),
        F("vibracion", "Vibración del grano", 1.0, 0.0, 1.0),
        C("profundidad", "Fuente de profundidad", FUENTES_PROF),
        F("corte", "Corte del fondo", 0.15, 0.0, 0.9),
        C("paleta", "Paleta", list(PALETAS.keys())),
    ]

    def reset(self):
        self.rng = np.random.default_rng(7)
        self.shape = None
        self.noise = None
        self.off = (0, 0)
        self.acc = None
        self.prev_small = None
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.midas = None
        self.midas_failed = False
        self.last_depth = None

    # -- profundidad -------------------------------------------------------
    def _midas_depth(self, frame, ctx):
        if self.midas_failed:
            return None
        if self.midas is None:
            try:
                import torch
                self.torch = torch
                self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
                self.midas.eval()
                tf = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
                self.midas_tf = tf.small_transform
                self.dev = "cuda" if torch.cuda.is_available() else "cpu"
                self.midas.to(self.dev)
            except Exception as e:
                self.midas_failed = True
                self.error = f"MiDaS no disponible (¿falta 'timm' o internet?): {e}"
                return None
        if self.last_depth is None or self.last_depth.shape != frame.shape[:2] or ctx.frame_index % 2 == 0:
            torch = self.torch
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                pred = self.midas(self.midas_tf(rgb).to(self.dev))
                pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=rgb.shape[:2],
                                                       mode="bicubic", align_corners=False).squeeze().cpu().numpy()
            lo, hi = np.percentile(pred, 2), np.percentile(pred, 98)
            self.last_depth = np.clip((pred - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)
        return self.last_depth

    def _depth(self, frame, gray, ctx):
        src = self.value("profundidad", ctx)
        g = cv2.GaussianBlur(gray, (0, 0), 1.0)
        if src == "detalle":
            gf = g.astype(np.float32) / 255.0
            local = np.abs(gf - _blur_lowres(gf, 14))
            d = np.clip(local * 5.0, 0, 1)
            d = 0.6 * d + 0.4 * _blur_lowres(d, 10)
        else:
            luma = self.clahe.apply(g).astype(np.float32) / 255.0
            luma = 0.7 * luma + 0.3 * _blur_lowres(luma, 12)
            d = 1.0 - luma if src == "luminancia invertida" else luma
            if src.startswith("modelo"):
                dm = self._midas_depth(frame, ctx)
                if dm is not None:
                    d = np.clip(0.75 * dm + 0.25 * luma, 0, 1)
        c = self.value("corte", ctx)
        return np.clip((d - c) / max(1e-3, 1.0 - c), 0, 1) if c > 0 else d

    # -- proceso -----------------------------------------------------------
    def process(self, frame, ctx):
        h, w = frame.shape[:2]
        rng = self.rng
        if self.shape != (h, w):
            self.shape = (h, w)
            self.noise = rng.random((2 * h, 2 * w), dtype=np.float32)
            self.acc = None
            self.prev_small = None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        d = self._depth(frame, gray, ctx)
        d = np.clip((d - 0.5) * self.value("contraste", ctx) + 0.5, 0, 1)

        # puntos (estarcido con ruido)
        if rng.random() < self.value("vibracion", ctx):
            self.off = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        oy, ox = self.off
        dots = (d * self.value("densidad", ctx) > self.noise[oy:oy + h, ox:ox + w]).astype(np.uint8)
        k = self.value("tamano", ctx)
        if k > 1:
            dots = cv2.dilate(dots, np.ones((k, k), np.uint8))
        v = dots.astype(np.float32)

        # movimiento: las partículas se arrastran y se dispersan
        sw = 256
        small = cv2.resize(gray, (sw, max(2, int(h * sw / w))), interpolation=cv2.INTER_AREA) if w > sw else gray
        if self.prev_small is not None and self.prev_small.shape == small.shape and self.acc is not None:
            flow = cv2.calcOpticalFlowFarneback(small, self.prev_small, None, 0.5, 2, 13, 2, 5, 1.1, 0)
            flow = cv2.resize(flow, (w, h)) * (w / small.shape[1])
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            mask = _blur_lowres(np.clip(mag / (0.02 * w), 0, 1).astype(np.float32), 6)
            mask = np.clip(mask * 3.0, 0, 1)
            nz = rng.standard_normal((max(2, h // 2), max(2, w // 2), 2), dtype=np.float32)
            jit = cv2.resize(nz, (w, h), interpolation=cv2.INTER_NEAREST) * (mask[..., None] * self.value("dispersion", ctx) * 3.0)
            gx, gy = grid(w, h)
            carry = cv2.remap(self.acc, gx + flow[..., 0] + jit[..., 0], gy + flow[..., 1] + jit[..., 1],
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            carry *= self.value("estela", ctx) * (0.1 + 0.9 * mask)
            v = np.maximum(v, carry)
        self.acc = v.copy()
        self.prev_small = small

        # líneas de contorno (curvas de nivel de la profundidad)
        cont = self.value("contornos", ctx)
        if cont > 0.02:
            ds = cv2.GaussianBlur(d, (0, 0), 2.5)
            q = (ds * self.value("niveles", ctx)).astype(np.int16)
            lines = np.zeros((h, w), np.float32)
            lines[:, 1:] = q[:, 1:] != q[:, :-1]
            lines[1:, :] = np.maximum(lines[1:, :], q[1:, :] != q[:-1, :])
            v = np.maximum(v, ds * 0.45 * cont)
            v = v * (1.0 - 0.9 * cont * lines)

        fg, bg = PALETAS[self.value("paleta", ctx)]
        fg, bg = np.array(fg, np.float32), np.array(bg, np.float32)
        # aberración cromática solo en los bordes: se desplaza la versión suavizada de cada canal
        # y se suma la diferencia, así el grano queda neutro y el color aparece como franjas
        a = int(round(self.value("aberracion", ctx)))
        chans = np.repeat(v[..., None], 3, axis=2)          # orden BGR
        if a > 0:
            vs = cv2.GaussianBlur(v, (0, 0), 2.0)
            chans[..., 2] += np.roll(vs, -a, axis=1) - vs      # rojo
            chans[..., 0] += np.roll(vs, a, axis=1) - vs       # azul
            np.clip(chans, 0.0, 1.0, out=chans)
        img = np.clip(bg + (fg - bg) * chans, 0, 255).astype(np.uint8)
        gl = self.value("glitch", ctx)
        if gl > 0.05:
            img = slice_glitch(img, gl, 0, 10, rng)
        return img
