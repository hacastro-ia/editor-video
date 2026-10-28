"""Paletas de color (degradados) y utilidades para mapear luminosidad a color."""
from __future__ import annotations

import numpy as np

# Paradas de color en RGB. Se interpolan de forma uniforme entre 0 y 1.
STOPS = {
    "arcoíris": [(255, 0, 0), (255, 200, 0), (60, 255, 0), (0, 230, 255), (60, 60, 255), (255, 0, 200)],
    "vaporwave": [(30, 0, 70), (255, 113, 206), (1, 205, 254), (5, 255, 161), (185, 103, 255), (255, 251, 150)],
    "ácido": [(8, 0, 30), (120, 0, 255), (0, 255, 200), (250, 255, 0)],
    "atardecer": [(15, 0, 45), (150, 20, 120), (255, 90, 50), (255, 200, 110), (255, 245, 200)],
    "océano": [(0, 8, 35), (0, 70, 150), (0, 190, 200), (190, 255, 240)],
    "fuego": [(0, 0, 0), (110, 10, 0), (255, 90, 0), (255, 220, 90), (255, 255, 255)],
    "tóxico": [(0, 8, 0), (0, 110, 40), (110, 255, 0), (235, 255, 150)],
    "hielo": [(4, 8, 28), (30, 90, 200), (140, 215, 255), (255, 255, 255)],
    "magma": [(0, 0, 4), (80, 18, 123), (182, 54, 121), (251, 135, 97), (252, 253, 191)],
    "neón rosa y cian": [(10, 0, 30), (255, 0, 140), (0, 240, 255), (255, 255, 255)],
    "ámbar (monitor)": [(0, 0, 0), (90, 40, 0), (255, 150, 20), (255, 225, 150)],
    "verde fósforo": [(0, 0, 0), (0, 70, 20), (60, 255, 90), (200, 255, 210)],
    "blanco y negro": [(0, 0, 0), (255, 255, 255)],
    "sepia": [(20, 10, 0), (110, 70, 30), (220, 180, 120), (255, 245, 220)],
}
NAMES = list(STOPS.keys())
_cache: dict = {}


def lut(name, n=256, shift=0.0, cyclic=False):
    """Tabla (n, 3) uint8 en orden BGR para mapear un valor 0..1 a un color. shift desplaza (0..1, cíclico)."""
    key = (name, n, round(float(shift) % 1.0, 3), cyclic)
    r = _cache.get(key)
    if r is not None:
        return r
    stops = np.array(STOPS.get(name, STOPS["arcoíris"]), np.float32)
    if cyclic:
        stops = np.vstack([stops, stops[:1]])
    pos = np.linspace(0.0, 1.0, len(stops))
    t = (np.linspace(0.0, 1.0, n, endpoint=not cyclic) + shift) % 1.0 if (cyclic or shift) else np.linspace(0, 1, n)
    rgb = np.stack([np.interp(t, pos, stops[:, c]) for c in range(3)], axis=1)
    out = np.ascontiguousarray(rgb[:, ::-1].astype(np.uint8))
    if len(_cache) > 400:
        _cache.clear()
    _cache[key] = out
    return out


def apply(gray_u8, name, shift=0.0, cyclic=False):
    """Mapa de degradado: imagen en gris (uint8) -> imagen BGR."""
    return lut(name, 256, shift, cyclic)[gray_u8]
