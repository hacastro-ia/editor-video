"""Utilidades compartidas por todos los módulos."""
from __future__ import annotations

import os
import shutil
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# Carpeta de datos del usuario: en el exe (frozen) _MEIPASS es temporal y la carpeta
# de instalación puede ser de solo lectura (Program Files), así que los modelos y la
# biblioteca van a una carpeta escribible del usuario.
DATA_DIR = ROOT
if getattr(sys, "frozen", False):
    _base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home())
    DATA_DIR = Path(_base) / "EditorVideo"
MODELS_DIR = DATA_DIR / "models"


def ffmpeg_exe():
    """Devuelve la ruta de FFmpeg (del sistema o del paquete imageio-ffmpeg)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def popen_kwargs():
    """Evita que en Windows aparezca una consola al llamar a FFmpeg."""
    if sys.platform.startswith("win"):
        return {"creationflags": 0x08000000}
    return {}


def ensure_model(filename: str, urls) -> Path:
    """Descarga un modelo a la carpeta models/ si todavía no existe (prueba varias URL)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / filename
    if dest.exists() and dest.stat().st_size > 20_000:
        return dest
    if isinstance(urls, str):
        urls = [urls]
    last = None
    for url in urls:
        tmp = dest.with_suffix(".part")
        try:
            with urllib.request.urlopen(url, timeout=90) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            if tmp.stat().st_size < 20_000:      # p. ej. un puntero de Git LFS, no el modelo
                raise RuntimeError("archivo descargado demasiado pequeño")
            tmp.replace(dest)
            return dest
        except Exception as e:
            last = e
            try:
                tmp.unlink()
            except OSError:
                pass
    raise RuntimeError(f"No se pudo descargar {filename}: {last}")


def tri(x):
    """Onda triangular entre 0 y 1 (periodo 2, par): sirve para reflejar sin costuras."""
    return 1.0 - np.abs(np.mod(x, 2.0) - 1.0)


_hue_cache: dict = {}


def hue_matrix(degrees):
    """Matriz 3x3 (orden BGR) que rota el matiz conservando la luminosidad."""
    key = int(round(degrees)) % 360
    m = _hue_cache.get(key)
    if m is None:
        a = np.deg2rad(key)
        c, s = np.cos(a), np.sin(a)
        rgb = np.array([[0.213 + 0.787 * c - 0.213 * s, 0.715 - 0.715 * c - 0.715 * s, 0.072 - 0.072 * c + 0.928 * s],
                        [0.213 - 0.213 * c + 0.143 * s, 0.715 + 0.285 * c + 0.140 * s, 0.072 - 0.072 * c - 0.283 * s],
                        [0.213 - 0.213 * c - 0.787 * s, 0.715 - 0.715 * c + 0.715 * s, 0.072 + 0.928 * c + 0.072 * s]],
                       np.float32)
        m = np.ascontiguousarray(rgb[::-1, ::-1])
        _hue_cache[key] = m
    return m


def shift_hue(img, degrees):
    """Rota el matiz de una imagen BGR (rápido: una sola transformación de color)."""
    if abs(degrees) < 0.01:
        return img
    return cv2.transform(img, hue_matrix(degrees))


def scale_saturation(img, s):
    if abs(s - 1.0) < 0.01:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * s, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def hue_color(deg, sat=1.0, val=1.0):
    """Color BGR (enteros) a partir de un matiz en grados."""
    hsv = np.uint8([[[int(deg / 2) % 180, int(np.clip(sat, 0, 1) * 255), int(np.clip(val, 0, 1) * 255)]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


_grid_cache: dict = {}


def grid(w: int, h: int):
    """Mallas X, Y (float32) cacheadas por tamaño."""
    key = (w, h)
    g = _grid_cache.get(key)
    if g is None:
        if len(_grid_cache) > 8:
            _grid_cache.clear()
        x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        g = (x, y)
        _grid_cache[key] = g
    return g


def slice_glitch(frame, amount, rgb_shift, n_slices, rng):
    """Glitch clásico: canales RGB desplazados y franjas horizontales movidas."""
    h, w = frame.shape[:2]
    out = frame.copy()
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0:
        return out
    s = int(round(rgb_shift * amount))
    if s > 0:
        out[..., 0] = np.roll(out[..., 0], s, axis=1)
        out[..., 2] = np.roll(out[..., 2], -s, axis=1)
    for _ in range(int(round(n_slices * amount))):
        y = int(rng.integers(0, h))
        hh = int(rng.integers(2, max(3, h // 10)))
        dx = int(rng.uniform(-1, 1) * amount * w * 0.15)
        out[y:y + hh] = np.roll(out[y:y + hh], dx, axis=1)
    return out


def prepare_frame(frame, max_width):
    """Reduce el fotograma a un ancho máximo y fuerza dimensiones pares."""
    h, w = frame.shape[:2]
    if max_width and w > max_width:
        s = max_width / w
        w2, h2 = int(round(w * s)), int(round(h * s))
    else:
        w2, h2 = w, h
    w2 -= w2 % 2
    h2 -= h2 % 2
    if (w2, h2) != (w, h):
        frame = cv2.resize(frame, (w2, h2), interpolation=cv2.INTER_AREA)
    return frame


def cover_fit(img, w, h):
    """Recorta al centro y escala para llenar exactamente w x h (sin deformar)."""
    w, h = max(1, int(w)), max(1, int(h))
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    nw, nh = max(w, int(round(iw * s))), max(h, int(round(ih * s)))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    return r[y0:y0 + h, x0:x0 + w]


def paste(dst, src, x, y, alpha=1.0, mask=None):
    """Pega src en dst en (x, y) recortando a los bordes. mask: float 0..1 (alto x ancho)."""
    H, W = dst.shape[:2]
    h, w = src.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    d = dst[y0:y1, x0:x1]
    if mask is None and alpha >= 0.999:
        d[:] = s
        return
    a = alpha if mask is None else mask[y0 - y:y1 - y, x0 - x:x1 - x, None] * alpha
    d[:] = (s.astype(np.float32) * a + d.astype(np.float32) * (1.0 - a)).astype(np.uint8)


def fast_blur(img, sigma):
    """Desenfoque gaussiano barato para radios grandes: se calcula a menor resolución y se vuelve a escalar."""
    if sigma < 2.5:
        return cv2.GaussianBlur(img, (0, 0), max(0.1, sigma)) if sigma > 0.05 else img
    k = 2 if sigma < 6 else 4 if sigma < 14 else 8
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(2, w // k), max(2, h // k)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / k)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


_lo_cache: dict = {}


def grid_lo(w, h, f=2):
    """Malla X, Y de resolución reducida, expresada en coordenadas de píxel del tamaño completo."""
    key = (w, h, f)
    g = _lo_cache.get(key)
    if g is None:
        if len(_lo_cache) > 8:
            _lo_cache.clear()
        lw, lh = max(2, w // f), max(2, h // f)
        x = (np.arange(lw, dtype=np.float32) + 0.5) * (w / lw) - 0.5
        y = (np.arange(lh, dtype=np.float32) + 0.5) * (h / lh) - 0.5
        g = np.meshgrid(x, y)
        _lo_cache[key] = g
    return g


def remap_lo(frame, mx, my, border=cv2.BORDER_REFLECT_101):
    """remap con mapas calculados a baja resolución: se amplían (son suaves) y se aplican a la imagen completa."""
    h, w = frame.shape[:2]
    mx = cv2.resize(mx.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    my = cv2.resize(my.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=border)
