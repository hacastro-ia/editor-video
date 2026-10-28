"""Texto con acentos y ñ sobre imágenes OpenCV (usa Pillow, porque cv2.putText no lo soporta)."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_CACHE: dict = {}


def font(size):
    if size in _CACHE:
        return _CACHE[size]
    f = None
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            f = ImageFont.truetype(name, size)
            break
        except Exception:
            continue
    if f is None:
        try:
            f = ImageFont.load_default(size)
        except TypeError:
            f = ImageFont.load_default()
    _CACHE[size] = f
    return f


def draw_texts(frame, items, size=16):
    """items: [(x, y, texto, color_texto_bgr, color_fondo_bgr_o_None)]. Devuelve una imagen nueva."""
    if not items:
        return frame
    fnt = font(size)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(img)
    for x, y, txt, fg, bg in items:
        l, t, r, b = dr.textbbox((0, 0), txt, font=fnt)
        x, y = int(x), int(y)
        if bg is not None:
            dr.rectangle([x, y, x + (r - l) + 8, y + (b - t) + 6], fill=(bg[2], bg[1], bg[0]))
        dr.text((x + 4, y + 2 - t), txt, font=fnt, fill=(fg[2], fg[1], fg[0]))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
