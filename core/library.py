"""Biblioteca de imágenes: carpetas con el nombre de la clase (person, dog, car… o persona, perro, auto…)."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np

from core.utils import DATA_DIR

EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_DIR = DATA_DIR / "biblioteca"
GENERAL = ("cualquiera", "any", "todo", "general")

LEEME = """BIBLIOTECA DE IMÁGENES
======================
Cada subcarpeta lleva el nombre de lo que reconoce el programa y dentro van tus imágenes:

    biblioteca/person/      (o "persona")
    biblioteca/dog/         (o "perro")
    biblioteca/car/         (o "auto")
    biblioteca/cup/  biblioteca/chair/  biblioteca/cell phone/ ...

Nombres válidos: los de YOLO en inglés (person, dog, cat, car, bicycle, bottle, cup, chair, laptop,
cell phone, book, clock, tv, ...) o su equivalente en español.

La carpeta "cualquiera" se usa para los objetos que no tengan carpeta propia.
Si una clase no tiene imágenes, el efecto usa imágenes abstractas de demostración.

Para descargar una colección de ejemplo desde Wikimedia Commons (necesita internet):
    python descargar_biblioteca.py            (clases comunes)
    python descargar_biblioteca.py dog cat --n 20
"""


def norm(s):
    s = unicodedata.normalize("NFD", s.lower()).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def proc_image(seed, size=160):
    """Imagen abstracta determinista (para que el efecto funcione aunque la biblioteca esté vacía)."""
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    xs = np.linspace(0, 1, size, dtype=np.float32)
    X, Y = np.meshgrid(xs, xs)
    a = rng.uniform(0, 2 * np.pi)
    g = X * np.cos(a) + Y * np.sin(a)
    hue = rng.integers(0, 180)
    hsv = np.dstack([np.mod(hue + 70 * g, 180), np.full_like(g, 190) + 50 * g, 70 + 170 * g]).astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    for _ in range(int(rng.integers(3, 8))):
        col = tuple(int(c) for c in rng.integers(30, 255, 3))
        c = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        if rng.random() < 0.5:
            cv2.circle(img, c, int(rng.integers(8, size // 3)), col, -1 if rng.random() < 0.6 else 3, cv2.LINE_AA)
        else:
            c2 = (c[0] + int(rng.integers(10, size // 2)), c[1] + int(rng.integers(10, size // 2)))
            cv2.rectangle(img, c, c2, col, -1 if rng.random() < 0.6 else 3)
    return img


class ImageLibrary:
    def __init__(self, folder=None):
        self.root = Path(folder).expanduser() if folder else DEFAULT_DIR
        self.index = {}
        self.cache = {}
        self.total = 0
        self.scan()

    def scan(self):
        self.index = {}
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            for sub in sorted(self.root.iterdir()):
                if sub.is_dir():
                    files = sorted(p for p in sub.rglob("*") if p.suffix.lower() in EXT)
                    if files:
                        self.index[norm(sub.name)] = files
        except OSError:
            pass
        self.total = sum(len(v) for v in self.index.values())

    def files_for(self, *names):
        for n in names:
            if norm(n) in self.index:
                return self.index[norm(n)]
        for g in GENERAL:
            if g in self.index:
                return self.index[g]
        return []

    def load(self, path, maxside=320):
        img = self.cache.get(path)
        if img is None:
            data = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                img = proc_image(hash(str(path)))
            h, w = img.shape[:2]
            s = maxside / max(h, w)
            if s < 1:
                img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            if len(self.cache) > 300:
                self.cache.pop(next(iter(self.cache)))
            self.cache[path] = img
        return img

    def image(self, names, idx, seed=0):
        """Devuelve (imagen, es_real). Si no hay imágenes para esa clase, genera una abstracta."""
        files = self.files_for(*names)
        if files:
            return self.load(files[idx % len(files)]), True
        return proc_image(seed * 7919 + idx), False

    def sample_all(self, n, rng):
        allf = [f for v in self.index.values() for f in v]
        if not allf:
            return []
        pick = rng.choice(len(allf), size=min(n, len(allf)), replace=False)
        return [self.load(allf[int(i)], 160) for i in pick]


def open_folder(path=None):
    """Crea (si falta) y abre la carpeta de la biblioteca en el explorador de archivos."""
    p = Path(path) if path else DEFAULT_DIR
    p.mkdir(parents=True, exist_ok=True)
    leeme = p / "LEEME.txt"
    if not leeme.exists():
        leeme.write_text(LEEME, encoding="utf-8")
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(p))                        # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p)])
    except Exception:
        pass
    return p
