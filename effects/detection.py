"""Detección de objetos con YOLO (Ultralytics): recuadros con el nombre del objeto."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.params import Effect, F, I, B, C, T
from core.pipeline import register

COCO_ES = [
    "persona", "bicicleta", "auto", "moto", "avión", "bus", "tren", "camión", "barco", "semáforo",
    "hidrante", "señal de pare", "parquímetro", "banca", "pájaro", "gato", "perro", "caballo", "oveja", "vaca",
    "elefante", "oso", "cebra", "jirafa", "mochila", "paraguas", "bolso", "corbata", "maleta", "frisbee",
    "esquís", "snowboard", "balón", "cometa", "bate de béisbol", "guante de béisbol", "patineta", "tabla de surf",
    "raqueta de tenis", "botella", "copa de vino", "taza", "tenedor", "cuchillo", "cuchara", "tazón", "banano",
    "manzana", "sándwich", "naranja", "brócoli", "zanahoria", "perro caliente", "pizza", "dona", "pastel", "silla",
    "sofá", "planta", "cama", "mesa", "inodoro", "televisor", "portátil", "ratón", "control remoto", "teclado",
    "celular", "microondas", "horno", "tostadora", "lavamanos", "nevera", "libro", "reloj", "florero", "tijeras",
    "oso de peluche", "secador de pelo", "cepillo de dientes"]

_FONT_CACHE: dict = {}


def _font(size):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(name, size)
            break
        except Exception:
            continue
    if font is None:
        try:
            font = ImageFont.load_default(size)
        except TypeError:
            font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def _color(cls_id):
    hsv = np.uint8([[[(cls_id * 37) % 180, 220, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


@register
class ObjectDetection(Effect):
    name = "Detección de objetos (YOLO)"
    category = "Detección"
    PARAMS = [F("confianza", "Confianza mínima", 0.35, 0.05, 0.95), I("cada", "Detectar cada N fotogramas", 2, 1, 15),
              I("grosor", "Grosor del recuadro", 2, 1, 6), B("etiquetas", "Mostrar etiquetas", True),
              B("mostrar_conf", "Mostrar % de confianza", True), C("idioma", "Idioma de las etiquetas", ["español", "inglés"]),
              T("clases", "Clases a mostrar (vacío = todas; ej: persona, auto)", ""),
              C("modelo", "Modelo", ["yolov8n.pt", "yolov8s.pt", "yolo11n.pt"])]

    def reset(self):
        self.model = getattr(self, "model", None)         # el modelo cargado se conserva
        self.model_name = getattr(self, "model_name", "")
        self.failed = False
        self.last = []
        self.names = getattr(self, "names", {})

    def _load(self, name):
        try:
            from ultralytics import YOLO
            from core.utils import MODELS_DIR
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            self.model = YOLO(str(MODELS_DIR / name))   # descarga los pesos la primera vez
            self.model_name = name
            self.names = dict(self.model.names)
            self.error = ""
        except Exception as e:
            self.failed = True
            self.error = f"No se pudo cargar YOLO ({e}). Instala con: pip install ultralytics"

    def _allowed(self, text):
        wanted = {t.strip().lower() for t in text.split(",") if t.strip()}
        if not wanted:
            return None
        ids = []
        for i, en in self.names.items():
            es = COCO_ES[i] if i < len(COCO_ES) else en
            if en.lower() in wanted or es.lower() in wanted:
                ids.append(i)
        return ids or None

    def process(self, frame, ctx):
        name = self.value("modelo", ctx)
        if self.failed and name == self.model_name:
            return frame
        if self.model is None or self.model_name != name:
            self.failed = False
            self._load(name)
            if self.failed:
                self.model_name = name
                return frame
        if ctx.frame_index % self.value("cada", ctx) == 0 or not self.last:
            res = self.model.predict(frame, conf=self.value("confianza", ctx), classes=self._allowed(
                self.value("clases", ctx)), verbose=False)[0]
            self.last = []
            for b in res.boxes:
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                self.last.append((x1, y1, x2, y2, int(b.cls[0]), float(b.conf[0])))
        out = frame.copy()
        th = self.value("grosor", ctx)
        labels = []
        for x1, y1, x2, y2, c, conf in self.last:
            cv2.rectangle(out, (x1, y1), (x2, y2), _color(c), th)
            if self.value("etiquetas", ctx):
                nm = self.names.get(c, str(c))
                if self.value("idioma", ctx) == "español" and c < len(COCO_ES):
                    nm = COCO_ES[c]
                txt = f"{nm} {int(conf * 100)}%" if self.value("mostrar_conf", ctx) else nm
                labels.append((x1, y1, txt, _color(c)))
        if labels:
            size = max(12, frame.shape[0] // 34)
            font = _font(size)
            img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            dr = ImageDraw.Draw(img)
            for x, y, txt, col in labels:
                l, t, r, b = dr.textbbox((0, 0), txt, font=font)
                ty = max(0, y - (b - t) - 6)
                dr.rectangle([x, ty, x + (r - l) + 8, ty + (b - t) + 6], fill=(col[2], col[1], col[0]))
                dr.text((x + 4, ty + 2 - t), txt, font=font, fill=(0, 0, 0))
            out = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        return out
