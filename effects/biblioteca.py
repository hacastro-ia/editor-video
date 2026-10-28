"""Biblioteca de imágenes: lo que YOLO reconoce se muestra con imágenes de una base de datos.

La «base de datos» es una carpeta (biblioteca/) con subcarpetas por clase, o los «recuerdos» del propio video
(recortes de objetos vistos antes). Si falta una clase, se usan imágenes abstractas de demostración.
"""
from __future__ import annotations

import cv2
import numpy as np

from core import yolo
from core.library import ImageLibrary, proc_image
from core.params import Effect, F, I, B, C, T
from core.pipeline import register
from core.textdraw import draw_texts
from core.utils import cover_fit, hue_color, paste
from effects.detection import COCO_ES

MODOS = {
    "reemplazar": "Cada objeto reconocido se cubre con una imagen de la biblioteca.",
    "fichas flotantes": "Una ficha con imagen y etiqueta flota junto a cada objeto, unida por una línea.",
    "fotomosaico": "Cada objeto se reconstruye con cientos de miniaturas de la biblioteca.",
    "órbita": "Pequeñas imágenes circulares giran alrededor de cada objeto.",
    "tira de memoria": "Una tira inferior muestra imágenes de todo lo que el programa está reconociendo.",
}


def _color(cls_id):
    return hue_color(cls_id * 37, 0.85, 1.0)


@register
class ImageLibraryEffect(Effect):
    name = "Biblioteca de imágenes (objetos)"
    category = "Biblioteca"
    HELP = {"modo": MODOS, "fuente": {
        "biblioteca": "Imágenes de tu carpeta «biblioteca» (botón «Biblioteca…» de la barra superior).",
        "recuerdos del video": "Recortes de objetos que ya aparecieron en este mismo video.",
        "mezcla": "Alterna entre la biblioteca y los recuerdos del video.",
    }}
    PARAMS = [
        C("modo", "Modo", list(MODOS.keys())),
        C("fuente", "De dónde salen las imágenes", ["biblioteca", "recuerdos del video", "mezcla"]),
        F("opacidad", "Opacidad de las imágenes", 0.95, 0.1, 1.0),
        F("tam", "Tamaño de fichas / órbita", 0.22, 0.08, 0.5),
        I("tesela", "Tamaño de las teselas del fotomosaico (px)", 14, 6, 40),
        I("cambio", "Cambiar de imagen cada N fotogramas (0 = nunca)", 45, 0, 240),
        B("cambio_beat", "Cambiar de imagen con el beat", True),
        B("etiquetas", "Mostrar etiquetas", True),
        F("confianza", "Confianza mínima", 0.35, 0.05, 0.95),
        I("cada", "Detectar cada N fotogramas", 2, 1, 10),
        T("clases", "Clases a mostrar (vacío = todas; ej: persona, taza)", ""),
        T("carpeta", "Carpeta de imágenes (vacío = carpeta «biblioteca»)", ""),
        C("modelo", "Modelo", ["yolov8n.pt", "yolov8s.pt", "yolo11n.pt"]),
    ]

    # -- estado ------------------------------------------------------------
    def reset(self):
        self.runner = getattr(self, "runner", None)          # el modelo cargado se conserva
        self.failed = False
        self.dets = []
        self.state = {}
        self.memory = {}
        self.lib = getattr(self, "lib", None)
        self.lib_dir = getattr(self, "lib_dir", None)
        self.tiles = None
        self.rng = np.random.default_rng(21)
        self._last_beat = 0.0
        self.beat_hit = False
        self._reset_trk = True
        self._circle = {}

    def _load(self, name):
        try:
            self.runner = yolo.Runner(name)
            self.error = ""
        except Exception as e:
            self.failed = True
            self.error = f"No se pudo cargar YOLO ({e}). Instala con: pip install ultralytics"

    def _library(self, ctx):
        folder = self.value("carpeta", ctx).strip()
        if self.lib is None or self.lib_dir != folder:
            self.lib = ImageLibrary(folder or None)
            self.lib_dir = folder
            self.tiles = None
        return self.lib

    def _allowed(self, text, names):
        wanted = {t.strip().lower() for t in text.split(",") if t.strip()}
        if not wanted:
            return None
        ids = [i for i, en in names.items()
               if en.lower() in wanted or (i < len(COCO_ES) and COCO_ES[i].lower() in wanted)]
        return ids or None

    # -- detección ---------------------------------------------------------
    def _detect(self, frame, ctx):
        names = self.runner.names
        res = self.runner(frame, conf=self.value("confianza", ctx), classes=self._allowed(self.value("clases", ctx), names))
        out = []
        if res is not None and res.boxes is not None and len(res.boxes) > 0:
            xyxy = yolo.to_np(res.boxes.xyxy)
            cls = yolo.to_np(res.boxes.cls).astype(int)
            conf = yolo.to_np(res.boxes.conf)
            ids = yolo.to_np(res.boxes.id).astype(int) if res.boxes.id is not None else None
            h, w = frame.shape[:2]
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue
                c = int(cls[i])
                en = names.get(c, str(c))
                es = COCO_ES[c] if c < len(COCO_ES) else en
                out.append({"id": int(ids[i]) if ids is not None else 1000 + i, "cls": c, "en": en, "es": es,
                            "conf": float(conf[i]), "box": (x1, y1, x2, y2)})
        self.dets = out

    def _remember(self, frame, ctx):
        """Guarda recortes de lo que se ve para usarlos como «recuerdos»."""
        for d in self.dets:
            st = self.state.setdefault(d["id"], {"idx": int(self.rng.integers(0, 1000)), "last": ctx.frame_index})
            st["seen"] = ctx.frame_index
            if d["conf"] >= 0.45 and ctx.frame_index - st.get("mem", -999) >= 20:
                x1, y1, x2, y2 = d["box"]
                crop = cover_fit(frame[y1:y2, x1:x2], 128, 128).copy()
                lst = self.memory.setdefault(d["cls"], [])
                if len(lst) < 10:
                    lst.append(crop)
                else:
                    lst[(ctx.frame_index // 20) % 10] = crop
                st["mem"] = ctx.frame_index
        for k in [k for k, v in self.state.items() if ctx.frame_index - v.get("seen", 0) > 150]:
            del self.state[k]

    # -- imágenes ----------------------------------------------------------
    def _image(self, det, k, ctx, w, h):
        st = self.state.setdefault(det["id"], {"idx": 0, "last": ctx.frame_index})
        every = self.value("cambio", ctx)
        if (every > 0 and ctx.frame_index - st["last"] >= every) or \
                (self.value("cambio_beat", ctx) and self.beat_hit):
            st["idx"] += 1
            st["last"] = ctx.frame_index
        idx = st["idx"] + k * 7
        fuente = self.value("fuente", ctx)
        img = None
        use_mem = fuente == "recuerdos del video" or (fuente == "mezcla" and idx % 2 == 1)
        if use_mem:
            mem = self.memory.get(det["cls"], [])
            if len(mem) > 1:
                img = mem[(idx + 1) % len(mem)]
            elif mem and fuente == "recuerdos del video":
                img = mem[0]
        if img is None:
            img, _ = self._library(ctx).image((det["en"], det["es"]), idx, seed=det["cls"] + 1)
        return cover_fit(img, w, h)

    def _tile_set(self, ctx, tile):
        fuente = self.value("fuente", ctx)
        key = (tile, fuente, self._library(ctx).total)
        if fuente != "biblioteca":
            mem = [m for lst in self.memory.values() for m in lst]
            if len(mem) >= 6:
                key = None
                imgs = mem[:80]
            else:
                imgs = None
        else:
            imgs = None
        if imgs is None:
            if self.tiles is not None and self.tiles[0] == key:
                return self.tiles[1], self.tiles[2]
            imgs = self._library(ctx).sample_all(150, self.rng)
            if len(imgs) < 12:
                imgs = imgs + [proc_image(1000 + i, 96) for i in range(96)]
        th = np.stack([cv2.resize(i, (tile, tile), interpolation=cv2.INTER_AREA) for i in imgs])
        means = th.reshape(len(th), -1, 3).mean(axis=1).astype(np.float32)
        if key is not None:
            self.tiles = (key, th, means)
        return th, means

    def _mosaic(self, region, ctx):
        tile = self.value("tesela", ctx)
        th, means = self._tile_set(ctx, tile)
        rh, rw = region.shape[:2]
        cw, ch = max(2, rw // tile), max(2, rh // tile)
        small = cv2.resize(region, (cw, ch), interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float32)
        d = ((small[:, None, :] - means[None, :, :]) ** 2).sum(-1)
        d += self.rng.random(d.shape, dtype=np.float32) * 600.0        # variedad entre teselas parecidas
        pick = np.argmin(d, axis=1).reshape(ch, cw)
        canvas = th[pick]                                              # (ch, cw, t, t, 3)
        canvas = canvas.transpose(0, 2, 1, 3, 4).reshape(ch * tile, cw * tile, 3)
        canvas = cv2.resize(canvas, (rw, rh), interpolation=cv2.INTER_AREA)
        tint = cv2.resize(cv2.resize(region, (cw, ch), interpolation=cv2.INTER_AREA), (rw, rh),
                          interpolation=cv2.INTER_LINEAR)
        return cv2.addWeighted(canvas, 0.70, tint, 0.30, 0)

    def _circle_mask(self, s):
        m = self._circle.get(s)
        if m is None:
            m = np.zeros((s, s), np.float32)
            cv2.circle(m, (s // 2, s // 2), max(1, s // 2 - 1), 1.0, -1, cv2.LINE_AA)
            self._circle[s] = m
        return m

    # -- modos -----------------------------------------------------------------
    def _m_reemplazar(self, out, ctx, labels):
        op = self.value("opacidad", ctx)
        for d in self.dets:
            x1, y1, x2, y2 = d["box"]
            paste(out, self._image(d, 0, ctx, x2 - x1, y2 - y1), x1, y1, alpha=op)
            cv2.rectangle(out, (x1, y1), (x2, y2), _color(d["cls"]), 2)
            labels.append((x1, max(0, y1 - 24), d, _color(d["cls"])))

    def _m_fichas(self, out, ctx, labels):
        h, w = out.shape[:2]
        cs = int(self.value("tam", ctx) * w)
        ch = int(cs * 0.75)
        used = []
        op = self.value("opacidad", ctx)
        for n, d in enumerate(sorted(self.dets, key=lambda d: d["box"][1])):
            x1, y1, x2, y2 = d["box"]
            col = _color(d["cls"])
            right = (x1 + x2) / 2 < w / 2
            cx = x2 + 28 if right else x1 - 28 - cs
            if cx < 4 or cx + cs > w - 4:
                cx = x1 + 6 if x1 + 6 + cs < w else max(4, w - cs - 4)
            cy0 = int(y1 + 6 * np.sin(ctx.time * 2 + n))
            lo, hi = 4, max(4, h - ch - 28)
            step = ch + 30
            cy = int(np.clip(cy0, lo, hi))
            for off in (0, -step, step, -2 * step, 2 * step, -3 * step, 3 * step):
                cand = cy0 + off
                if cand < lo or cand > hi:
                    continue
                if not any(abs(cand - uy) < step and abs(cx - ux) < cs + 10 for ux, uy in used):
                    cy = int(cand)
                    break
            used.append((cx, cy))
            ax, ay = (x2, (y1 + y2) // 2) if right else (x1, (y1 + y2) // 2)
            bx, by = (cx, cy + ch // 2) if right else (cx + cs, cy + ch // 2)
            cv2.line(out, (ax, ay), (bx, by), col, 2, cv2.LINE_AA)
            cv2.circle(out, (ax, ay), 5, col, -1, cv2.LINE_AA)
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
            paste(out, np.zeros((ch + 4, cs + 4, 3), np.uint8), cx + 6, cy + 6, alpha=0.35)
            paste(out, self._image(d, 0, ctx, cs, ch), cx, cy, alpha=op)
            cv2.rectangle(out, (cx, cy), (cx + cs, cy + ch), (255, 255, 255), 3)
            labels.append((cx, cy + ch + 2, d, col))

    def _m_mosaico(self, out, ctx, labels):
        op = self.value("opacidad", ctx)
        for d in self.dets:
            x1, y1, x2, y2 = d["box"]
            mos = self._mosaic(out[y1:y2, x1:x2].copy(), ctx)
            paste(out, mos, x1, y1, alpha=op)
            cv2.rectangle(out, (x1, y1), (x2, y2), _color(d["cls"]), 1)
            labels.append((x1, max(0, y1 - 24), d, _color(d["cls"])))

    def _m_orbita(self, out, ctx, labels):
        vol = float(ctx.audio.get("volumen", 0.0))
        for d in self.dets:
            x1, y1, x2, y2 = d["box"]
            bw, bh = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            base = int(np.clip(min(bw, bh) * 0.35, 30, 120) * (1 + 0.5 * self.value("tam", ctx)))
            n = 6
            rx, ry = bw * 0.62 + base * 0.45, bh * 0.62 + base * 0.45
            col = _color(d["cls"])
            sprites = []
            for k in range(n):
                a = ctx.time * (0.9 + 1.6 * vol) + k * 2 * np.pi / n + d["id"]
                depth = 0.5 + 0.5 * np.sin(a)
                sprites.append((depth, k, a))
            for depth, k, a in sorted(sprites):
                s = max(16, int(base * (0.65 + 0.4 * depth)))
                px, py = int(cx + rx * np.cos(a) - s / 2), int(cy + ry * np.sin(a) * 0.8 - s / 2)
                img = self._image(d, k + 1, ctx, s, s)
                paste(out, img, px, py, alpha=self.value("opacidad", ctx), mask=self._circle_mask(s))
                cv2.circle(out, (px + s // 2, py + s // 2), s // 2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 1)
            labels.append((x1, max(0, y1 - 24), d, col))

    def _m_tira(self, out, ctx, labels):
        h, w = out.shape[:2]
        groups = {}
        for d in self.dets:
            g = groups.setdefault(d["cls"], {"d": d, "n": 0})
            g["n"] += 1
        if not groups:
            return
        sh = int(h * 0.2)
        paste(out, np.zeros((sh + 14, w, 3), np.uint8), 0, h - sh - 14, alpha=0.55)
        cw = int(sh * 1.15)
        x = 10
        for k, (cls, g) in enumerate(list(groups.items())[:8]):
            img = self._image(g["d"], 0, ctx, cw, sh - 22)
            y = h - sh - 8
            paste(out, img, x, y, alpha=self.value("opacidad", ctx))
            cv2.rectangle(out, (x, y), (x + cw, y + sh - 22), _color(cls), 2)
            labels.append((x, y + sh - 22, dict(g["d"], n=g["n"]), _color(cls)))
            x += cw + 10
            if x + cw > w:
                break

    # -- proceso ---------------------------------------------------------------
    def process(self, frame, ctx):
        name = self.value("modelo", ctx)
        if self.failed and self.runner is None:
            return frame
        if self.runner is None or self.runner.name != name:
            self.failed = False
            self._load(name)
            if self.failed:
                return frame
        if self._reset_trk:
            self.runner.reset_tracking()
            self._reset_trk = False
        b = float(ctx.audio.get("beat", 0.0))
        self.beat_hit = b > 0.95 and self._last_beat <= 0.95
        self._last_beat = b
        if ctx.frame_index % self.value("cada", ctx) == 0 or not self.dets:
            self._detect(frame, ctx)
        self._remember(frame, ctx)
        lib = self._library(ctx)
        if lib.total == 0 and self.value("fuente", ctx) != "recuerdos del video":
            self.error = ("La carpeta «biblioteca» está vacía: se usan imágenes de demostración. "
                          "Pestaña Salida → Herramientas → «Biblioteca de imágenes…» para añadir las tuyas.")
        elif self.error.startswith("La carpeta"):
            self.error = ""
        out = frame.copy()
        labels = []
        {"reemplazar": self._m_reemplazar, "fichas flotantes": self._m_fichas, "fotomosaico": self._m_mosaico,
         "órbita": self._m_orbita, "tira de memoria": self._m_tira}[self.value("modo", ctx)](out, ctx, labels)
        if self.value("etiquetas", ctx) and labels:
            items = []
            for x, y, d, col in labels:
                txt = d["es"] if d["es"] else d["en"]
                txt = f"{txt} ×{d['n']}" if "n" in d else f"{txt} {int(d['conf'] * 100)}%"
                items.append((x, y, txt, (0, 0, 0), col))
            out = draw_texts(out, items, max(12, frame.shape[0] // 34))
        return out
