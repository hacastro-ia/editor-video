"""Detección, seguimiento y selección de caras (sin dependencias pesadas).

* Detector: YuNet (OpenCV). Si no se puede descargar, usa el clasificador Haar incluido en OpenCV.
* Seguimiento: emparejamiento por IoU con memoria (la censura se mantiene si se pierde la cara un momento).
* Reconocimiento opcional por foto de referencia: SFace (OpenCV).
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from core.utils import ensure_model

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
YUNET_URL = ["https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/" + YUNET_FILE,
             "https://huggingface.co/opencv/face_detection_yunet/resolve/main/" + YUNET_FILE]
SFACE_FILE = "face_recognition_sface_2021dec.onnx"
SFACE_URL = ["https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/" + SFACE_FILE,
             "https://huggingface.co/opencv/face_recognition_sface/resolve/main/" + SFACE_FILE]
MATCH_THRESHOLD = 0.363     # umbral coseno recomendado para SFace


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class FaceDetector:
    def __init__(self):
        self.yunet = None
        self.haar = None
        self.backend = ""
        self.error = ""
        self._tried = False

    def _ensure(self):
        if self._tried:
            return
        self._tried = True
        try:
            path = ensure_model(YUNET_FILE, YUNET_URL)
            self.yunet = cv2.FaceDetectorYN.create(str(path), "", (320, 320), 0.6, 0.3, 5000)
            self.backend = "YuNet"
        except Exception as e:
            self.error = f"YuNet no disponible ({e}); usando Haar."
            self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self.backend = "Haar"

    def detect(self, frame, score_thr=0.6):
        """Devuelve [(x, y, w, h, score, fila_yunet_o_None)]."""
        self._ensure()
        h, w = frame.shape[:2]
        out = []
        if self.yunet is not None:
            self.yunet.setInputSize((w, h))
            self.yunet.setScoreThreshold(float(score_thr))
            _, faces = self.yunet.detect(frame)
            if faces is not None:
                for row in faces:
                    x, y, fw, fh = [float(v) for v in row[:4]]
                    out.append((x, y, fw, fh, float(row[14]), row.copy()))
        else:
            sc = min(1.0, 480.0 / w)
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if sc < 1.0:
                g = cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            g = cv2.equalizeHist(g)
            for (x, y, fw, fh) in self.haar.detectMultiScale(g, 1.15, 7, minSize=(36, 36)):
                out.append((x / sc, y / sc, fw / sc, fh / sc, 1.0, None))
        return out


DETECTOR = FaceDetector()


class Track:
    def __init__(self, tid, box, score, frame_idx):
        self.id = tid
        self.box = np.array(box, np.float32)      # x, y, w, h
        self.vel = np.zeros(4, np.float32)
        self.score = score
        self.missed = 0
        self.identity = None
        self.row = None
        self.last = frame_idx
        self.thumb = None
        self.thumb_frame = -999
        self.id_try = -999

    @property
    def key(self):
        return self.identity or f"Cara {self.id}"


class FaceState:
    """Estado compartido entre el efecto de censura y la interfaz."""

    def __init__(self):
        self.lock = threading.RLock()
        self.tracks = {}
        self.next_id = 1
        self.selected = set()
        self.mode = "todas"
        self.references = []          # {"name", "feature", "thumb"}
        self.known = {}               # clave -> {"thumb", "seen"}
        self.recognizer = None
        self.recognizer_error = ""

    # -- ciclo de vida -----------------------------------------------------
    def clear_tracks(self):
        """Olvida los seguimientos (al abrir o saltar en el video). Los IDs vuelven a empezar
        para que una selección siga valiendo al reiniciar la reproducción o exportar."""
        with self.lock:
            self.tracks = {}
            self.next_id = 1

    # -- decisión de censura -------------------------------------------------
    def should_censor(self, track):
        sel = track.key in self.selected
        if self.mode == "todas":
            return True
        if self.mode == "todas menos las elegidas":
            return not sel
        return sel

    # -- actualización -------------------------------------------------------
    def update(self, dets, frame, frame_idx, max_missed=20):
        with self.lock:
            tracks = list(self.tracks.values())
            cand = []
            for ti, t in enumerate(tracks):
                for di, d in enumerate(dets):
                    v = _iou(t.box, d[:4])
                    if v > 0.1:
                        cand.append((v, ti, di))
            cand.sort(reverse=True)
            used_t, used_d = set(), set()
            for _, ti, di in cand:
                if ti in used_t or di in used_d:
                    continue
                used_t.add(ti)
                used_d.add(di)
                t, d = tracks[ti], dets[di]
                new = np.array(d[:4], np.float32)
                t.vel = 0.5 * t.vel + 0.5 * (new - t.box)
                t.box = 0.3 * t.box + 0.7 * new
                t.score, t.missed, t.last, t.row = d[4], 0, frame_idx, d[5]
            for ti, t in enumerate(tracks):
                if ti not in used_t:
                    t.missed += 1
                    t.box = t.box + t.vel
                    t.vel *= 0.8
                    t.row = None
                    if t.missed > max_missed:
                        self.tracks.pop(t.id, None)
            for di, d in enumerate(dets):
                if di not in used_d:
                    t = Track(self.next_id, d[:4], d[4], frame_idx)
                    t.row = d[5]
                    self.tracks[t.id] = t
                    self.next_id += 1
            # una sola identificación por fotograma para no frenar la reproducción
            done = False
            for t in list(self.tracks.values()):
                if t.missed == 0:
                    if not done and self._identify(frame, t, frame_idx):
                        done = True
                    self._thumb(frame, t, frame_idx)

    def _identify(self, frame, t, frame_idx):
        if not self.references or t.row is None or self.recognizer is None:
            return False
        if frame_idx - t.id_try < (45 if t.identity else 8):
            return False
        t.id_try = frame_idx
        try:
            feat = self.recognizer.feature(self.recognizer.alignCrop(frame, t.row))
            best, best_s = None, -1.0
            for ref in self.references:
                s = self.recognizer.match(feat, ref["feature"], cv2.FaceRecognizerSF_FR_COSINE)
                if s > best_s:
                    best, best_s = ref, s
            t.identity = best["name"] if best is not None and best_s >= MATCH_THRESHOLD else None
        except Exception as e:
            self.recognizer_error = str(e)
        return True

    def _thumb(self, frame, t, frame_idx):
        if t.thumb is not None and frame_idx - t.thumb_frame < 60:
            self.known.setdefault(t.key, {"thumb": t.thumb, "seen": time.time()})["seen"] = time.time()
            return
        fh, fw = frame.shape[:2]
        x, y, w, h = t.box
        m = 0.25
        x0, y0 = int(max(0, x - w * m)), int(max(0, y - h * m))
        x1, y1 = int(min(fw, x + w * (1 + m))), int(min(fh, y + h * (1 + m)))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return
        t.thumb = cv2.resize(frame[y0:y1, x0:x1], (56, 56), interpolation=cv2.INTER_AREA)
        t.thumb_frame = frame_idx
        self.known[t.key] = {"thumb": t.thumb, "seen": time.time()}
        if len(self.known) > 12:                       # conservar los 12 más recientes
            for k, _ in sorted(self.known.items(), key=lambda kv: kv[1]["seen"])[:len(self.known) - 12]:
                if k not in self.selected and not any(r["name"] == k for r in self.references):
                    self.known.pop(k, None)

    # -- interacción ---------------------------------------------------------
    def active(self):
        with self.lock:
            return [(t.key, t.box.copy(), self.should_censor(t)) for t in self.tracks.values()]

    def toggle(self, key):
        with self.lock:
            if key in self.selected:
                self.selected.discard(key)
            else:
                self.selected.add(key)

    def toggle_at(self, x, y):
        with self.lock:
            hit = [t for t in self.tracks.values()
                   if t.box[0] <= x <= t.box[0] + t.box[2] and t.box[1] <= y <= t.box[1] + t.box[3]]
            if not hit:
                return None
            t = min(hit, key=lambda t: t.box[2] * t.box[3])
            self.toggle(t.key)
            return t.key

    def select_all(self):
        with self.lock:
            self.selected |= set(self.known.keys())
            self.selected |= {t.key for t in self.tracks.values()}

    def select_none(self):
        with self.lock:
            self.selected.clear()

    def known_list(self):
        with self.lock:
            return [(k, v["thumb"], k in self.selected) for k, v in self.known.items()]

    # -- fotos de referencia -------------------------------------------------
    def ensure_recognizer(self):
        if self.recognizer is None:
            path = ensure_model(SFACE_FILE, SFACE_URL)
            self.recognizer = cv2.FaceRecognizerSF.create(str(path), "")

    def add_reference(self, name, image_bgr):
        """Registra una persona a partir de una foto. Lanza excepción si no hay una cara clara."""
        self.ensure_recognizer()
        dets = DETECTOR.detect(image_bgr, 0.6)
        dets = [d for d in dets if d[5] is not None]
        if not dets:
            raise RuntimeError("No se encontró ninguna cara en la foto (o el detector YuNet no está disponible).")
        d = max(dets, key=lambda d: d[2] * d[3])
        feat = self.recognizer.feature(self.recognizer.alignCrop(image_bgr, d[5]))
        x, y, w, h = [int(v) for v in d[:4]]
        crop = image_bgr[max(0, y):y + h, max(0, x):x + w]
        thumb = cv2.resize(crop, (56, 56), interpolation=cv2.INTER_AREA)
        with self.lock:
            self.references = [r for r in self.references if r["name"] != name]
            self.references.append({"name": name, "feature": feat, "thumb": thumb})
            self.known[name] = {"thumb": thumb, "seen": time.time()}

    def clear_references(self):
        with self.lock:
            names = {r["name"] for r in self.references}
            self.references = []
            for n in names:
                self.known.pop(n, None)
                self.selected.discard(n)
            for t in self.tracks.values():
                t.identity = None
