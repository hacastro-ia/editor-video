"""Personas: seguimiento por puntos (pose), siluetas (segmentación) y traqueo de puntos con HUD."""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from core import yolo
from core.params import Effect, F, I, B, C
from core.pipeline import register
from core.textdraw import draw_texts
from core.utils import grid, hue_color, shift_hue
from effects.pointcloud import PALETAS

# --------------------------------------------------------------------------- esqueleto
# Puntos COCO: 0 nariz, 1-2 ojos, 3-4 orejas, 5-6 hombros, 7-8 codos, 9-10 muñecas,
# 11-12 caderas, 13-14 rodillas, 15-16 tobillos; añadimos 17 = cuello y 18 = pelvis.
BONES = [(0, 17), (17, 5), (17, 6), (5, 7), (7, 9), (6, 8), (8, 10), (17, 18), (18, 11), (18, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
N_PTS = 19
KCONF = 0.3


def _person_color(pid, mode, t):
    if mode == "blanco":
        return (255, 255, 255)
    if mode == "arcoíris":
        return hue_color(pid * 40 + t * 60, 0.85, 1.0)
    return hue_color(pid * 67, 0.85, 1.0)


def _glow(ov, sigma):
    """Resplandor barato: se desenfoca a la mitad de resolución."""
    if sigma < 0.5:
        return None
    h, w = ov.shape[:2]
    small = cv2.resize(ov, (max(2, w // 2), max(2, h // 2)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(0.5, sigma / 2))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _compose(base, ov, glow_sigma, glow_gain=1.5):
    out = cv2.add(base, ov)
    g = _glow(ov, glow_sigma)
    if g is not None:
        out = cv2.add(out, cv2.convertScaleAbs(g, alpha=glow_gain))
    return out


def _pt(p):
    return int(round(float(p[0]))), int(round(float(p[1])))


def _line(ov, a, b, color, th):
    cv2.line(ov, _pt(a), _pt(b), color, int(th), cv2.LINE_AA)


def _scale(color, k):
    return tuple(int(max(0, min(255, c * k))) for c in color)


class Particles:
    """Sistema de partículas con NumPy (posición, velocidad, vida y color)."""

    def __init__(self, cap=3000):
        self.cap = cap
        self.clear()

    def clear(self):
        self.pos = np.zeros((0, 2), np.float32)
        self.vel = np.zeros((0, 2), np.float32)
        self.life = np.zeros(0, np.float32)
        self.maxlife = np.ones(0, np.float32)
        self.col = np.zeros((0, 3), np.float32)

    def emit(self, pos, vel, color, life):
        n = len(pos)
        if n == 0:
            return
        col = np.broadcast_to(np.asarray(color, np.float32), (n, 3))
        lf = np.broadcast_to(np.asarray(life, np.float32), (n,))
        self.pos = np.vstack([self.pos, np.asarray(pos, np.float32)])[-self.cap:]
        self.vel = np.vstack([self.vel, np.asarray(vel, np.float32)])[-self.cap:]
        self.col = np.vstack([self.col, col])[-self.cap:]
        self.life = np.concatenate([self.life, lf])[-self.cap:]
        self.maxlife = np.concatenate([self.maxlife, lf])[-self.cap:]

    def step(self, gravity=0.0, drag=0.985):
        if len(self.pos) == 0:
            return
        self.pos += self.vel
        self.vel *= drag
        self.vel[:, 1] += gravity
        self.life -= 1
        keep = self.life > 0
        if not keep.all():
            self.pos, self.vel, self.life = self.pos[keep], self.vel[keep], self.life[keep]
            self.maxlife, self.col = self.maxlife[keep], self.col[keep]

    def draw(self, canvas, size=1):
        if len(self.pos) == 0:
            return
        h, w = canvas.shape[:2]
        val = (self.col * (self.life / self.maxlife)[:, None]).astype(np.uint8)
        for dy in range(size):
            for dx in range(size):
                xs = self.pos[:, 0].astype(np.int32) + dx
                ys = self.pos[:, 1].astype(np.int32) + dy
                ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
                canvas[ys[ok], xs[ok]] = np.maximum(canvas[ys[ok], xs[ok]], val[ok])


def parse_people(res):
    """Resultado de YOLO pose -> lista de dicts con id, caja y 19 puntos (con cuello y pelvis)."""
    people = []
    if res is None or res.boxes is None or len(res.boxes) == 0 or getattr(res, "keypoints", None) is None:
        return people
    xyxy = yolo.to_np(res.boxes.xyxy)
    ids = yolo.to_np(res.boxes.id) if res.boxes.id is not None else None
    kxy = yolo.to_np(res.keypoints.xy)
    kc = yolo.to_np(res.keypoints.conf) if res.keypoints.conf is not None else np.ones(kxy.shape[:2], np.float32)
    for i in range(len(xyxy)):
        k = kxy[i].astype(np.float32)
        c = kc[i].astype(np.float32)
        k = np.vstack([k, (k[5] + k[6]) / 2, (k[11] + k[12]) / 2])
        c = np.concatenate([c, [min(c[5], c[6]), min(c[11], c[12])]]).astype(np.float32)
        people.append({"id": int(ids[i]) if ids is not None else i + 1, "box": xyxy[i], "pts": k, "conf": c})
    return people


def detect_gesture(pts, conf, hist):
    ok = lambda i: conf[i] > KCONF  # noqa: E731
    shw = abs(float(pts[5][0] - pts[6][0])) + 1e-3 if ok(5) and ok(6) else 60.0
    if ok(0) and ok(9) and ok(10) and pts[9][1] < pts[0][1] and pts[10][1] < pts[0][1]:
        return "¡Brazos arriba!"
    if len(hist) >= 4 and ok(18) and (hist[-1][18][1] - hist[-4][18][1]) < -0.45 * shw:
        return "¡Salto!"
    if ok(5) and ok(6) and ok(9) and ok(10) and abs(pts[9][0] - pts[10][0]) > 1.7 * shw \
            and abs(pts[9][1] - pts[5][1]) < 0.7 * shw and abs(pts[10][1] - pts[6][1]) < 0.7 * shw:
        return "Posición en T"
    if ok(0) and ok(10) and pts[10][1] < pts[0][1]:
        return "Mano derecha arriba"
    if ok(0) and ok(9) and pts[9][1] < pts[0][1]:
        return "Mano izquierda arriba"
    return ""


@register
class PosePoints(Effect):
    name = "Personas: puntos y esqueleto (YOLO)"
    category = "Personas"
    HELP = {"modo": {
        "esqueleto neón": "Huesos y articulaciones luminosos, un color por persona (ID estable).",
        "constelación": "Todos los puntos del cuerpo se unen en una red de estrellas; las personas se conectan entre sí.",
        "relámpagos": "Rayos eléctricos saltan de las manos hacia otras personas y partes del cuerpo.",
        "estelas de luz": "Pintura de luz: manos, pies y cabeza dejan largas estelas.",
        "explosión de partículas": "Chispas nacen de cada articulación según el movimiento; el beat las dispara.",
        "radiografía": "Rayos X: huesos azules, costillas y calavera sobre la imagen en negativo.",
        "marioneta": "Hilos de marioneta cuelgan de lo alto del cuadro hasta cabeza, manos y pies.",
        "multibrazos": "Brazos extra nacen de los hombros y se abren en abanico: una deidad de muchos brazos.",
        "ecos de pose": "Una estela de esqueletos pasados cambia de color y se desvanece.",
        "gestos": "Reconoce gestos (brazos arriba, T, salto…) y los celebra con explosiones.",
    }}
    PARAMS = [
        C("modo", "Modo", list(HELP["modo"].keys())),
        F("fondo", "Imagen original de fondo", 0.3, 0.0, 1.0),
        F("energia", "Energía (alcance / chispas / copias)", 0.5, 0.0, 1.0, mod=("graves", 0.4)),
        I("grosor", "Grosor", 3, 1, 9),
        F("brillo", "Resplandor", 8, 0, 30),
        C("color", "Color", ["por persona", "arcoíris", "blanco"]),
        B("numerar", "Mostrar el número de cada persona", False),
        F("confianza", "Confianza mínima", 0.35, 0.05, 0.95),
        I("cada", "Detectar cada N fotogramas", 1, 1, 6),
        C("modelo", "Modelo", ["yolov8n-pose.pt", "yolov8s-pose.pt", "yolo11n-pose.pt"]),
    ]

    def reset(self):
        self.runner = getattr(self, "runner", None)       # el modelo cargado se conserva
        self.failed = False
        self.people = {}
        self.det_n = 0
        self.parts = Particles()
        self.trail = None
        self.rings = []
        self.rng = np.random.default_rng(5)
        self._last_beat = 0.0
        self._reset_trk = True
        self.labels = []

    # -- modelo ------------------------------------------------------------
    def _load(self, name):
        try:
            self.runner = yolo.Runner(name)
            self.error = ""
        except Exception as e:
            self.failed = True
            self.error = f"No se pudo cargar YOLO pose ({e}). Instala con: pip install ultralytics"

    def _update(self, people, ctx):
        self.det_n += 1
        for p in people:
            st = self.people.get(p["id"])
            if st is None:
                st = {"pts": p["pts"].copy(), "conf": p["conf"], "hist": deque(maxlen=48), "gesture": "",
                      "vel": np.zeros_like(p["pts"]), "seen": self.det_n}
                self.people[p["id"]] = st
            else:
                ok = p["conf"] > KCONF
                new = st["pts"].copy()
                new[ok] = 0.55 * st["pts"][ok] + 0.45 * p["pts"][ok]
                st["vel"] = new - st["pts"]
                st["pts"], st["conf"] = new, p["conf"]
            st["box"], st["seen"] = p["box"], self.det_n
            st["hist"].append(st["pts"].copy())
        for pid in [k for k, v in self.people.items() if self.det_n - v["seen"] > 12]:
            del self.people[pid]

    # -- utilidades de dibujo ---------------------------------------------
    def _vis(self, st, i):
        return st["conf"][i] > KCONF and (st["pts"][i][0] > 0 or st["pts"][i][1] > 0)

    def _bones(self, ov, st, color, th, pts=None):
        pts = st["pts"] if pts is None else pts
        for a, b in BONES:
            if self._vis(st, a) and self._vis(st, b):
                _line(ov, pts[a], pts[b], color, th)

    def _joints(self, ov, st, color, r):
        for i in range(N_PTS):
            if self._vis(st, i):
                cv2.circle(ov, _pt(st["pts"][i]), r, color, -1, cv2.LINE_AA)
                cv2.circle(ov, _pt(st["pts"][i]), max(1, r // 2), (255, 255, 255), -1, cv2.LINE_AA)

    def _bolt(self, ov, a, b, color, th):
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        d = float(np.linalg.norm(b - a))
        if d < 4:
            return
        n = max(4, int(d / 22))
        dirv = (b - a) / d
        perp = np.array([-dirv[1], dirv[0]], np.float32)
        pts = [a]
        for i in range(1, n):
            base = a + (b - a) * i / n
            pts.append(base + perp * self.rng.normal(0, d * 0.045))
        pts.append(b)
        arr = np.array(pts, np.int32).reshape(-1, 1, 2)
        cv2.polylines(ov, [arr], False, color, th + 1, cv2.LINE_AA)
        cv2.polylines(ov, [arr], False, (255, 255, 255), 1, cv2.LINE_AA)

    # -- modos -----------------------------------------------------------------
    def _m_neon(self, ov, ppl, ctx, th, cm, en):
        for pid, st in ppl:
            col = _person_color(pid, cm, ctx.time)
            self._bones(ov, st, col, th)
            self._joints(ov, st, col, th + 2)

    def _m_constelacion(self, ov, ppl, ctx, th, cm, en):
        pts, cols, owner = [], [], []
        for pid, st in ppl:
            col = _person_color(pid, cm, ctx.time)
            for i in range(N_PTS):
                if self._vis(st, i):
                    pts.append(st["pts"][i])
                    cols.append(col)
                    owner.append(pid)
        if len(pts) < 2:
            return
        P = np.array(pts, np.float32)
        diag = float(np.hypot(*ov.shape[:2]))
        reach = (0.06 + 0.34 * en) * diag
        D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
        k = 2 + int(en * 4)
        for i in range(len(P)):
            for j in np.argsort(D[i])[1:k + 1]:
                if D[i, j] < reach:
                    a = 1.0 - D[i, j] / reach
                    c = _scale(tuple((np.array(cols[i]) + np.array(cols[j])) / 2), 0.3 + 0.7 * a)
                    cv2.line(ov, _pt(P[i]), _pt(P[j]), c, 1, cv2.LINE_AA)
        own = np.array(owner)
        for i in range(len(P)):                               # puente hacia la persona más cercana
            other = np.where(own != own[i])[0]
            if len(other):
                j = other[int(np.argmin(D[i, other]))]
                c = _scale(tuple((np.array(cols[i]) + np.array(cols[j])) / 2), 0.45)
                cv2.line(ov, _pt(P[i]), _pt(P[j]), c, 1, cv2.LINE_AA)
        vol = float(ctx.audio.get("volumen", 0.0))
        for i, p in enumerate(P):
            r = 2 + int(2 * (0.5 + 0.5 * np.sin(ctx.time * 5 + i)) + 3 * vol)
            cv2.circle(ov, _pt(p), r, _scale(cols[i], 0.7), -1, cv2.LINE_AA)
            cv2.circle(ov, _pt(p), max(1, r // 2), (255, 255, 255), -1, cv2.LINE_AA)

    def _m_relampagos(self, ov, ppl, ctx, th, cm, en):
        diag = float(np.hypot(*ov.shape[:2]))
        reach = (0.12 + 0.55 * en) * diag
        for pid, st in ppl:
            col = _person_color(pid, cm, ctx.time)
            self._bones(ov, st, _scale(col, 0.45), 1)
            others = [p2["pts"][i] for pid2, p2 in ppl if pid2 != pid for i in range(N_PTS) if self._vis(p2, i)]
            own = [st["pts"][i] for i in (0, 9, 10, 15, 16) if self._vis(st, i)]
            targets = np.array(others if others else own, np.float32).reshape(-1, 2)
            for wi in (9, 10):
                if not self._vis(st, wi) or len(targets) == 0:
                    continue
                w = st["pts"][wi]
                d = np.linalg.norm(targets - w, axis=1)
                d[d < 8] = 1e9
                k = int(np.argmin(d))
                if d[k] < reach:
                    self._bolt(ov, w, targets[k], col, th)
            extra = int(en * 4 + 4 * float(ctx.audio.get("beat", 0.0)))
            for _ in range(extra):
                wi = 9 if self.rng.random() < 0.5 else 10
                if self._vis(st, wi):
                    ang = self.rng.uniform(0, 2 * np.pi)
                    r = self.rng.uniform(0.05, 0.16) * diag
                    self._bolt(ov, st["pts"][wi], st["pts"][wi] + r * np.array([np.cos(ang), np.sin(ang)]), col, th)

    def _m_estelas(self, ov, ppl, ctx, th, cm, en):
        h, w = ov.shape[:2]
        if self.trail is None or self.trail.shape[:2] != (h, w):
            self.trail = np.zeros((h, w, 3), np.float32)
        self.trail *= 0.90 + 0.085 * en
        tmp = np.zeros((h, w, 3), np.uint8)
        for pid, st in ppl:
            col = _person_color(pid, cm, ctx.time)
            if len(st["hist"]) >= 2:
                a, b = st["hist"][-2], st["hist"][-1]
                for j in (0, 9, 10, 15, 16, 7, 8):
                    if self._vis(st, j):
                        _line(tmp, a[j], b[j], col, th + 1)
        self.trail = np.maximum(self.trail, tmp.astype(np.float32))
        ov[:] = np.clip(self.trail, 0, 255).astype(np.uint8)
        for pid, st in ppl:
            self._bones(ov, st, _scale(_person_color(pid, cm, ctx.time), 0.35), 1)

    def _emit_from(self, st, col, rate, burst, spread):
        pos, vel = [], []
        for i in range(N_PTS):
            if not self._vis(st, i):
                continue
            sp = float(np.linalg.norm(st["vel"][i]))
            n = int(sp * 0.35 * rate + burst)
            if self.rng.random() < 0.15 * rate:
                n += 1
            for _ in range(min(n, 30)):
                pos.append(st["pts"][i])
                vel.append(st["vel"][i] * 0.5 + self.rng.normal(0, spread, 2))
        if pos:
            lf = self.rng.uniform(20, 55, len(pos))
            self.parts.emit(np.array(pos), np.array(vel), np.array(col, np.float32) * 0.9 + 25, lf)

    def _m_particulas(self, ov, ppl, ctx, th, cm, en):
        beat = ctx.audio.get("beat", 0.0)
        hit = beat > 0.95 and self._last_beat <= 0.95
        for pid, st in ppl:
            col = _person_color(pid, cm, ctx.time)
            self._emit_from(st, col, 0.5 + en * 2, int(6 + 20 * en) if hit else 0, 1.0 + 3 * en)
            self._bones(ov, st, _scale(col, 0.35), 1)
        self.parts.step(gravity=0.12)
        self.parts.draw(ov, 2 if th > 2 else 1)

    def _m_xray(self, ov, ppl, ctx, th, cm, en):
        for pid, st in ppl:
            bone, dark = (255, 215, 150), (90, 40, 10)
            t = th + 2
            for a, b in BONES[:14]:
                if self._vis(st, a) and self._vis(st, b):
                    _line(ov, st["pts"][a], st["pts"][b], bone, t)
                    _line(ov, st["pts"][a], st["pts"][b], dark, max(1, t // 3))
            for i in (5, 6, 7, 8, 11, 12, 13, 14, 17, 18):
                if self._vis(st, i):
                    cv2.circle(ov, _pt(st["pts"][i]), t, bone, -1, cv2.LINE_AA)
            if all(self._vis(st, i) for i in (5, 6, 11, 12)):
                for f in (0.18, 0.34, 0.50, 0.66):
                    lft = st["pts"][5] + (st["pts"][11] - st["pts"][5]) * f
                    rgt = st["pts"][6] + (st["pts"][12] - st["pts"][6]) * f
                    mid = (lft + rgt) / 2 + np.array([0, 5 + 4 * f])
                    arr = np.array([lft, mid, rgt], np.int32).reshape(-1, 1, 2)
                    cv2.polylines(ov, [arr], False, bone, max(1, t - 1), cv2.LINE_AA)
            if self._vis(st, 0) and self._vis(st, 17):
                head = st["pts"][0]
                r = int(max(14, np.linalg.norm(st["pts"][17] - head) * 0.75))
                c = (int(head[0]), int(head[1] - r * 0.25))
                cv2.circle(ov, c, r, bone, -1, cv2.LINE_AA)
                cv2.rectangle(ov, (c[0] - r // 2, c[1] + r - 4), (c[0] + r // 2, c[1] + r + r // 3), bone, -1)
                for sx in (-1, 1):
                    cv2.circle(ov, (c[0] + sx * r // 2, c[1] - r // 8), max(3, r // 4), dark, -1, cv2.LINE_AA)
                cv2.fillPoly(ov, [np.array([[c[0], c[1] + r // 6], [c[0] - r // 8, c[1] + r // 2],
                                            [c[0] + r // 8, c[1] + r // 2]], np.int32)], dark)
                for k in range(-2, 3):
                    cv2.line(ov, (c[0] + k * r // 5, c[1] + r + 2), (c[0] + k * r // 5, c[1] + r + r // 3 - 2), dark, 1)

    def _m_marioneta(self, ov, ppl, ctx, th, cm, en):
        for pid, st in ppl:
            wood, line = (70, 140, 205), (225, 225, 225)
            for a, b in BONES[:14]:
                if self._vis(st, a) and self._vis(st, b):
                    _line(ov, st["pts"][a], st["pts"][b], wood, th + 2)
            for j in (0, 9, 10, 15, 16):
                if not self._vis(st, j):
                    continue
                p = st["pts"][j]
                sway = 12 * np.sin(ctx.time * 1.7 + j + pid)
                top = np.array([p[0] + sway, 0], np.float32)
                mid = (top + p) / 2 + np.array([sway * 0.6, 0])
                arr = np.array([top, mid, p], np.int32).reshape(-1, 1, 2)
                cv2.polylines(ov, [arr], False, line, 1, cv2.LINE_AA)
                cv2.circle(ov, _pt(top), 3, line, -1, cv2.LINE_AA)
            for i in range(N_PTS):
                if self._vis(st, i):
                    cv2.circle(ov, _pt(st["pts"][i]), th + 3, (35, 80, 130), -1, cv2.LINE_AA)
                    cv2.circle(ov, _pt(st["pts"][i]), th + 1, wood, -1, cv2.LINE_AA)

    def _m_multibrazos(self, ov, ppl, ctx, th, cm, en):
        n = 1 + int(en * 5)                                   # pares de brazos extra
        for pid, st in ppl:
            base = _person_color(pid, cm, ctx.time)
            self._bones(ov, st, base, th)
            self._joints(ov, st, base, th + 1)
            for sh, el, wr, sign in ((5, 7, 9, 1), (6, 8, 10, -1)):
                if not (self._vis(st, sh) and self._vis(st, el) and self._vis(st, wr)):
                    continue
                for k in range(1, n + 1):
                    ang = sign * (k * 0.55 + 0.35 * np.sin(ctx.time * 2.2 + k + pid))
                    cs, sn = np.cos(ang), np.sin(ang)
                    pts = []
                    for j in (el, wr):
                        rel = st["pts"][j] - st["pts"][sh]
                        pts.append(st["pts"][sh] + np.array([rel[0] * cs - rel[1] * sn, rel[0] * sn + rel[1] * cs]))
                    col = hue_color(pid * 67 + k * 55 + ctx.time * 30, 0.85, 1.0)
                    _line(ov, st["pts"][sh], pts[0], col, th)
                    _line(ov, pts[0], pts[1], col, th)
                    cv2.circle(ov, _pt(pts[1]), th + 1, (255, 255, 255), -1, cv2.LINE_AA)

    def _m_ecos(self, ov, ppl, ctx, th, cm, en):
        steps = 3 + int(en * 4)
        for pid, st in ppl:
            base = (pid * 67 + ctx.time * 40) % 360
            for k in range(14, 0, -1):
                idx = -1 - k * 2
                if len(st["hist"]) < -idx:
                    continue
                col = _scale(hue_color(base + k * 18, 0.9, 1.0), 1.0 - k / 15.0)
                self._bones(ov, st, col, max(1, th - k // steps), pts=st["hist"][idx])
            self._bones(ov, st, (255, 255, 255), max(1, th - 1))

    def _m_gestos(self, ov, ppl, ctx, th, cm, en):
        beat = ctx.audio.get("beat", 0.0)
        for pid, st in ppl:
            col = _person_color(pid, cm, ctx.time)
            self._bones(ov, st, _scale(col, 0.8), max(1, th - 1))
            self._joints(ov, st, col, th + 1)
            g = detect_gesture(st["pts"], st["conf"], st["hist"])
            if g and g != st["gesture"]:
                c = np.array([(st["box"][0] + st["box"][2]) / 2, (st["box"][1] + st["box"][3]) / 2], np.float32)
                ang = self.rng.uniform(0, 2 * np.pi, 140)
                sp = self.rng.uniform(2, 9, 140)
                self.parts.emit(np.tile(c, (140, 1)), np.stack([np.cos(ang) * sp, np.sin(ang) * sp], 1),
                                np.array(col, np.float32) * 0.9 + 30, self.rng.uniform(25, 60, 140))
                self.rings.append([c, 10.0, 22, col])
            st["gesture"] = g
            if g:
                x = st["pts"][0][0] if self._vis(st, 0) else (st["box"][0] + st["box"][2]) / 2
                y = (st["pts"][0][1] if self._vis(st, 0) else st["box"][1]) - 55
                self.labels.append((x - 40, max(4, y), g, (0, 0, 0), col))
        for r in self.rings:
            cv2.circle(ov, _pt(r[0]), int(r[1]), _scale(r[3], r[2] / 22.0), max(1, th), cv2.LINE_AA)
            r[1] += 9
            r[2] -= 1
        self.rings = [r for r in self.rings if r[2] > 0]
        self.parts.step(gravity=0.1)
        self.parts.draw(ov, 2)

    MODES = {"esqueleto neón": "_m_neon", "constelación": "_m_constelacion", "relámpagos": "_m_relampagos",
             "estelas de luz": "_m_estelas", "explosión de partículas": "_m_particulas",
             "radiografía": "_m_xray", "marioneta": "_m_marioneta", "multibrazos": "_m_multibrazos",
             "ecos de pose": "_m_ecos", "gestos": "_m_gestos"}

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
        h, w = frame.shape[:2]
        if self._reset_trk:
            self.runner.reset_tracking()
            self._reset_trk = False
        if ctx.frame_index % self.value("cada", ctx) == 0 or not self.people:
            res = self.runner(frame, conf=self.value("confianza", ctx), classes=[0])
            self._update(parse_people(res), ctx)
        modo = self.value("modo", ctx)
        th, en, cm = self.value("grosor", ctx), self.value("energia", ctx), self.value("color", ctx)
        ppl = [(pid, st) for pid, st in self.people.items() if self.det_n - st["seen"] <= 6]
        ov = np.zeros((h, w, 3), np.uint8)
        self.labels = []
        getattr(self, self.MODES[modo])(ov, ppl, ctx, th, cm, en)
        self._last_beat = float(ctx.audio.get("beat", 0.0))
        fondo = self.value("fondo", ctx)
        if modo == "radiografía":
            g = cv2.cvtColor(255 - cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            base = (g.astype(np.float32) * np.array([1.0, 0.78, 0.45], np.float32) * (0.15 + 0.6 * fondo)).astype(np.uint8)
        else:
            base = cv2.convertScaleAbs(frame, alpha=fondo)
        out = _compose(base, ov, self.value("brillo", ctx))
        if self.value("numerar", ctx):
            for pid, st in ppl:
                self.labels.append((st["box"][0], max(2, st["box"][1] - 24), f"#{pid}", (0, 0, 0),
                                    _person_color(pid, cm, ctx.time)))
        if self.labels:
            out = draw_texts(out, self.labels, max(13, h // 30))
        return out


# --------------------------------------------------------------------------- siluetas
@register
class PersonSilhouettes(Effect):
    name = "Personas: siluetas (YOLO seg)"
    category = "Personas"
    HELP = {"modo": {
        "persona de puntos": "La persona se convierte en una nube de granos crema que se deshace al moverse.",
        "aislar persona": "Solo la persona queda intacta; el fondo se transforma.",
        "silueta neón": "Contorno luminoso con un color por persona.",
        "clones": "Copias retardadas y teñidas de la persona, como un efecto de eco corporal.",
        "desintegración": "La persona se rompe en partículas que vuelan; el beat controla cuánto.",
    }}
    PARAMS = [
        C("modo", "Modo", list(HELP["modo"].keys())),
        C("fondo_aislar", "Fondo (modo «aislar»)", ["negro", "desenfocado", "pixelado", "arcoíris", "negativo"]),
        F("opacidad_fondo", "Imagen original de fondo", 0.25, 0.0, 1.0),
        F("energia", "Energía (dispersión / clones / desintegración)", 0.5, 0.0, 1.0, mod=("graves", 0.4)),
        F("brillo", "Resplandor", 8, 0, 30),
        F("aberracion", "Aberración cromática (px)", 3.0, 0.0, 20.0, mod=("beat", 0.2)),
        C("paleta", "Paleta (persona de puntos)", list(PALETAS.keys())),
        F("confianza", "Confianza mínima", 0.35, 0.05, 0.95),
        I("cada", "Detectar cada N fotogramas", 1, 1, 6),
        C("modelo", "Modelo", ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolo11n-seg.pt"]),
    ]

    def reset(self):
        self.runner = getattr(self, "runner", None)       # el modelo cargado se conserva
        self.failed = False
        self.masks = []
        self.rng = np.random.default_rng(11)
        self.noise = None
        self.shape = None
        self.prev_mask = None
        self.dis_noise = None
        self.hist = deque(maxlen=40)
        self.parts = Particles(4000)
        self._reset_trk = True

    def _load(self, name):
        try:
            self.runner = yolo.Runner(name)
            self.error = ""
        except Exception as e:
            self.failed = True
            self.error = f"No se pudo cargar YOLO seg ({e}). Instala con: pip install ultralytics"

    @staticmethod
    def parse_masks(res, h, w):
        out = []
        if res is None or res.boxes is None or len(res.boxes) == 0 or getattr(res, "masks", None) is None:
            return out
        ids = yolo.to_np(res.boxes.id) if res.boxes.id is not None else None
        for i, poly in enumerate(res.masks.xy):
            if poly is None or len(poly) < 3:
                continue
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [np.asarray(poly, np.int32)], 255)
            out.append((int(ids[i]) if ids is not None else i + 1, m))
        return out

    # -- modos -----------------------------------------------------------------
    def _m_puntos(self, frame, mask, ctx, en):
        h, w = mask.shape
        lum = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        lum = np.clip((lum - 0.15) * 1.6, 0, 1)
        oy, ox = int(self.rng.integers(0, h)), int(self.rng.integers(0, w))
        n1 = self.noise[oy:oy + h, ox:ox + w]
        oy, ox = int(self.rng.integers(0, h)), int(self.rng.integers(0, w))
        n2 = self.noise[oy:oy + h, ox:ox + w]
        inside = (mask > 0) & (lum * 1.4 + 0.1 > n1)
        dist = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
        moved = cv2.GaussianBlur(cv2.absdiff(mask, self.prev_mask).astype(np.float32) / 255.0, (0, 0), 6)
        scatter = np.exp(-dist / (3 + 14 * en)) * (0.10 + 0.45 * en) * (1.0 + 4.0 * np.clip(moved * 4, 0, 1))
        outside = (mask == 0) & (n2 < scatter)
        v = (inside | outside).astype(np.float32)
        fg, bg = PALETAS[self.value("paleta", ctx)]
        fg, bg = np.array(fg, np.float32), np.array(bg, np.float32)
        chans = np.repeat(v[..., None], 3, axis=2)
        a = int(round(self.value("aberracion", ctx)))
        if a > 0:
            vs = cv2.GaussianBlur(v, (0, 0), 2.0)
            chans[..., 2] += np.roll(vs, -a, axis=1) - vs
            chans[..., 0] += np.roll(vs, a, axis=1) - vs
            np.clip(chans, 0, 1, out=chans)
        return np.clip(bg + (fg - bg) * chans, 0, 255).astype(np.uint8)

    def _m_aislar(self, frame, mask, ctx, en):
        kind = self.value("fondo_aislar", ctx)
        if kind == "negro":
            bg = np.zeros_like(frame)
        elif kind == "desenfocado":
            bg = cv2.convertScaleAbs(cv2.GaussianBlur(frame, (0, 0), 14), alpha=0.6)
        elif kind == "pixelado":
            h, w = frame.shape[:2]
            bg = cv2.resize(cv2.resize(frame, (max(2, w // 24), max(2, h // 24)), interpolation=cv2.INTER_AREA),
                            (w, h), interpolation=cv2.INTER_NEAREST)
        elif kind == "arcoíris":
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            gx, gy = grid(frame.shape[1], frame.shape[0])
            hsv[..., 0] = np.mod(gx / frame.shape[1] * 120 + ctx.time * 40, 180).astype(np.uint8)
            hsv[..., 1] = 255
            bg = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            bg = 255 - frame
        soft = cv2.GaussianBlur(mask, (0, 0), 2.0).astype(np.float32)[..., None] / 255.0
        return (frame * soft + bg * (1 - soft)).astype(np.uint8)

    def _m_neon(self, frame, masks, ctx, en):
        base = cv2.convertScaleAbs(frame, alpha=self.value("opacidad_fondo", ctx))
        ov = np.zeros_like(frame)
        for pid, m in masks:
            col = hue_color(pid * 67 + ctx.time * 30, 0.85, 1.0)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            tint = np.zeros_like(frame)
            tint[:] = _scale(col, 0.18)
            ov = np.where(m[..., None] > 0, cv2.add(ov, tint), ov)
            cv2.drawContours(ov, cnts, -1, col, 3, cv2.LINE_AA)
            cv2.drawContours(ov, cnts, -1, (255, 255, 255), 1, cv2.LINE_AA)
        return _compose(base, ov, self.value("brillo", ctx) + 2)

    @staticmethod
    def _shift(img, dx):
        out = np.zeros_like(img)
        if dx >= 0:
            out[:, dx:] = img[:, :img.shape[1] - dx]
        else:
            out[:, :dx] = img[:, -dx:]
        return out

    @staticmethod
    def _hue_matrix(deg):
        """Rotación de matiz como matriz 3x3 sobre BGR (mucho más barata que pasar por HSV)."""
        a = np.deg2rad(deg)
        c, s = np.cos(a), np.sin(a)
        r = np.array([[0.299 + 0.701 * c + 0.168 * s, 0.587 - 0.587 * c + 0.330 * s, 0.114 - 0.114 * c - 0.497 * s],
                      [0.299 - 0.299 * c - 0.328 * s, 0.587 + 0.413 * c + 0.035 * s, 0.114 - 0.114 * c + 0.292 * s],
                      [0.299 - 0.300 * c + 1.250 * s, 0.587 - 0.588 * c - 1.050 * s, 0.114 + 0.886 * c - 0.203 * s]],
                     np.float32)                                       # filas/columnas en orden RGB
        return r[::-1, ::-1].copy()                                    # a orden BGR

    def _m_clones(self, frame, soft, ctx, en):
        h, w = soft.shape
        canvas = cv2.convertScaleAbs(frame, alpha=self.value("opacidad_fondo", ctx))
        n, delay = 2 + int(en * 3), 4
        for k in range(n, 0, -1):
            idx = -1 - k * delay
            if len(self.hist) < -idx:
                continue
            pf, ps = self.hist[idx]
            dx = (1 if k % 2 else -1) * ((k + 1) // 2) * int(w * 0.06)
            pf = cv2.transform(self._shift(pf, dx), self._hue_matrix(k * 28))
            a = (self._shift(ps, dx).astype(np.float32) / 255.0)[..., None] * (0.85 - 0.08 * k)
            canvas = (pf * a + canvas * (1 - a)).astype(np.uint8)
        a = (soft.astype(np.float32) / 255.0)[..., None]
        return (frame * a + canvas * (1 - a)).astype(np.uint8)

    def _m_desintegra(self, frame, mask, ctx, en):
        h, w = mask.shape
        p = float(np.clip(en, 0.0, 1.0))
        if self.dis_noise is None or self.dis_noise.shape != (h, w):
            nz = self.rng.random((max(2, h // 8), max(2, w // 8)), dtype=np.float32)
            self.dis_noise = cv2.GaussianBlur(cv2.resize(nz, (w, h), interpolation=cv2.INTER_CUBIC), (0, 0), 6)
            self.dis_noise = (self.dis_noise - self.dis_noise.min()) / (np.ptp(self.dis_noise) + 1e-6)
        gone = (mask > 0) & (self.dis_noise < p)
        ys, xs = np.nonzero(gone)
        if len(ys) > 0:
            k = min(len(ys), int(250 + 1400 * p))
            sel = self.rng.choice(len(ys), size=k, replace=False)
            pos = np.stack([xs[sel], ys[sel]], 1).astype(np.float32)
            wind = np.array([2.0 + 6 * p, -1.5 - 3 * p], np.float32)
            vel = wind + self.rng.normal(0, 1.4 + 3 * p, (k, 2))
            self.parts.emit(pos, vel, frame[ys[sel], xs[sel]].astype(np.float32), self.rng.uniform(18, 60, k))
        keep = ((mask > 0) & ~gone).astype(np.float32)[..., None]
        canvas = cv2.convertScaleAbs(frame, alpha=self.value("opacidad_fondo", ctx))
        out = (frame * keep + canvas * (1 - keep)).astype(np.uint8)
        ov = np.zeros_like(out)
        self.parts.step(gravity=-0.02, drag=0.97)
        self.parts.draw(ov, 2)
        return _compose(out, ov, self.value("brillo", ctx) * 0.5, 1.0)

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
        h, w = frame.shape[:2]
        if self.shape != (h, w):
            self.shape = (h, w)
            self.noise = self.rng.random((2 * h, 2 * w), dtype=np.float32)
            self.prev_mask = np.zeros((h, w), np.uint8)
            self.dis_noise = None
            self.hist.clear()
        if self._reset_trk:
            self.runner.reset_tracking()
            self._reset_trk = False
        if ctx.frame_index % self.value("cada", ctx) == 0:
            res = self.runner(frame, conf=self.value("confianza", ctx), classes=[0])
            self.masks = self.parse_masks(res, h, w)
        mask = np.zeros((h, w), np.uint8)
        for _, m in self.masks:
            mask = np.maximum(mask, m)
        modo, en = self.value("modo", ctx), self.value("energia", ctx)
        soft = cv2.GaussianBlur(mask, (0, 0), 1.5)
        if modo == "persona de puntos":
            out = self._m_puntos(frame, mask, ctx, en)
        elif modo == "aislar persona":
            out = self._m_aislar(frame, mask, ctx, en)
        elif modo == "silueta neón":
            out = self._m_neon(frame, self.masks, ctx, en)
        elif modo == "clones":
            out = self._m_clones(frame, soft, ctx, en)
        else:
            out = self._m_desintegra(frame, mask, ctx, en)
        self.hist.append((frame.copy(), soft))
        self.prev_mask = mask
        return out


# --------------------------------------------------------------------------- traqueo con HUD
@register
class PointTracker(Effect):
    name = "Traqueo de puntos (HUD)"
    category = "Personas"
    HELP = {"modo": {
        "red de puntos": "Puntos que siguen lo que se mueve y se unen con líneas a sus vecinos.",
        "HUD de coordenadas": "Cuadritos de seguimiento con coordenadas y líneas, estilo visión artificial.",
        "estelas": "Cada punto deja una estela con el color de su velocidad.",
    }}
    L = 16                                          # largo de las estelas (fotogramas)
    PARAMS = [
        C("modo", "Modo", list(HELP["modo"].keys())),
        I("puntos", "Cantidad de puntos", 140, 20, 500),
        F("alcance", "Alcance de las conexiones", 0.12, 0.03, 0.4, mod=("graves", 0.3)),
        B("solo_movimiento", "Seguir solo lo que se mueve", True),
        F("fondo", "Imagen original de fondo", 0.5, 0.0, 1.0),
        F("brillo", "Resplandor", 6, 0, 25),
        I("grosor", "Grosor", 1, 1, 5),
    ]

    def reset(self):
        self.pts = np.zeros((0, 2), np.float32)
        self.tr = np.zeros((0, self.L, 2), np.float32)      # historial de cada punto
        self.vel = np.zeros(0, np.float32)
        self.prev_gray = None
        self.rng = np.random.default_rng(3)

    def _detect(self, gray, n, only_motion):
        mask = np.full(gray.shape, 255, np.uint8)
        if only_motion and self.prev_gray is not None:
            diff = cv2.GaussianBlur(cv2.absdiff(gray, self.prev_gray), (0, 0), 3)
            mask = cv2.dilate((diff > 7).astype(np.uint8) * 255, np.ones((9, 9), np.uint8))
        for p in self.pts:                                   # no repetir sobre puntos existentes
            cv2.circle(mask, (int(p[0]), int(p[1])), 8, 0, -1)
        new = cv2.goodFeaturesToTrack(gray, maxCorners=max(1, n - len(self.pts)), qualityLevel=0.01,
                                      minDistance=9, mask=mask)
        if new is not None:
            new = new.reshape(-1, 2).astype(np.float32)
            self.pts = np.vstack([self.pts, new])
            self.tr = np.concatenate([self.tr, np.repeat(new[:, None, :], self.L, axis=1)])
            self.vel = np.concatenate([self.vel, np.zeros(len(new), np.float32)])

    def _track(self, gray, w, h, only_motion):
        if self.prev_gray is None or self.prev_gray.shape != gray.shape or len(self.pts) == 0:
            return
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.pts.reshape(-1, 1, 2), None,
                                              winSize=(21, 21), maxLevel=3)
        nxt = nxt.reshape(-1, 2)
        ok = (st.reshape(-1) == 1) & (nxt[:, 0] >= 0) & (nxt[:, 0] < w) & (nxt[:, 1] >= 0) & (nxt[:, 1] < h)
        vel = np.linalg.norm(nxt - self.pts, axis=1)
        if only_motion:                                      # los puntos quietos se van renovando
            ok &= ~((vel < 0.15) & (self.rng.random(len(vel)) < 0.08))
        self.pts, self.vel = nxt[ok], vel[ok]
        self.tr = np.concatenate([self.tr[ok][:, 1:], self.pts[:, None, :]], axis=1)

    def process(self, frame, ctx):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        n = self.value("puntos", ctx)
        only_motion = bool(self.value("solo_movimiento", ctx))
        self._track(gray, w, h, only_motion)
        if len(self.pts) < n * 0.7 or ctx.frame_index % 12 == 0:
            self._detect(gray, n, only_motion)
        self.prev_gray = gray

        P, vel = self.pts, self.vel
        modo, th = self.value("modo", ctx), self.value("grosor", ctx)
        ov = np.zeros_like(frame)
        texts = []
        if modo == "estelas":
            half = self.L // 2
            for i in range(len(P)):
                col = hue_color(200 - min(float(vel[i]), 25) * 8, 0.8, 1.0)
                cv2.polylines(ov, [self.tr[i, :half + 1].astype(np.int32).reshape(-1, 1, 2)], False,
                              _scale(col, 0.4), th, cv2.LINE_AA)
                cv2.polylines(ov, [self.tr[i, half:].astype(np.int32).reshape(-1, 1, 2)], False, col, th, cv2.LINE_AA)
        else:
            reach = self.value("alcance", ctx) * float(np.hypot(w, h))
            if len(P) > 1:
                D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
                ii, jj = np.where(np.triu(D < reach, 1))
                for k in np.argsort(D[ii, jj])[:600]:
                    i, j = ii[k], jj[k]
                    a = 1.0 - D[i, j] / reach
                    cv2.line(ov, _pt(P[i]), _pt(P[j]), _scale((255, 230, 120), 0.2 + 0.8 * a), th, cv2.LINE_AA)
            for i, p in enumerate(P):
                sp = float(vel[i])
                col = hue_color(200 - min(sp, 25) * 8, 0.8, 1.0)
                if modo == "HUD de coordenadas":
                    s = 5 + int(min(sp, 12))
                    cv2.rectangle(ov, (int(p[0]) - s, int(p[1]) - s), (int(p[0]) + s, int(p[1]) + s), col, 1)
                    if i % 4 == 0 and len(texts) < 30:
                        texts.append((p[0] + s + 2, p[1] - s, f"{int(p[0])},{int(p[1])}", col, None))
                else:
                    cv2.circle(ov, _pt(p), 2 + int(min(sp, 6) / 2), col, -1, cv2.LINE_AA)
        out = _compose(cv2.convertScaleAbs(frame, alpha=self.value("fondo", ctx)), ov, self.value("brillo", ctx))
        if texts:
            out = draw_texts(out, texts, max(11, h // 42))
        return out
