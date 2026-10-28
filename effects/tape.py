"""Cinta analógica (Hi8 / 8 mm / VHS): el ruido característico al pausar, rebobinar o avanzar rápido.

Un video digital (Digital8, DV, MP4…) no guarda ese ruido: nace de que el cabezal giratorio de una cinta
analógica deja de seguir las pistas al detenerse o al ir rápido. Aquí se reconstruye de forma procedural,
imitando lo que se ve en una videocámara real:

  · bandas horizontales de «nieve» que se desplazan (más y más rápidas al buscar, 1-2 lentas en pausa);
  · errores de base de tiempo: las líneas tiemblan, la parte superior se «dobla» (flagging) y el borde inferior
    se curva con ruido del cabezal;
  · imagen blanda, color emborronado y desfasado, que se desvanece dentro de las bandas;
  · motas blancas (dropouts), parpadeo de ganancia y grano;
  · en pausa, solo se ve un campo de la imagen (menos resolución vertical y un leve temblor);
  · al cambiar de estado hay un «golpe» mecánico y las bandas entran o se disipan poco a poco.

El estado (play / pause / rewind / ff) lo entrega el reproductor (ctx.transport).
"""
from __future__ import annotations

import datetime

import cv2
import numpy as np

from core.params import Effect, F, I, B, C
from core.pipeline import register
from core.textdraw import draw_texts

MODOS = [
    "automático (detecta pausa y rebobinado)",
    "siempre pausa",
    "siempre rebobinado",
    "siempre avance rápido",
    "reproducción (cinta gastada)",
]
FORZADO = {MODOS[1]: "pause", MODOS[2]: "rewind", MODOS[3]: "ff", MODOS[4]: "wear"}

# (desenfoque horizontal de luma, desenfoque de croma, factor de ruido) según el formato de cinta
CALIDAD = {"Hi8": (0.7, 3.0, 0.9), "8 mm estándar": (1.1, 4.0, 1.0), "VHS": (1.5, 5.5, 1.15)}

# Valores objetivo de cada estado. «bands» = bandas simultáneas, «speed» = fracción de alto por fotograma
# (positivo = hacia abajo), «hmin/hmax» = alto de las bandas, el resto son intensidades 0..1.
_Z = dict(bands=0.0, speed=0.0, hmin=0.06, hmax=0.12, amp=0.0, jitter=0.0, flag=0.0, head=0.0, color=0.0,
          drop=0.0, grain=0.0, flicker=0.0, vjit=0.0, field=0.0, soft=0.0)
VEC = {
    "play": dict(_Z),
    "wear": dict(bands=0.25, speed=0.006, hmin=0.04, hmax=0.09, amp=0.55, jitter=0.25, flag=0.30, head=0.40,
                 color=0.08, drop=0.30, grain=0.30, flicker=0.22, vjit=0.15, field=0.0, soft=0.25),
    "pause": dict(bands=1.5, speed=0.005, hmin=0.07, hmax=0.20, amp=0.88, jitter=0.55, flag=0.60, head=0.90,
                  color=0.28, drop=0.35, grain=0.50, flicker=0.30, vjit=0.70, field=1.0, soft=0.60),
    "rewind": dict(bands=5.0, speed=0.017, hmin=0.05, hmax=0.15, amp=0.92, jitter=1.0, flag=1.0, head=1.0,
                   color=0.68, drop=1.0, grain=0.70, flicker=0.60, vjit=1.0, field=0.0, soft=0.70),
    "ff": dict(bands=4.0, speed=-0.015, hmin=0.05, hmax=0.14, amp=0.90, jitter=0.90, flag=0.90, head=1.0,
               color=0.50, drop=0.85, grain=0.65, flicker=0.55, vjit=0.90, field=0.0, soft=0.65),
}
OSD = {"pause": "PAUSE II", "rewind": "<< REW", "ff": "FF >>"}


@register
class TapeNoise(Effect):
    name = "Cinta 8mm: ruido de pausa y rebobinado"
    category = "Cinta analógica"
    LIVE_WHEN_PAUSED = True          # el ruido se sigue animando aunque el video esté detenido
    HELP = {
        "modo": {
            MODOS[0]: "Sin ruido al reproducir; aparece solo al pausar (🡒 bandas lentas) o al rebobinar / avanzar rápido.",
            MODOS[1]: "Fuerza el ruido de pausa en todo momento (para probar el aspecto).",
            MODOS[2]: "Fuerza el ruido de rebobinado: muchas bandas rápidas, color perdido y temblor.",
            MODOS[3]: "Fuerza el ruido de avance rápido: bandas que suben en sentido contrario.",
            MODOS[4]: "Reproducción normal de una cinta usada: temblor leve, motas, grano y alguna banda suelta.",
        },
        "calidad": {
            "Hi8": "Más nítida (video 8 de alta banda).",
            "8 mm estándar": "Más suave y con más color emborronado.",
            "VHS": "Más blanda y con más ruido.",
        },
    }
    PARAMS = [
        C("modo", "Modo", MODOS),
        C("calidad", "Formato de cinta", list(CALIDAD.keys()), "8 mm estándar"),
        F("intensidad", "Intensidad general", 1.0, 0.0, 1.5),
        F("bandas", "Cantidad de bandas de ruido", 1.0, 0.0, 2.0),
        F("jitter", "Temblor de las líneas", 1.0, 0.0, 2.0),
        F("color", "Pérdida y desfase de color", 1.0, 0.0, 2.0),
        F("blancos", "Motas blancas (dropouts)", 1.0, 0.0, 2.0),
        F("cabezal", "Ruido del cabezal (borde inferior)", 1.0, 0.0, 2.0),
        F("grano", "Grano de fondo", 1.0, 0.0, 2.0),
        I("transicion", "Suavidad al cambiar de estado (fotogramas)", 8, 1, 30),
        B("osd", "Mostrar indicador en pantalla (PAUSE / REW / FF)", False),
        F("contornos", "Halos de contorno (ringing)", 0.0, 0.0, 1.0),
        F("ondulacion", "Ondulación lenta (wobble)", 0.5, 0.0, 2.0),
        C("marca", "Marca de la cámara", ["ninguna", "REC ● fecha y hora", "PLAY ▶ fecha y hora"]),
    ]

    # ------------------------------------------------------------------ estado
    def reset(self):
        self.rng = np.random.default_rng(8)
        self.shape = None
        self.vec = dict(VEC["play"])
        self.prev_state = None
        self.bands = []
        self.n = 0
        self.gain = 1.0
        self.kick = 0.0
        self.roll = 0.0
        self.burst = 0.0
        self.flag_sign = 1.0
        self.curl = 0.0
        self.t0 = datetime.datetime.now().replace(second=0, microsecond=0)

    def _init(self, h, w):
        self.shape = (h, w)
        rng = self.rng
        snow = rng.standard_normal((2 * h, 2 * w), dtype=np.float32)
        snow = cv2.GaussianBlur(snow, (0, 0), sigmaX=1.5 * w / 640.0, sigmaY=0.35)      # nieve alargada en horizontal
        self.snow = (snow / (snow.std() + 1e-6)).astype(np.float32)
        self.fine = rng.standard_normal((2 * h, 2 * w), dtype=np.float32)               # grano fino
        self.yy = np.arange(h, dtype=np.float32)
        self.X, self.Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    def _crop(self, tile, h, w):
        oy = int(self.rng.integers(0, h))
        ox = int(self.rng.integers(0, w))
        return tile[oy:oy + h, ox:ox + w]

    # ------------------------------------------------------------------ bandas
    def _update_bands(self, want, V, h):
        rng = self.rng
        for b in self.bands:
            b["age"] += 1
            b["y"] += V["speed"] * (0.7 + 0.6 * b["spd"])
        self.bands = [b for b in self.bands if b["age"] < b["life"] and -0.35 < b["y"] < 1.35]
        active = [b for b in self.bands if not b.get("dying")]
        if len(active) > want + 0.5 and active:
            old = max(active, key=lambda b: b["age"])
            old["dying"] = True
            old["life"] = min(old["life"], old["age"] + 8)
        n_active = len(active)
        for _ in range(3 if self.burst > 0.25 else 1):               # al cambiar de estado entran varias a la vez
            if n_active >= np.ceil(want - 1e-6):
                break
            p = min(1.0, (want - n_active) * 0.14 + self.burst * 0.6)
            if rng.random() < p:
                edge_top = V["speed"] > 0
                if self.burst < 0.25 and n_active and abs(V["speed"]) > 1e-4 and rng.random() < 0.7:
                    y = -0.08 if edge_top else 1.08                 # entra por el borde, como una banda que pasa
                else:
                    y = float(rng.uniform(0.02, 0.98))              # al empezar o si no hay ninguna visible
                self.bands.append({"y": y, "h": float(rng.uniform(V["hmin"], V["hmax"])),
                                   "tilt": float(rng.uniform(-0.045, 0.045)), "spd": float(rng.random()),
                                   "age": 0, "life": int(rng.integers(70, 260))})
                n_active += 1
        self.burst *= 0.8

    def _band_mask(self, h, w, V, inten):
        if not self.bands:
            return None
        wl = max(16, w // 16)
        yy = (np.arange(h, dtype=np.float32) / h)[:, None]
        xx = (np.arange(wl, dtype=np.float32) / wl - 0.5)[None, :]
        M = np.zeros((h, wl), np.float32)
        for b in self.bands:
            env = min(1.0, b["age"] / 4.0) * min(1.0, (b["life"] - b["age"]) / 8.0)
            env *= (0.75 + 0.25 * self.rng.random()) * V["amp"] * inten
            d = (yy - (b["y"] + b["tilt"] * xx)) / (b["h"] * 0.5)
            M = np.maximum(M, np.exp(-np.abs(d) ** 3.0) * env)
        M = cv2.resize(M, (w, h), interpolation=cv2.INTER_LINEAR)
        return M * (0.68 + 0.32 * self.rng.random((h, 1))).astype(np.float32)     # bordes irregulares

    # ------------------------------------------------------------------ proceso
    def process(self, frame, ctx):
        h, w = frame.shape[:2]
        if self.shape != (h, w):
            self._init(h, w)
        rng = self.rng
        self.n += 1
        sc = w / 640.0

        # --- estado de la cinta
        modo = self.value("modo", ctx)
        if modo in FORZADO:
            st = FORZADO[modo]
        else:
            st = getattr(ctx, "transport", "play")
            st = st if st in VEC else "play"
        rate = 1.0 / max(1, self.value("transicion", ctx))
        if st in ("rewind", "ff"):
            rate = min(1.0, rate * 2.2)                              # al buscar, el ruido llega casi de inmediato
        for k, tv in VEC[st].items():
            self.vec[k] += (tv - self.vec[k]) * rate
        if self.prev_state is not None and st != self.prev_state:
            self.kick = float(rng.choice([-1.0, 1.0]) * rng.uniform(4.0, 9.0) * h / 360.0)   # «golpe» del mecanismo
            self.burst = 1.0
            self.flag_sign = float(rng.choice([-1.0, 1.0]))
            self.curl = float(rng.uniform(-0.5, 1.0))
        self.prev_state = st

        V = self.vec
        inten = self.value("intensidad", ctx)
        want = V["bands"] * self.value("bandas", ctx)
        self._update_bands(want, V, h)
        if not self.bands and abs(self.kick) < 0.25 and max(
                V[k] for k in ("amp", "jitter", "flag", "head", "color", "drop", "grain", "flicker", "vjit",
                               "soft", "field")) < 0.012:
            self.kick = 0.0
            return frame                                              # reproducción normal: imagen intacta

        M = self._band_mask(h, w, V, inten)
        Mrow = M.mean(axis=1) if M is not None else np.zeros(h, np.float32)
        jit = self.value("jitter", ctx)
        hd = self.value("cabezal", ctx)

        # --- geometría: temblor de líneas, flagging, curvatura inferior, saltos verticales
        amp_x = V["jitter"] * jit * inten * 2.0 * sc
        walk = rng.standard_normal(h).astype(np.float32)
        walk = cv2.GaussianBlur(walk.reshape(h, 1), (0, 0), sigmaX=0.3, sigmaY=2.5).reshape(h)
        walk /= (walk.std() + 1e-6)
        dx = walk * amp_x * (1.0 + 5.0 * Mrow) + rng.standard_normal(h).astype(np.float32) * amp_x * 0.8 * Mrow
        dx += (V["flag"] * jit * inten * 14.0 * sc) * np.exp(-self.yy / (0.04 * h)) * self.flag_sign
        wob = self.value("ondulacion", ctx)
        if wob > 0.01:                                               # la cinta «respira»: ondulación lenta de las líneas
            dx += wob * V["jitter"] * 1.4 * sc * np.sin(self.yy / h * 2 * np.pi * 1.6 + self.n * 0.07)
        hb = max(4, int(0.06 * h))
        ramp = np.clip((self.yy - (h - hb)) / hb, 0, 1) ** 2
        dx += ramp * (V["head"] * hd * inten * 42.0 * sc) * (self.curl + 0.35 * rng.standard_normal())
        if st in ("rewind", "ff") and rng.random() < 0.015 * V["vjit"]:
            self.roll = float(rng.uniform(0.02, 0.07) * h * rng.choice([-1, 1]))
        field = 0.45 * V["field"] * (1.0 if self.n % 2 else -1.0)
        dy = V["vjit"] * float(rng.normal(0, 0.55)) * sc + field + self.kick + self.roll
        dy = float(dy + wob * V["vjit"] * 1.2 * sc * np.sin(self.n * 0.09))
        self.kick *= 0.78
        self.roll *= 0.86
        img = frame
        if np.abs(dx).max() > 0.05 or abs(dy) > 0.05:
            img = cv2.remap(frame, self.X + dx[:, None], self.Y + dy, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)

        # --- filtro analógico: luma blanda, croma emborronado y desfasado, color que se va
        sl, scx, nm = CALIDAD[self.value("calidad", ctx)]
        soft = V["soft"]
        ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        Y = cv2.GaussianBlur(ycc[..., 0], (0, 0), sigmaX=max(0.3, sl * sc * (0.6 + 0.8 * soft)),
                             sigmaY=0.35 + 0.9 * soft * V["field"])
        ring = self.value("contornos", ctx)
        if ring > 0.01:
            Y = cv2.addWeighted(Y, 1.0 + 1.4 * ring, cv2.GaussianBlur(Y, (0, 0), 1.6 * sc), -1.4 * ring, 0)
        shift = int(round((2 + 2 * soft) * sc))
        chroma = []
        for c in (1, 2):
            ch = cv2.GaussianBlur(ycc[..., c].astype(np.float32), (0, 0),
                                  sigmaX=scx * sc * (1 + 0.5 * soft), sigmaY=1.0)
            chroma.append(np.roll(ch, shift, axis=1) - 128.0)
        cr, cb = chroma
        loss = float(np.clip(V["color"] * self.value("color", ctx) * (0.6 + 0.4 * rng.random()), 0, 0.95))
        sat = (1.0 - loss) * (1.0 - 0.85 * Mrow)[:, None]
        ang = float(rng.normal(0, 0.06)) * V["color"] * self.value("color", ctx)
        ca, sa = np.cos(ang), np.sin(ang)
        cr, cb = (ca * cr - sa * cb) * sat + 128.0, (sa * cr + ca * cb) * sat + 128.0
        out = cv2.cvtColor(cv2.merge([Y, np.clip(cr, 0, 255).astype(np.uint8),
                                      np.clip(cb, 0, 255).astype(np.uint8)]), cv2.COLOR_YCrCb2BGR)

        # --- bandas de nieve
        if M is not None:
            snow = self._crop(self.snow, h, w)
            luma = Y.astype(np.float32)
            gray = np.clip(108.0 + 76.0 * nm * snow, 0, 255)
            gray += np.where(snow > 2.1, 90.0, 0.0)                    # chispas brillantes
            band = 0.72 * gray + 0.28 * luma
            band3 = np.dstack([band * 1.05, band, band * 0.93])        # tinte frío de la nieve
            m = np.clip(M, 0, 0.93)[..., None]
            out = (out.astype(np.float32) * (1.0 - m) + band3 * m).astype(np.uint8)

        # --- motas blancas (dropouts) con cola hacia la derecha
        nd = int(min(40, rng.poisson(V["drop"] * self.value("blancos", ctx) * inten * 5.0)))
        for _ in range(nd):
            y = int(rng.integers(0, max(1, h - 2)))
            ln = max(6, int(rng.uniform(12, 140) * sc))
            x0 = int(rng.integers(-ln // 2, max(1, w - ln // 2)))
            xa, xb = max(0, x0), min(w, x0 + ln)
            if xb - xa < 2:
                continue
            tt = 1 if rng.random() < 0.75 else 2
            fade = (np.linspace(1.0, 0.0, xb - xa, dtype=np.float32) ** 1.4)[None, :, None] * 0.9
            tgt = 255.0 if rng.random() < 0.82 else 18.0
            seg = out[y:y + tt, xa:xb].astype(np.float32)
            out[y:y + tt, xa:xb] = (seg * (1 - fade) + tgt * fade).astype(np.uint8)

        # --- ruido del cabezal (borde inferior)
        hv = V["head"] * hd * inten
        if hv > 0.02:
            rows = slice(h - hb, h)
            snow = self._crop(self.snow, h, w)[rows]
            g = np.clip(104.0 + 72.0 * nm * snow, 0, 255)
            mh = (np.linspace(0.0, 1.0, hb, dtype=np.float32) ** 1.4 * 0.85 * min(1.0, hv))[:, None, None]
            gray3 = np.dstack([g * 1.05, g, g * 0.93])
            out[rows] = (out[rows].astype(np.float32) * (1 - mh) + gray3 * mh).astype(np.uint8)
            out[h - 2:] = (out[h - 2:] * 0.25).astype(np.uint8)

        # --- ganancia (AGC) con parpadeo, y grano
        self.gain += (1.0 - self.gain) * 0.25 + V["flicker"] * inten * 0.03 * float(rng.standard_normal())
        self.gain = float(np.clip(self.gain, 0.8, 1.2))
        out = cv2.convertScaleAbs(out, alpha=self.gain, beta=3.0 * V["flicker"] * inten)
        gs = V["grain"] * self.value("grano", ctx) * inten * 9.0 * nm
        if gs > 0.3:
            g = (self._crop(self.fine, h, w) * gs).astype(np.int16)
            out = np.clip(out.astype(np.int16) + g[..., None], 0, 255).astype(np.uint8)

        # --- indicador en pantalla (se dibuja encima, sin ruido, como lo haría la cámara)
        if self.value("osd", ctx) and st in OSD:
            txt = OSD[st]
            sz = max(0.5, 0.9 * sc)
            org = (int(0.05 * w), int(0.12 * h))
            cv2.putText(out, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sz, (0, 0, 0), max(3, int(4 * sc)), cv2.LINE_AA)
            cv2.putText(out, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sz, (255, 255, 255), max(1, int(2 * sc)), cv2.LINE_AA)
        marca = self.value("marca", ctx)
        if marca != "ninguna":
            out = self._marca(out, ctx, marca, sc)
        return out

    MESES = ["ENE", "FEB", "MAR", "ABR", "MAY", "JUN", "JUL", "AGO", "SEP", "OCT", "NOV", "DIC"]

    def _marca(self, out, ctx, marca, sc):
        """Texto típico de una videocámara: estado arriba a la izquierda y fecha y hora abajo."""
        h, w = out.shape[:2]
        t = self.t0 + datetime.timedelta(seconds=float(ctx.time))
        hora = f"{t.hour % 12 or 12}:{t.minute:02d}:{t.second:02d} {'PM' if t.hour >= 12 else 'AM'}"
        fecha = f"{self.MESES[t.month - 1]} {t.day} {t.year}"
        size = max(14, int(22 * sc))
        x0, y0 = int(0.05 * w), int(0.06 * h)
        items = []
        icon, label = ("●", "REC") if marca.startswith("REC") else ("▶", "PLAY")
        blink = marca.startswith("PLAY") or int(ctx.time * 1.5) % 2 == 0
        shadow = [(x0 + 2, int(0.84 * h) + 2, hora, (0, 0, 0), None), (x0 + 2, int(0.84 * h) + size + 4, fecha, (0, 0, 0), None)]
        if blink:
            shadow.append((x0 + 2, y0 + 2, f"{icon} {label}", (0, 0, 0), None))
        if blink:
            items.append((x0, y0, icon, (40, 40, 255) if icon == "●" else (255, 255, 255), None))
            items.append((x0 + int(size * 1.1), y0, label, (255, 255, 255), None))
        items += [(x0, int(0.84 * h), hora, (255, 255, 255), None), (x0, int(0.84 * h) + size + 2, fecha, (255, 255, 255), None)]
        return draw_texts(out, shadow + items, size)


@register
class AnalogCam(TapeNoise):
    name = "Cámara analógica (VHS / Hi8)"
    category = "Cinta analógica"
    OVERRIDES = {"modo": MODOS[4], "contornos": 0.35, "marca": "REC ● fecha y hora", "grano": 1.2, "ondulacion": 0.8}
