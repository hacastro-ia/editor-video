"""Visuales psicodélicos: caleidoscopio, mandala, túnel, vórtice, fluidos, mosaicos, color y ecos."""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from core import palettes
from core.noise import FlowNoise
from core.params import Effect, F, I, B, C
from core.pipeline import register
from core.utils import grid, grid_lo, hue_color, remap_lo, scale_saturation, shift_hue, tri


def _remap(frame, mx, my, border=cv2.BORDER_REFLECT_101):
    return cv2.remap(frame, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderMode=border)


def _screen(a, b):
    return 255 - cv2.multiply(255 - a, 255 - b, scale=1 / 255.0)


# --------------------------------------------------------------------------- caleidoscopio y mandala
@register
class Kaleidoscope(Effect):
    name = "Caleidoscopio"
    category = "Psicodélico"
    HELP = {"modo": {
        "imagen completa (polar)": "Toda la imagen se despliega dentro de cada segmento: siempre se ve rico en detalle.",
        "cono (clásico)": "Cada segmento muestra un cono de la imagen que sale del centro, como un caleidoscopio físico.",
    }}
    PARAMS = [C("modo", "Modo", ["imagen completa (polar)", "cono (clásico)"]),
              I("segmentos", "Segmentos", 8, 2, 32), F("rotacion", "Rotación (°/s)", 20, -180, 180),
              F("deriva", "Deriva de la imagen (°/s)", 10, -90, 90), F("zoom", "Zoom", 1.0, 0.3, 3.0),
              F("centro_x", "Centro X", 0.0, -0.5, 0.5), F("centro_y", "Centro Y", 0.0, -0.5, 0.5),
              F("orbita", "Órbita del centro", 0.0, 0.0, 0.3), F("torsion", "Torsión (remolino)", 0.0, -3.0, 3.0),
              F("ondulacion", "Ondulación radial", 0.0, 0.0, 0.3), F("anillos", "Anillos (0 = ninguno)", 0.0, 0, 8),
              F("color_segmento", "Cambio de color por segmento (°)", 0, 0, 60),
              F("retro", "Persistencia (estela)", 0.0, 0.0, 0.95)]

    def reset(self):
        self.prev = None

    def _layer(self, frame, ctx, seg_mult=1.0, rot_mult=1.0, zoom_mult=1.0):
        h, w = frame.shape[:2]
        X, Y = grid_lo(w, h)
        t = ctx.time
        orb = self.value("orbita", ctx)
        cx = w / 2 + self.value("centro_x", ctx) * w + orb * w * np.cos(t * 0.8)
        cy = h / 2 + self.value("centro_y", ctx) * h + orb * h * np.sin(t * 1.1)
        dx, dy = X - cx, Y - cy
        r = np.sqrt(dx * dx + dy * dy)
        th = np.arctan2(dy, dx)
        tw = self.value("torsion", ctx)
        if abs(tw) > 0.01:
            th = th + tw * r / (0.5 * min(w, h))
        seg = max(2, int(round(self.value("segmentos", ctx) * seg_mult)))
        sa = 2 * np.pi / seg
        raw = th + np.deg2rad(self.value("rotacion", ctx) * rot_mult * t)
        idx = np.floor(raw / sa)
        thf = np.mod(raw, sa)
        thf = np.where(thf > sa / 2, sa - thf, thf)
        rings = self.value("anillos", ctx)
        if rings > 0.05:
            period = min(w, h) / (2.0 * rings)
            r = tri(r / period) * period
        rip = self.value("ondulacion", ctx)
        if rip > 0.002:
            r = r * (1.0 + rip * np.sin(r / (0.07 * min(w, h)) - t * 3.0))
        zoom = max(self.value("zoom", ctx) * zoom_mult, 0.05)
        if self.value("modo", ctx).startswith("imagen"):
            u = r / (0.5 * np.hypot(w, h)) * zoom
            v = thf / (sa / 2) + self.value("deriva", ctx) * t / 180.0
            out = remap_lo(frame, tri(u) * (w - 1), tri(v) * (h - 1))
        else:
            th2 = thf + np.deg2rad(self.value("deriva", ctx) * t)
            r = r / zoom
            out = remap_lo(frame, w / 2 + r * np.cos(th2), h / 2 + r * np.sin(th2))
        return out, idx, seg

    def _color(self, out, idx, seg, ctx):
        cs = self.value("color_segmento", ctx)
        if cs < 0.5:
            return out
        h, w = out.shape[:2]
        add = cv2.resize(np.mod(idx, seg).astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) * (cs / 2.0)
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = np.mod(hsv[..., 0].astype(np.int16) + add.astype(np.int16), 180).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _finish(self, out, ctx):
        p = self.value("retro", ctx)
        if p > 0.01:
            if self.prev is not None and self.prev.shape == out.shape:
                out = cv2.addWeighted(out, 1.0 - p, self.prev, p, 0)
            self.prev = out.copy()
        return out

    def process(self, frame, ctx):
        out, idx, seg = self._layer(frame, ctx)
        return self._finish(self._color(out, idx, seg, ctx), ctx)


@register
class Mandala(Kaleidoscope):
    name = "Mandala"
    category = "Psicodélico"
    PARAMS = Kaleidoscope.PARAMS + [I("capas", "Capas", 2, 1, 3), F("contra_giro", "Contra-giro de las capas", 0.7, 0.0, 2.0)]
    OVERRIDES = {"segmentos": 12, "anillos": 3.0, "rotacion": 12, "deriva": 6, "color_segmento": 14}

    def process(self, frame, ctx):
        n, cg = self.value("capas", ctx), self.value("contra_giro", ctx)
        out, idx, seg = self._layer(frame, ctx)
        for k in range(1, n):
            layer, _, _ = self._layer(frame, ctx, seg_mult=1 + k, rot_mult=-cg if k % 2 else cg, zoom_mult=1.0 + 0.45 * k)
            out = _screen(cv2.convertScaleAbs(out, alpha=0.72), cv2.convertScaleAbs(layer, alpha=0.68))
        return self._finish(self._color(out, idx, seg, ctx), ctx)


# --------------------------------------------------------------------------- túnel y vórtice
@register
class Tunnel(Effect):
    name = "Túnel infinito"
    category = "Psicodélico"
    PARAMS = [F("velocidad", "Velocidad de avance", 0.6, -4, 4), F("profundidad", "Profundidad", 0.4, 0.05, 2.0),
              I("repeticiones", "Repeticiones angulares", 4, 1, 12), F("torsion", "Torsión (espiral)", 0.3, -2, 2),
              F("rotacion", "Rotación (°/s)", 0.0, -180, 180), F("bandas", "Bandas de luz", 0.0, 0.0, 12.0),
              F("brillo_centro", "Resplandor del centro", 0.2, 0.0, 1.0), F("oscuridad", "Oscuridad del centro", 1.0, 0.0, 1.0)]

    def process(self, frame, ctx):
        h, w = frame.shape[:2]
        X, Y = grid_lo(w, h)
        dx, dy = X - w / 2, Y - h / 2
        rn = np.sqrt(dx * dx + dy * dy) / (min(w, h) / 2) + 0.03
        th = np.arctan2(dy, dx) + np.deg2rad(self.value("rotacion", ctx) * ctx.time)
        prof = self.value("profundidad", ctx)
        v = prof / rn + ctx.time * self.value("velocidad", ctx)
        u = th / (2 * np.pi) * self.value("repeticiones", ctx) + self.value("torsion", ctx) * prof / rn * 0.15
        out = remap_lo(frame, tri(u) * (w - 1), tri(v) * (h - 1)).astype(np.float32)
        light = np.ones_like(rn)
        bands = self.value("bandas", ctx)
        if bands > 0.05:
            light *= 0.55 + 0.45 * np.sin(v * 2 * np.pi * bands)
        light *= 1.0 - self.value("oscuridad", ctx) * (1.0 - np.clip(rn * 3.0, 0, 1) ** 1.5)
        out *= cv2.resize(light, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
        glow = self.value("brillo_centro", ctx)
        if glow > 0.01:
            out += (cv2.resize(np.exp(-rn * 5.0), (w, h), interpolation=cv2.INTER_LINEAR) * 255 * glow)[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)


@register
class Vortex(Effect):
    name = "Vórtice (remolino)"
    category = "Psicodélico"
    PARAMS = [F("fuerza", "Fuerza del giro (rad)", 4.0, -12, 12, mod=("graves", 0.15)), F("radio", "Radio de influencia", 0.7, 0.1, 1.6),
              F("pellizco", "Pellizco / abultamiento", 0.0, -0.8, 0.8), F("giro", "Giro global (°/s)", 0.0, -180, 180),
              F("centro_x", "Centro X", 0.0, -0.5, 0.5), F("centro_y", "Centro Y", 0.0, -0.5, 0.5),
              F("orbita", "Órbita del centro", 0.0, 0.0, 0.3), F("cromatico", "Dispersión cromática", 0.0, 0.0, 0.6)]

    def process(self, frame, ctx):
        h, w = frame.shape[:2]
        X, Y = grid_lo(w, h)
        t, orb = ctx.time, self.value("orbita", ctx)
        cx = w / 2 + self.value("centro_x", ctx) * w + orb * w * np.cos(t)
        cy = h / 2 + self.value("centro_y", ctx) * h + orb * h * np.sin(t * 1.3)
        dx, dy = X - cx, Y - cy
        rn = np.sqrt(dx * dx + dy * dy) / (0.5 * min(w, h))
        fall = np.exp(-(rn / self.value("radio", ctx)) ** 2)
        k = 1.0 / (1.0 + self.value("pellizco", ctx) * fall)
        fuerza, giro = self.value("fuerza", ctx), np.deg2rad(self.value("giro", ctx) * t)
        crom = self.value("cromatico", ctx)

        def sample(img, mult):
            a = fuerza * mult * fall + giro
            c, s = np.cos(a), np.sin(a)
            return remap_lo(img, cx + (dx * c - dy * s) * k, cy + (dx * s + dy * c) * k)

        if crom < 0.01:
            return sample(frame, 1.0)
        b, g, r = cv2.split(frame)
        return cv2.merge([sample(b, 1.0 - crom), sample(g, 1.0), sample(r, 1.0 + crom)])


# --------------------------------------------------------------------------- fluidos
@register
class Liquid(Effect):
    name = "Líquido"
    category = "Psicodélico"
    PARAMS = [F("amplitud", "Amplitud", 0.02, 0.0, 0.1), F("frecuencia", "Frecuencia", 4, 1, 20),
              F("velocidad", "Velocidad", 0.5, 0, 5), F("radial", "Ondas radiales", 0.5, 0, 1),
              F("fase_rgb", "Desfase entre canales RGB", 0.0, 0.0, 1.5)]

    def process(self, frame, ctx):
        h, w = frame.shape[:2]
        X, Y = grid(w, h)
        amp = self.value("amplitud", ctx) * w
        f = self.value("frecuencia", ctx)
        rad = self.value("radial", ctx)
        dxc, dyc = X - w / 2, Y - h / 2
        r = np.sqrt(dxc * dxc + dyc * dyc) + 1e-3
        base = ctx.time * self.value("velocidad", ctx) * 2 * np.pi

        def field(phase):
            mx = X + amp * np.sin(2 * np.pi * f * Y / h + phase)
            my = Y + amp * np.cos(2 * np.pi * f * X / w + phase * 0.8)
            if rad > 0.01:
                rip = amp * rad * np.sin(2 * np.pi * f * r / min(w, h) - phase * 1.5)
                mx, my = mx + rip * dxc / r, my + rip * dyc / r
            return mx, my

        sep = self.value("fase_rgb", ctx)
        if sep < 0.01:
            return _remap(frame, *field(base))
        b, g, rr = cv2.split(frame)
        return cv2.merge([_remap(b, *field(base - sep)), _remap(g, *field(base)), _remap(rr, *field(base + sep))])


@register
class FluidWarp(Effect):
    name = "Warp de fluido (ruido)"
    category = "Psicodélico"
    PARAMS = [F("fuerza", "Fuerza", 0.5, 0.0, 2.0, mod=("graves", 0.4)), F("escala", "Escala del ruido", 2.5, 0.5, 10.0),
              I("octavas", "Detalle (octavas)", 3, 1, 5), F("velocidad", "Velocidad", 0.3, 0.0, 2.0),
              I("iteraciones", "Pasadas de deformación", 2, 1, 3), F("cromatico", "Separación cromática", 0.0, 0.0, 1.0),
              B("remolino", "Campo en remolino (curl)", True)]

    def reset(self):
        self.noise = FlowNoise(11)

    def process(self, frame, ctx):
        h, w = frame.shape[:2]
        X, Y = grid(w, h)
        sc, oc, sp = self.value("escala", ctx), self.value("octavas", ctx), self.value("velocidad", ctx)
        amp = self.value("fuerza", ctx) * w * 0.07
        crom = self.value("cromatico", ctx)
        t = ctx.time
        fields = []
        for k in range(self.value("iteraciones", ctx)):
            nx = self.noise.field(h, w, t + k * 7.3, sc * (1 + 0.5 * k), oc, sp, ch=0)
            ny = self.noise.field(h, w, t + k * 7.3, sc * (1 + 0.5 * k), oc, sp, ch=1)
            if self.value("remolino", ctx):                              # campo sin divergencia: fluye sin apelotonarse
                gx = cv2.Sobel(ny, cv2.CV_32F, 1, 0, ksize=3) * 6.0
                gy = -cv2.Sobel(nx, cv2.CV_32F, 0, 1, ksize=3) * 6.0
                nx, ny = nx * 0.5 + gx, ny * 0.5 + gy
            fields.append((nx, ny))

        def warp(img, mult):
            mx, my = X, Y
            for nx, ny in fields:
                mx, my = mx + nx * amp * mult, my + ny * amp * mult
            return _remap(img, mx, my)

        if crom < 0.01:
            return warp(frame, 1.0)
        b, g, r = cv2.split(frame)
        return cv2.merge([warp(b, 1.0 - 0.6 * crom), warp(g, 1.0), warp(r, 1.0 + 0.6 * crom)])


@register
class Plasma(Effect):
    name = "Plasma (generador)"
    category = "Psicodélico"
    OVERRIDES = {"_blend": "pantalla", "_mix": 0.8}
    PARAMS = [F("escala", "Escala", 2.2, 0.5, 8.0, mod=("graves", 0.2)), F("velocidad", "Velocidad", 0.35, 0.0, 2.0),
              I("complejidad", "Complejidad", 3, 1, 5), F("distorsion", "Distorsión interna", 0.8, 0.0, 2.0),
              F("contraste", "Contraste", 1.3, 0.4, 3.0), I("bandas", "Bandas de color", 3, 1, 12),
              C("paleta", "Paleta", palettes.NAMES, "vaporwave"), F("ciclo", "Ciclar la paleta (vueltas/s)", 0.08, 0.0, 1.0),
              F("brillo", "Brillo", 1.0, 0.1, 2.0, mod=("volumen", 0.4)), B("sobre_imagen", "Modular con la imagen", False)]

    def reset(self):
        self.noise = FlowNoise(5)

    def process(self, frame, ctx):
        h, w = frame.shape[:2]
        t = ctx.time
        sc, oc, sp = self.value("escala", ctx), self.value("complejidad", ctx), self.value("velocidad", ctx)
        a = self.noise.field(h, w, t, sc, oc, sp, ch=0, lowres=3)
        b = self.noise.field(h, w, t + 31.0, sc, oc, sp, ch=1, lowres=3)
        v = a + self.value("distorsion", ctx) * np.sin(b * 2.2 + t * 0.5)
        v = 0.5 + 0.5 * np.sin(v * np.pi * self.value("bandas", ctx) * 0.5)
        v = np.clip((v - 0.5) * self.value("contraste", ctx) + 0.5, 0, 1)
        if self.value("sobre_imagen", ctx):
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            v = np.mod(v + g * 0.6, 1.0)
        img = palettes.apply((v * 255).astype(np.uint8), self.value("paleta", ctx), t * self.value("ciclo", ctx), cyclic=True)
        return cv2.convertScaleAbs(img, alpha=self.value("brillo", ctx))


# --------------------------------------------------------------------------- mosaico, color, ecos, flash
@register
class MirrorTiles(Effect):
    name = "Mosaico espejo"
    category = "Psicodélico"
    PARAMS = [I("teselas", "Teselas por lado", 3, 1, 12), F("scroll_x", "Desplazamiento X", 0.2, -2, 2),
              F("scroll_y", "Desplazamiento Y", 0.1, -2, 2), F("rotacion", "Rotación (°)", 0.0, -180, 180),
              F("giro", "Giro automático (°/s)", 0.0, -90, 90), F("ladrillo", "Desfase tipo ladrillo", 0.0, 0.0, 1.0),
              B("espejo", "Reflejar teselas alternas", True)]

    def process(self, frame, ctx):
        h, w = frame.shape[:2]
        X, Y = grid_lo(w, h)
        a = np.deg2rad(self.value("rotacion", ctx) + self.value("giro", ctx) * ctx.time)
        dx, dy = X - w / 2, Y - h / 2
        xr = dx * np.cos(a) - dy * np.sin(a) + w / 2
        yr = dx * np.sin(a) + dy * np.cos(a) + h / 2
        n = self.value("teselas", ctx)
        u = xr / w * n + self.value("scroll_x", ctx) * ctx.time
        v = yr / h * n + self.value("scroll_y", ctx) * ctx.time
        br = self.value("ladrillo", ctx)
        if br > 0.01:
            u = u + (np.floor(v) % 2) * br * 0.5
        if self.value("espejo", ctx):
            mx, my = tri(u) * (w - 1), tri(v) * (h - 1)
        else:
            mx, my = (u % 1.0) * (w - 1), (v % 1.0) * (h - 1)
        return remap_lo(frame, mx, my) if self.value("espejo", ctx) else _remap(frame, cv2.resize(mx, (w, h)), cv2.resize(my, (w, h)))


@register
class HueCycle(Effect):
    name = "Ciclo de color"
    category = "Psicodélico"
    HELP = {"modo": {
        "arcoíris (rotar el matiz)": "El matiz de toda la imagen gira continuamente.",
        "mapa de degradado": "La luminosidad se convierte en una paleta que se desplaza (estilo TouchDesigner Lookup).",
        "neón": "Matiz rotando con saturación y contraste altos.",
        "duotono": "Dos colores que giran por el círculo cromático.",
        "solarizar": "Invierte las zonas claras y rota el color.",
    }}
    PARAMS = [C("modo", "Modo", list(HELP["modo"].keys()), "mapa de degradado"),
              C("paleta", "Paleta (mapa de degradado)", palettes.NAMES, "vaporwave"),
              F("velocidad", "Velocidad (°/s)", 60, 0, 360), F("saturacion", "Saturación", 1.5, 0, 3),
              F("contraste", "Contraste", 1.0, 0.4, 2.5), F("umbral", "Umbral de solarizado", 128, 0, 255)]

    def process(self, frame, ctx):
        modo = self.value("modo", ctx)
        hue = self.value("velocidad", ctx) * ctx.time
        sat = self.value("saturacion", ctx)
        cst = self.value("contraste", ctx)
        if abs(cst - 1) > 0.02:
            frame = cv2.convertScaleAbs(frame, alpha=cst, beta=(1 - cst) * 100)
        if modo == "mapa de degradado":
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return palettes.apply(g, self.value("paleta", ctx), hue / 360.0, cyclic=True)
        if modo == "duotono":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            a, b = hue_color(hue, 0.9, 0.15), hue_color(hue + 140, 0.85, 1.0)
            return np.linspace(a, b, 256).astype(np.uint8)[gray]
        if modo == "solarizar":
            thr = self.value("umbral", ctx)
            frame = np.where(frame > thr, 255 - frame, frame).astype(np.uint8)
        out = scale_saturation(shift_hue(frame, hue), sat)
        return cv2.convertScaleAbs(out, alpha=1.3, beta=-30) if modo == "neón" else out


@register
class Echoes(Effect):
    name = "Ecos de color"
    category = "Psicodélico"
    PARAMS = [I("ecos", "Cantidad de ecos", 3, 1, 8), I("retardo", "Retardo (fotogramas)", 4, 1, 15),
              F("matiz", "Desfase de color por eco (°)", 25, 0, 90), F("mezcla", "Intensidad", 0.6, 0.2, 1.0),
              F("escala", "Escala por eco", 1.0, 0.9, 1.1), F("rotacion", "Rotación por eco (°)", 0.0, -20, 20),
              C("modo", "Modo", ["fantasma", "aclarar", "estela de color"]),
              C("paleta", "Paleta (estela de color)", palettes.NAMES, "vaporwave")]

    def reset(self):
        self.hist = deque(maxlen=160)

    def process(self, frame, ctx):
        h, w = frame.shape[:2]
        if self.hist and self.hist[-1].shape != frame.shape:
            self.hist.clear()
        self.hist.append(frame.copy())
        n, delay = self.value("ecos", ctx), self.value("retardo", ctx)
        wgt, hue = self.value("mezcla", ctx), self.value("matiz", ctx)
        sc, rot = self.value("escala", ctx), self.value("rotacion", ctx)
        modo = self.value("modo", ctx)
        pal = palettes.lut(self.value("paleta", ctx))
        used = [k for k in range(1, n + 1) if len(self.hist) - 1 - k * delay >= 0]
        if not used:
            return frame
        weights = {k: wgt ** k for k in used}
        total = 1.0 + sum(weights.values()) if modo == "fantasma" else 1.0
        acc = cv2.convertScaleAbs(frame, alpha=1.0 / total) if modo == "fantasma" else frame.copy()
        for k in used:
            past = self.hist[len(self.hist) - 1 - k * delay]
            if abs(sc - 1) > 0.002 or abs(rot) > 0.05:
                m = cv2.getRotationMatrix2D((w / 2, h / 2), rot * k, sc ** k)
                past = cv2.warpAffine(past, m, (w, h), borderMode=cv2.BORDER_REFLECT)
            if modo == "estela de color":
                tint = pal[int(255 * k / (n + 1))].astype(np.float32) / 255.0 * 1.1
                gray3 = cv2.cvtColor(cv2.cvtColor(past, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                past = cv2.addWeighted(past, 0.35, cv2.multiply(gray3, (float(tint[0]), float(tint[1]), float(tint[2]), 0.0)), 1.0, 0)
            else:
                past = shift_hue(past, hue * k)
            if modo == "fantasma":
                acc = cv2.addWeighted(acc, 1.0, past, weights[k] / total, 0)
            else:
                acc = np.maximum(acc, cv2.convertScaleAbs(past, alpha=weights[k]))
        return acc


@register
class Strobe(Effect):
    name = "Estroboscopio / Flash"
    category = "Psicodélico"
    PARAMS = [C("disparo", "Disparo", ["beat del audio", "BPM del programa", "frecuencia fija"]),
              F("frecuencia", "Frecuencia (Hz, modo fijo)", 4.0, 0.5, 20.0), I("beats", "Beats entre flashes (modo BPM)", 1, 1, 8),
              C("tipo", "Tipo de flash", ["blanco", "negro", "invertir", "color de la paleta"]),
              C("paleta", "Paleta", palettes.NAMES, "neón rosa y cian"),
              F("intensidad", "Intensidad", 0.8, 0.0, 1.0), F("decaimiento", "Duración del destello", 3.0, 0.5, 10.0)]

    def process(self, frame, ctx):
        modo = self.value("disparo", ctx)
        if modo == "beat del audio":
            f = float(ctx.audio.get("beat", 0.0))
        else:
            rate = self.value("frecuencia", ctx) if modo == "frecuencia fija" else ctx.bpm / 60.0 / self.value("beats", ctx)
            f = (1.0 - (ctx.time * rate) % 1.0) ** self.value("decaimiento", ctx)
        f *= self.value("intensidad", ctx)
        if f < 0.01:
            return frame
        tipo = self.value("tipo", ctx)
        if tipo == "negro":
            return cv2.convertScaleAbs(frame, alpha=1.0 - f)
        if tipo == "invertir":
            return cv2.addWeighted(frame, 1.0 - f, 255 - frame, f, 0)
        col = np.full_like(frame, 255)
        if tipo == "color de la paleta":
            col[:] = palettes.lut(self.value("paleta", ctx), 256, 0, True)[int((ctx.time * 40) % 256)]
        return cv2.addWeighted(frame, 1.0 - f, col, f, 0)
