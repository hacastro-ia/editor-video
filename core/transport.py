"""Transporte de cinta: guion de pausas / rebobinados / avances y lectura de fotogramas con saltos."""
from __future__ import annotations

import cv2

STATES = ("play", "pause", "rewind", "ff")


def build_plan(events, total, fps, hold=0.6):
    """Convierte un guion de eventos en la lista de fotogramas de salida: [(índice_de_origen, estado)].

    events: [{"tipo": "pausa"|"rebobinar"|"avance", "frame": n, "valor": segundos, "vel": n}]
      · pausa:     el fotograma se congela `valor` segundos y luego sigue la reproducción.
      · rebobinar: retrocede `valor` segundos de video a `vel`× (se oye silencio), se detiene `hold` s y
                   vuelve a reproducir desde ahí (lo rebobinado se vuelve a ver, como en una cinta real).
      · avance:    salta hacia delante `valor` segundos a `vel`×, se detiene `hold` s y sigue reproduciendo.
    """
    total = max(1, int(total))
    hold_n = max(1, int(round(hold * fps)))
    plan = []
    cur = 0

    def play(a, b):
        for i in range(a, b):
            plan.append((i, "play"))

    for ev in sorted(events, key=lambda e: e["frame"]):
        p = int(min(max(ev["frame"], cur), total - 1))
        play(cur, p)
        n = max(1, int(round(float(ev.get("valor", 1.0)) * fps)))
        step = max(1, int(ev.get("vel", 8)))
        tipo = ev["tipo"]
        if tipo == "pausa":
            plan.extend([(p, "pause")] * n)
            cur = p
        elif tipo == "rebobinar":
            target = max(0, p - n)
            i = p
            while i > target:
                plan.append((i, "rewind"))
                i -= step
            plan.append((target, "rewind"))
            plan.extend([(target, "pause")] * hold_n)
            cur = target
        elif tipo == "avance":
            target = min(total - 1, p + n)
            i = p
            while i < target:
                plan.append((i, "ff"))
                i += step
            plan.append((target, "ff"))
            plan.extend([(target, "pause")] * hold_n)
            cur = target
    play(cur, total)
    return plan


class FrameFetcher:
    """Lee fotogramas por índice (adelante, atrás o repetidos) minimizando los saltos de búsqueda."""

    def __init__(self, cap):
        self.cap = cap
        self.pos = None          # índice del próximo fotograma que devolvería cap.read()
        self.last = None
        self.last_idx = None

    def get(self, idx):
        idx = max(0, int(idx))
        if idx == self.last_idx and self.last is not None:
            return self.last.copy()
        if self.pos is None or idx < self.pos or idx - self.pos > 12:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self.pos = idx
        else:
            while self.pos < idx:
                self.cap.grab()
                self.pos += 1
        ok, fr = self.cap.read()
        if not ok:
            return None if self.last is None else self.last.copy()
        self.pos = idx + 1
        self.last, self.last_idx = fr, idx
        return fr.copy()
