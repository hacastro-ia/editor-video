"""Parámetros, clase base de efectos y contexto de procesamiento."""
from __future__ import annotations

import copy

import numpy as np

# Señales que pueden modular cualquier parámetro numérico (matriz de modulación).
FUENTES = ["ninguna", "graves", "medios", "agudos", "volumen", "beat", "seno", "triángulo", "sierra"]
N_SPEC = 32


def zero_audio():
    return {"graves": 0.0, "medios": 0.0, "agudos": 0.0, "volumen": 0.0, "beat": 0.0,
            "spectrum": np.zeros(N_SPEC, np.float32)}


class Param:
    def __init__(self, key, label, kind, default, lo=0.0, hi=1.0, choices=None, mod=None):
        self.key = key
        self.label = label
        self.kind = kind            # float | int | bool | choice | text
        self.default = default
        self.value = default
        self.lo = lo
        self.hi = hi
        self.choices = list(choices) if choices else []
        self.mod_source, self.mod_amount = mod if mod else ("ninguna", 0.0)

    def clone(self):
        return copy.copy(self)


def F(key, label, default, lo, hi, mod=None):
    return Param(key, label, "float", default, lo, hi, mod=mod)


def I(key, label, default, lo, hi, mod=None):  # noqa: E743
    return Param(key, label, "int", default, lo, hi, mod=mod)


def B(key, label, default):
    return Param(key, label, "bool", default)


def C(key, label, choices, default=None):
    return Param(key, label, "choice", choices[0] if default is None else default, choices=choices)


def T(key, label, default=""):
    return Param(key, label, "text", default)


def Btn(key, label):
    """Botón de acción: la interfaz llama a effect.on_button(key)."""
    return Param(key, label, "button", None)


BLEND_MODES = ["normal", "sumar", "pantalla", "multiplicar", "superponer", "diferencia", "aclarar", "oscurecer"]


class Context:
    """Información que reciben los efectos en cada fotograma."""

    def __init__(self):
        self.frame_index = 0
        self.time = 0.0
        self.fps = 30.0
        self.audio = zero_audio()
        self.bpm = 120.0
        self.beats_per_cycle = 4.0
        self.faces = None           # FaceState compartido
        self.transport = "play"     # estado de la «cinta»: play | pause | rewind | ff

    def signal(self, name: str) -> float:
        if name in ("graves", "medios", "agudos", "volumen", "beat"):
            return float(self.audio.get(name, 0.0))
        cycle = max(self.bpm, 1.0) / 60.0 / max(self.beats_per_cycle, 0.25)
        ph = (self.time * cycle) % 1.0
        if name == "seno":
            return 0.5 - 0.5 * float(np.cos(2 * np.pi * ph))
        if name == "triángulo":
            return 1.0 - abs(2.0 * ph - 1.0)
        if name == "sierra":
            return ph
        return 0.0


class Effect:
    name = "Efecto"
    category = "Básicos"
    description = ""
    pin_first = False               # True: siempre va al inicio de la cadena
    PARAMS: list = []
    OVERRIDES: dict = {}            # valores por defecto distintos (subclases)
    DEFAULT_MODS: dict = {}         # modulaciones por defecto (subclases)
    NO_MIX = False                  # True: sin controles universales de mezcla (p. ej. la censura)
    RANDOM_SKIP = ("modelo", "carpeta", "clases")

    def __init__(self):
        self.enabled = True
        self.error = ""
        self.params = {p.key: p.clone() for p in self.PARAMS}
        if not self.NO_MIX:                       # controles universales, comunes a todos los efectos
            self.params["_mix"] = F("_mix", "Mezcla con la imagen anterior", 1.0, 0.0, 1.0)
            self.params["_blend"] = C("_blend", "Modo de mezcla", BLEND_MODES)
        for k, v in self.OVERRIDES.items():
            self.params[k].value = v
            self.params[k].default = v
        for k, (src, amt) in self.DEFAULT_MODS.items():
            self.params[k].mod_source = src
            self.params[k].mod_amount = amt
        self.reset()

    # -- estado temporal -------------------------------------------------
    def reset(self):
        """Se llama al abrir un video o saltar en la línea de tiempo."""

    # -- parámetros --------------------------------------------------------
    def value(self, key, ctx):
        p = self.params[key]
        v = p.value
        if p.kind in ("float", "int"):
            if p.mod_source != "ninguna" and p.mod_amount != 0:
                v = v + p.mod_amount * ctx.signal(p.mod_source) * (p.hi - p.lo)
                v = min(max(v, p.lo), p.hi)
            if p.kind == "int":
                v = int(round(v))
        return v

    def on_button(self, key):
        """Se llama al pulsar un parámetro de tipo botón (lo sobrescriben los efectos que los usan)."""

    def reset_params(self):
        for p in self.params.values():
            if p.kind != "button":
                p.value = p.default

    def randomize(self, rng, amount=0.5):
        """Aleatoriza los parámetros alrededor de su valor actual (sin tocar mezcla, texto ni botones)."""
        for k, p in self.params.items():
            if k.startswith("_") or k in self.RANDOM_SKIP or p.kind in ("button", "text"):
                continue
            if p.kind in ("float", "int"):
                span = (p.hi - p.lo) * amount
                v = float(np.clip(p.value + rng.normal(0, span * 0.5), p.lo, p.hi))
                p.value = int(round(v)) if p.kind == "int" else v
            elif p.kind == "choice":
                if rng.random() < amount:
                    p.value = p.choices[int(rng.integers(0, len(p.choices)))]
            elif p.kind == "bool":
                if rng.random() < amount * 0.3:
                    p.value = not p.value

    def set_value(self, key, v):
        if key in self.params:
            self.params[key].value = v

    def set_mod(self, key, src, amt):
        if key in self.params:
            self.params[key].mod_source = src
            self.params[key].mod_amount = amt

    def process(self, frame, ctx):
        raise NotImplementedError
