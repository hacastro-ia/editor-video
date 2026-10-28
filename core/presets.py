"""Combinaciones de efectos listas para usar."""
from __future__ import annotations

from core.pipeline import REGISTRY

# (efecto, {parámetro: valor}, {parámetro: (señal, cantidad)})
PRESETS = {
    "Mandala reactivo": [
        ("Ciclo de color", {"paleta": "neón"}, {}),
        ("Mandala", {}, {"zoom": ("graves", 0.25)}),
        ("Audio: Pulso de zoom", {"zoom": 1.0}, {}),
    ],
    "Túnel infinito": [
        ("Túnel infinito", {}, {"velocidad": ("graves", 0.15)}),
        ("Ciclo de color", {"paleta": "arcoíris"}, {}),
        ("Audio: Glitch (beat)", {}, {}),
    ],
    "Líquido": [
        ("Líquido", {}, {"amplitud": ("graves", 0.3)}),
        ("Ciclo de color", {"paleta": "neón", "velocidad": 40}, {}),
        ("Ecos de color", {}, {}),
    ],
    "Viaje total": [
        ("Caleidoscopio", {"segmentos": 6}, {"segmentos": ("beat", 0.25)}),
        ("Fractal (Droste)", {}, {}),
        ("Líquido", {"amplitud": 0.012}, {}),
        ("Ciclo de color", {"paleta": "arcoíris"}, {}),
    ],
    "Nube de puntos": [("Nube de puntos", {}, {})],
    "Datamosh + glitch": [
        ("Datamosh: Pixel-bending", {"iframe_beat": True}, {}),
        ("Audio: Glitch (beat)", {}, {}),
    ],
    "Personas: esqueleto neón": [("Personas: puntos y esqueleto (YOLO)", {"modo": "esqueleto neón"}, {})],
    "Personas: radiografía": [("Personas: puntos y esqueleto (YOLO)", {"modo": "radiografía", "fondo": 0.5}, {})],
    "Personas: relámpagos": [("Personas: puntos y esqueleto (YOLO)", {"modo": "relámpagos", "fondo": 0.15}, {})],
    "Personas: constelación": [("Personas: puntos y esqueleto (YOLO)", {"modo": "constelación", "fondo": 0.1}, {})],
    "Personas: multibrazos": [("Personas: puntos y esqueleto (YOLO)", {"modo": "multibrazos", "fondo": 0.1}, {}),
                              ("Ciclo de color", {"paleta": "neón", "velocidad": 30}, {})],
    "Personas de puntos (silueta)": [("Personas: siluetas (YOLO seg)", {"modo": "persona de puntos"}, {})],
    "Personas: desintegración": [("Personas: siluetas (YOLO seg)", {"modo": "desintegración"}, {})],
    "Biblioteca: fotomosaico": [("Biblioteca de imágenes (objetos)", {"modo": "fotomosaico"}, {})],
    "Biblioteca: fichas flotantes": [("Biblioteca de imágenes (objetos)", {"modo": "fichas flotantes"}, {})],
    "Biblioteca: órbita": [("Biblioteca de imágenes (objetos)", {"modo": "órbita"}, {})],
    "Traqueo de puntos (sin YOLO)": [("Traqueo de puntos (HUD)", {"modo": "HUD de coordenadas"}, {})],
    "Cinta 8mm: pausa y rebobinado (automático)": [("Cinta 8mm: ruido de pausa y rebobinado", {}, {})],
    "Cinta 8mm: con indicador en pantalla": [("Cinta 8mm: ruido de pausa y rebobinado", {"osd": True}, {})],
    "Cinta 8mm: reproducción de cinta gastada": [
        ("Cinta 8mm: ruido de pausa y rebobinado", {"modo": "reproducción (cinta gastada)"}, {})],
    "Datamosh fluido + bloom": [
        ("Datamosh: Pixel-bending", {"modo": "fluid (suave y persistente)", "fuerza": 1.4, "refresco": 0.015}, {}),
        ("Bloom (resplandor)", {"intensidad": 0.9}, {})],
    "Datamosh en bloques": [
        ("Datamosh: Pixel-bending", {"modo": "pack (bloques de códec)", "ruido": 0.25, "iframe_beat": True}, {}),
        ("Glitch RGB", {"intensidad": 0.25, "modo": "bloques"}, {"intensidad": ("beat", 0.5)})],
    "Mandala de neón": [
        ("Mandala", {"capas": 2, "color_segmento": 22}, {"rotacion": ("medios", 0.1)}),
        ("Bloom (resplandor)", {"intensidad": 0.5, "umbral": 0.8}, {"intensidad": ("graves", 0.4)}),
        ("Lente cinematográfica", {"aberracion": 0.5, "vineta": 0.5}, {})],
    "Fluido ácido": [
        ("Warp de fluido (ruido)", {"fuerza": 0.9, "cromatico": 0.6}, {"fuerza": ("graves", 0.5)}),
        ("Ciclo de color", {"modo": "mapa de degradado", "paleta": "ácido"}, {}),
        ("Bloom (resplandor)", {}, {})],
    "Plasma + caleidoscopio": [
        ("Plasma (generador)", {"_mix": 1.0, "_blend": "normal", "paleta": "neón rosa y cian"}, {}),
        ("Caleidoscopio", {"segmentos": 10, "color_segmento": 18, "torsion": 0.5}, {"segmentos": ("beat", 0.2)})],
    "Túnel con flash": [
        ("Túnel infinito", {"bandas": 4, "brillo_centro": 0.5}, {"velocidad": ("graves", 0.25)}),
        ("Ciclo de color", {"modo": "neón"}, {}),
        ("Estroboscopio / Flash", {"disparo": "beat del audio", "tipo": "invertir", "intensidad": 0.5}, {})],
    "Time machine": [("Slit-scan (Time Machine)", {"onda": 0.5}, {}), ("Lente cinematográfica", {}, {})],
    "Pop art (halftone)": [("Halftone (retícula)", {"modo": "tintas CMY (imprenta)", "tamano": 9}, {"tamano": ("graves", 0.2)}),
                           ("Ajuste de color", {"saturacion": 1.4}, {})],
    "CRT: TV de tubo": [("Pantalla CRT", {}, {})],
    "CRT: terminal verde": [("Pantalla CRT", {"fosforo": "verde fósforo", "lineas_pantalla": 300, "resplandor": 0.9, "persistencia": 0.45,
                                                  "mascara": "ninguna", "convergencia": 0.3, "barra": 0.35, "curvatura": 0.3}, {})],
    "CRT: arcade": [("Pantalla CRT", {"lineas": 0.7, "lineas_pantalla": 224, "mascara": "rejilla de apertura (RGB)", "mascara_fuerza": 0.45,
                                          "resplandor": 0.8, "saturacion": 1.35, "brillo": 1.2}, {})],
    "Cámara analógica con CRT": [("Cámara analógica (VHS / Hi8)", {}, {}), ("Pantalla CRT", {"curvatura": 0.2, "lineas": 0.3}, {})],
    "Privacidad: censurar todas las caras": [("Censura de caras", {"modo": "todas"}, {})],
    "Detección de objetos": [("Detección de objetos (YOLO)", {}, {})],
}


def build_preset(name):
    effects = []
    for ename, params, mods in PRESETS[name]:
        e = REGISTRY[ename]()
        for k, v in params.items():
            e.set_value(k, v)
        for k, (src, amt) in mods.items():
            e.set_mod(k, src, amt)
        effects.append(e)
    return effects
