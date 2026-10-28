"""Tema visual: paleta sobria y minimalista, tipografías y estilo de los widgets ttk."""
from __future__ import annotations

import sys
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

BG = "#0b0b0e"          # ventana
PANEL = "#111115"       # paneles laterales
CARD = "#17171c"        # tarjetas y campos
CARD_HOVER = "#1f1f26"
BORDER = "#26262e"
TEXT = "#e8e8ee"
MUTED = "#7d7d8a"
FAINT = "#4a4a55"
ACCENT = "#7c6cff"      # único color de acento
ACCENT_HOVER = "#9486ff"
ACCENT_DIM = "#2a2550"
GOOD = "#35d0ba"
WARN = "#ffb454"
DANGER = "#ff6b7a"

FAMILY = "TkDefaultFont"
FONT = (FAMILY, 10)
FONT_SMALL = (FAMILY, 9)
FONT_BOLD = (FAMILY, 10, "bold")
FONT_TITLE = (FAMILY, 12, "bold")
FONT_MONO = ("Courier", 9)


def init_fonts(root):
    """Elige la mejor tipografía disponible en el sistema."""
    global FAMILY, FONT, FONT_SMALL, FONT_BOLD, FONT_TITLE, FONT_MONO
    available = set(tkfont.families(root))
    for fam in ("Inter", "Segoe UI Variable Text", "Segoe UI", "SF Pro Text", "Helvetica Neue", "Noto Sans", "Ubuntu",
                "DejaVu Sans"):
        if fam in available:
            FAMILY = fam
            break
    else:
        FAMILY = tkfont.nametofont("TkDefaultFont").actual("family")
    size = 10 if not sys.platform == "darwin" else 12
    FONT = (FAMILY, size)
    FONT_SMALL = (FAMILY, size - 1)
    FONT_BOLD = (FAMILY, size, "bold")
    FONT_TITLE = (FAMILY, size + 2, "bold")
    for fam in ("JetBrains Mono", "Cascadia Mono", "Consolas", "Menlo", "DejaVu Sans Mono"):
        if fam in available:
            FONT_MONO = (fam, size - 1)
            break
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
        try:
            tkfont.nametofont(name).configure(family=FAMILY, size=size)
        except tk.TclError:
            pass


def apply_style(root):
    s = ttk.Style(root)
    s.theme_use("clam")
    s.configure(".", background=PANEL, foreground=TEXT, fieldbackground=CARD, bordercolor=BORDER,
                lightcolor=PANEL, darkcolor=PANEL, troughcolor=CARD, focuscolor=PANEL, font=FONT)
    s.configure("TFrame", background=PANEL)
    s.configure("TLabel", background=PANEL, foreground=TEXT)
    s.configure("Muted.TLabel", foreground=MUTED)
    s.configure("TCombobox", fieldbackground=CARD, background=CARD, foreground=TEXT, arrowcolor=MUTED, bordercolor=BORDER,
                lightcolor=CARD, darkcolor=CARD, padding=(6, 4), selectbackground=CARD, selectforeground=TEXT)
    s.map("TCombobox", fieldbackground=[("readonly", CARD), ("focus", CARD)], foreground=[("readonly", TEXT)],
          background=[("active", CARD_HOVER), ("readonly", CARD)], bordercolor=[("focus", ACCENT)],
          arrowcolor=[("active", TEXT)], selectbackground=[("readonly", CARD)], selectforeground=[("readonly", TEXT)])
    s.configure("TEntry", fieldbackground=CARD, foreground=TEXT, insertcolor=TEXT, bordercolor=BORDER,
                lightcolor=CARD, darkcolor=CARD, padding=(6, 4))
    s.map("TEntry", bordercolor=[("focus", ACCENT)])
    s.configure("TSpinbox", fieldbackground=CARD, foreground=TEXT, arrowcolor=MUTED, bordercolor=BORDER,
                lightcolor=CARD, darkcolor=CARD, padding=(6, 3), background=CARD)
    s.configure("Vertical.TScrollbar", background=CARD, troughcolor=PANEL, bordercolor=PANEL, arrowcolor=PANEL,
                lightcolor=CARD, darkcolor=CARD, gripcount=0, width=8)
    s.map("Vertical.TScrollbar", background=[("active", FAINT)])
    s.configure("TCheckbutton", background=PANEL, foreground=TEXT)
    root.option_add("*TCombobox*Listbox.background", CARD)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
    root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
    root.option_add("*TCombobox*Listbox.font", FONT)
    root.option_add("*Menu.background", CARD)
    root.option_add("*Menu.foreground", TEXT)
    root.option_add("*Menu.activeBackground", ACCENT)
    root.option_add("*Menu.activeForeground", "#ffffff")
    root.option_add("*Menu.borderWidth", 0)
    root.option_add("*Menu.font", FONT)
